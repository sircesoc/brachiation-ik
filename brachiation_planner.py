"""
Brachiation trajectory planner — swing from bar 1 to bar 2.

Uses Sequential Quadratic Programming (scipy SLSQP) to find a sequence of
joint-angle waypoints that move the free hand from bar 1 to bar 2 while
keeping the whole-body COM directly under the gripped bar.

The robot hangs from one gripper (fixed at bar 1).  The gripper is held
vertical by a parallelogram linkage, so the wrist-to-bar vector is always
+y.  That means the shoulder position is fully determined by the gripping
arm's joint angles and the bar position:

    shoulder_grip = bar_pos - FK_hand_local(grip_arm) - (0, GRIPPER_LEN)

Once the gripping shoulder is known, the body center and the free shoulder
follow from the body geometry.  The free hand position is then FK from the
free shoulder.

COM is computed with the *original, unmodified* COM_Prediction() function
by converting the planned angles into its expected naming convention.
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# Import COM model (still used for mass-based COM estimation)
from InvserseKinematic_COM_Combined import (
    TAIL_ANGLE_STOP_DEG,
    TAIL_GUI_OFFSET_DEG,
    TAIL_PIVOT_Y,
    TAIL_LENGTH,
    COM_Prediction,
    INCH_TO_M,
)

# Import Willy IK — the real arm kinematics
import willy_ik

# ── Willy arm geometry (from willy_ik.py / context.md) ───────────────────────
A_CM  = willy_ik.A              # 9.778 cm upper arm
L_CM  = willy_ik.L              # 36.786 cm forearm + wrist
K1    = willy_ik.K1             # +27.963 deg shoulder offset
K2    = willy_ik.K2             # -31.965 deg elbow offset

# Convert to metres for the planner
A_M   = A_CM / 100.0           # 0.09778 m
L_M   = L_CM / 100.0           # 0.36786 m

# Bilateral arm spacing: 11.7055 in = 29.7320 cm between M1 pivots.
# Each arm's M1 sits at +/- HALF_BASE_SPACING from body centre.
HALF_BASE_SPACING_CM = 11.7055 * 2.54 / 2.0   # 14.866 cm
BODY_PIVOT_OFFSET    = HALF_BASE_SPACING_CM / 100.0   # m

# Gripper: 6 cm vertical from wrist plate midpoint (from visualizer).
# The double parallelogram keeps the wrist plate parallel to the base,
# so the gripper always points straight up (+v direction).
GRIPPER_DRAW_LENGTH = 0.06   # 6 cm in metres

# Default link lengths for the planner (overrides old 3.4" / 11.8")
LINK1_LENGTH = A_M
LINK2_LENGTH = L_M

# ── Bar geometry ─────────────────────────────────────────────────────────────
BAR_SPACING = 12.0 * INCH_TO_M          # 12 in -> metres
BAR_Y       = (A_M + L_M) + GRIPPER_DRAW_LENGTH + 0.02
# Bars positioned so that bar 1 is at x=0 (body-centered) and bar 2 is to
# the right (+x in plot frame).


# ── Willy FK / IK wrappers (work in metres, return numpy arrays) ─────────────

_D2R = math.pi / 180.0
_R2D = 180.0 / math.pi


def willy_fk_m(theta1_deg, theta2_deg):
    """Willy FK: motor angles (deg) → end-effector (u, v) in metres."""
    u_cm, v_cm = willy_ik.fk(theta1_deg, theta2_deg)
    return np.array([u_cm / 100.0, v_cm / 100.0])


def willy_fk_joints_m(theta1_deg, theta2_deg):
    """Willy FK: return (shoulder, elbow, hand) positions in metres.

    Uses the same angle convention as willy_ik.py:
        alpha1 = theta1 + K1,  alpha2 = theta2 + K2
    """
    alpha1 = (theta1_deg + K1) * _D2R
    alpha2 = (theta2_deg + K2) * _D2R
    alpha12 = alpha1 + alpha2
    shoulder = np.array([0.0, 0.0])
    elbow = np.array([A_M * math.cos(alpha1), A_M * math.sin(alpha1)])
    hand = np.array([
        A_M * math.cos(alpha1) + L_M * math.cos(alpha12),
        A_M * math.sin(alpha1) + L_M * math.sin(alpha12),
    ])
    return shoulder, elbow, hand


def willy_ik_m(u_m, v_m, elbow='up'):
    """Willy IK: target (u, v) in metres → motor angles (deg)."""
    u_cm = u_m * 100.0
    v_cm = v_m * 100.0
    return willy_ik.ik(u_cm, v_cm, elbow=elbow)


# ── Collision detection (optional) ────────────────────────────────────────────

_collision_model = None

def get_collision_model():
    """Lazy-load the collision model (heavy — only load once)."""
    global _collision_model
    if _collision_model is not None:
        return _collision_model
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'files'))
        from willy_collision import ArmCollisionModel
        mesh_dir = os.path.join(os.path.dirname(__file__), 'willy_meshes')
        _collision_model = ArmCollisionModel(
            mesh_dir, use_convex_hulls=True, include_tail=True)
        print(f"[collision] Model loaded from {mesh_dir}")
    except (ImportError, FileNotFoundError) as e:
        print(f"[collision] Could not load: {e}")
        _collision_model = None
    return _collision_model


# Inter-arm pairs that always overlap due to shared bilateral base structure
_INTER_ARM_SKIP = {
    frozenset(['base', 'base']),
    frozenset(['base', 'Component139']),
    frozenset(['base', 'Component148']),
    frozenset(['base', 'Component151']),
}


def check_collision_at_waypoint(t1_grip_deg, t2_grip_deg, t1_free_deg, t2_free_deg,
                                 tail_deg=0.0, grip_is_right=False):
    """Check collision for a single waypoint using the mesh model.

    Builds a proper asymmetric collision scene: left arm at its angles,
    right arm at its angles, then checks all pairwise contacts.
    """
    model = get_collision_model()
    if model is None:
        return True, []

    import trimesh.collision
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'files'))
    from willy_collision import link_transforms, tail_transform

    # Determine which motor angles go to which side
    if grip_is_right:
        right_t1, right_t2 = t1_grip_deg, t2_grip_deg
        left_t1, left_t2 = t1_free_deg, t2_free_deg
    else:
        left_t1, left_t2 = t1_grip_deg, t2_grip_deg
        right_t1, right_t2 = t1_free_deg, t2_free_deg

    # Build collision manager with each arm at its own angles
    manager = trimesh.collision.CollisionManager()

    right_tfs = link_transforms(right_t1, right_t2, side='right')
    left_tfs = link_transforms(left_t1, left_t2, side='left')

    for name, mesh in model.meshes.items():
        if name in right_tfs:
            manager.add_object(f'right_{name}', mesh, transform=right_tfs[name])
        if name in left_tfs:
            manager.add_object(f'left_{name}', mesh, transform=left_tfs[name])

    # Add tail
    if model.tail_mesh is not None:
        manager.add_object('tail', model.tail_mesh,
                           transform=tail_transform(tail_deg))

    # Check collisions
    is_collision, names = manager.in_collision_internal(return_names=True)

    # Extended skip pairs — same-arm adjacent links + inter-arm base overlap
    _SAME_ARM_SKIP = {
        frozenset(['Component139', 'Component151']),
        frozenset(['Component139', 'Component146']),
        frozenset(['Component139', 'Component148']),  # share M1/M2 pivot area
        frozenset(['Component151', 'Component146']),
        frozenset(['Component151', 'Component147']),
        frozenset(['Component151', 'Component148']),
        frozenset(['Component146', 'Component143']),
        frozenset(['Component146', 'Component147']),  # close at J_distal
        frozenset(['Component147', 'Component143']),
        frozenset(['Component139', 'base']),
        frozenset(['Component148', 'base']),
        frozenset(['Component151', 'base']),
    }

    contacts = []
    for a, b in names:
        side_a = a.split('_', 1)[0] if '_' in a else ''
        side_b = b.split('_', 1)[0] if '_' in b else ''
        base_a = a.split('_', 1)[1] if '_' in a else a
        base_b = b.split('_', 1)[1] if '_' in b else b

        # Skip ALL inter-arm pairs — the bilateral spacing physically
        # separates the arms; the 3D mesh overlap is an artifact of
        # both arms being placed in the same working plane.
        if side_a != side_b:
            continue

        # Skip same-arm adjacent pairs (share pin joints)
        if frozenset([base_a, base_b]) in _SAME_ARM_SKIP:
            continue

        # Skip tail-base (pinned)
        if 'tail' in a or 'tail' in b:
            if 'base' in a or 'base' in b:
                continue

        contacts.append((a, b))

    return len(contacts) == 0, contacts


def mesh_outlines_2d(t1_deg, t2_deg, shoulder_world, side='right', is_mirrored=False):
    """Get 2D convex-hull outlines of all arm meshes at the given pose.

    Transforms each STL mesh into the working plane, projects to 2D,
    computes the convex hull, and returns polygon vertices in the planner's
    world frame (metres).

    Parameters
    ----------
    t1_deg, t2_deg : motor angles
    shoulder_world : (x, y) shoulder position in planner frame (metres)
    side : 'right' or 'left' for the collision model
    is_mirrored : if True, negate x (for left arm in planner frame)

    Returns list of (xs, ys, name) tuples for each link's 2D outline.
    """
    model = get_collision_model()
    if model is None:
        return []

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'files'))
    from willy_collision import link_transforms, HALF_BASE_SPACING, WORLD_ORIGIN_Y, WORLD_ORIGIN_Z

    tfs = link_transforms(t1_deg, t2_deg, side=side)

    # M1 position in collision model's world Y-Z coords
    sign = 1.0 if side == 'right' else -1.0
    m1_world_y = WORLD_ORIGIN_Y + sign * HALF_BASE_SPACING
    m1_world_z = WORLD_ORIGIN_Z

    outlines = []
    for name, mesh in model.meshes.items():
        if name not in tfs:
            continue
        tf = tfs[name]

        # Transform vertices to collision world frame
        verts_h = np.ones((len(mesh.vertices), 4))
        verts_h[:, :3] = mesh.vertices
        world_verts = (tf @ verts_h.T).T[:, :3]

        # Project to working plane: u = worldY - M1_worldY, v = worldZ - M1_worldZ
        # STLs are in MM (Fusion default), transforms are in CM.
        # The transform translations (cm) are mixed with mesh coords (mm)
        # in the result, but empirically the output is in mm-scale.
        # Convert mm → m by dividing by 1000.
        u_mm = world_verts[:, 1] - m1_world_y * 10.0  # convert cm origin to mm
        v_mm = world_verts[:, 2] - m1_world_z * 10.0

        # Convert to planner frame (metres), apply mirror for left arm
        if is_mirrored:
            x_m = shoulder_world[0] - u_mm / 1000.0
        else:
            x_m = shoulder_world[0] + u_mm / 1000.0
        y_m = shoulder_world[1] + v_mm / 1000.0

        # 2D convex hull
        points_2d = np.column_stack([x_m, y_m])
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points_2d)
            hull_xs = points_2d[hull.vertices, 0].tolist() + [points_2d[hull.vertices[0], 0]]
            hull_ys = points_2d[hull.vertices, 1].tolist() + [points_2d[hull.vertices[0], 1]]
            outlines.append((hull_xs, hull_ys, name))
        except Exception:
            pass  # degenerate hull, skip

    return outlines


def willy_gripper_tip_local(t1_deg, t2_deg):
    """Compute the gripper contact point (top of green line) in arm-local
    frame (M1 at origin, metres).

    This is the midpoint of the wrist plate (Comp143) plus 6 cm vertically.
    The wrist plate stays horizontal (parallel to base) due to the
    parallelogram, so its horizontal offset is constant in world coords.
    """
    alpha1 = (t1_deg + K1) * _D2R
    alpha2 = (t2_deg + K2) * _D2R
    alpha12 = alpha1 + alpha2
    fx, fy = math.cos(alpha12), math.sin(alpha12)

    # Elbow position
    elbow_x = A_M * math.cos(alpha1)
    elbow_y = A_M * math.sin(alpha1)

    # J4 = end of Comp146 (forearm)
    j4_x = elbow_x + _FOREARM_DRAW * fx
    j4_y = elbow_y + _FOREARM_DRAW * fy

    # J7 = J4 + wrist plate (horizontal, stays parallel to base)
    j7_x = j4_x + _WRIST_PLATE
    j7_y = j4_y  # same y because plate is horizontal

    # Gripper tip = midpoint of wrist plate + 6 cm up
    ee_x = (j4_x + j7_x) / 2.0
    ee_y = (j4_y + j7_y) / 2.0 + _EE_LENGTH

    return np.array([ee_x, ee_y])


# Parallelogram geometry constants (from Fusion model, in metres)
_FOREARM_DRAW = (L_CM - 7.62) / 100.0     # Comp146 length (29.166 cm)
_WRIST_PLATE  = 7.62 / 100.0              # Comp143 length (7.62 cm)
_M2_OFFSET    = np.array([7.62, -3.493]) / 100.0   # M2 base pivot offset from M1
_COMP151_DISTAL = np.array([7.62, 0.0]) / 100.0    # Comp151 distal corner from elbow
_COMP151_PURPLE = np.array([7.62, -3.493]) / 100.0 # Comp151 purple corner from elbow
_EE_LENGTH    = 0.06                       # 6 cm end-effector above wrist plate


def willy_arm_segments(t1_deg, t2_deg):
    """Compute all drawing segments for one Willy double-parallelogram arm.

    Returns a list of dicts, each with 'xs', 'ys', 'color', 'width', 'style'.
    All coordinates in arm-local frame (M1 at origin, metres).
    Matches the drawing from willy_ik_visualizer.html.
    """
    alpha1 = (t1_deg + K1) * _D2R
    alpha2 = (t2_deg + K2) * _D2R
    alpha12 = alpha1 + alpha2

    sx, sy = math.cos(alpha1), math.sin(alpha1)
    fx, fy = math.cos(alpha12), math.sin(alpha12)

    # Key positions
    M1 = np.array([0.0, 0.0])
    M2 = _M2_OFFSET.copy()
    elbow = np.array([A_M * sx, A_M * sy])

    # Comp151 triangle (stays parallel to base — only translates with elbow)
    J_orange = elbow.copy()
    J_distal = elbow + _COMP151_DISTAL
    J_purple = elbow + _COMP151_PURPLE

    # Comp146 forearm: elbow → J4
    J4 = elbow + np.array([_FOREARM_DRAW * fx, _FOREARM_DRAW * fy])

    # Comp143 wrist plate (parallel to base): J4 → J7
    J7 = J4 + np.array([_WRIST_PLATE, 0.0])

    # End-effector (vertical from wrist plate midpoint)
    ee_base = (J4 + J7) / 2.0
    ee_tip = ee_base + np.array([0.0, _EE_LENGTH])

    segs = []

    # 1. Base offset M1 → M2 (faint dashed)
    segs.append({'xs': [M1[0], M2[0]], 'ys': [M1[1], M2[1]],
                 'color': '#8a96a8', 'width': 0.8, 'style': '--'})

    # 2. Comp148 passive rod (M2 → J_purple)
    segs.append({'xs': [M2[0], J_purple[0]], 'ys': [M2[1], J_purple[1]],
                 'color': '#b388ff', 'width': 1.8, 'style': '-'})

    # 3. Comp139 upper arm (M1 → elbow)
    segs.append({'xs': [M1[0], elbow[0]], 'ys': [M1[1], elbow[1]],
                 'color': '#5fb3ff', 'width': 2.5, 'style': '-'})

    # 4. Comp151 triangle (3 edges)
    tri_xs = [J_orange[0], J_purple[0], J_distal[0], J_orange[0]]
    tri_ys = [J_orange[1], J_purple[1], J_distal[1], J_orange[1]]
    segs.append({'xs': tri_xs, 'ys': tri_ys,
                 'color': '#d4a8ff', 'width': 1.2, 'style': '-'})

    # 5. Comp146 forearm (elbow → J4)
    segs.append({'xs': [elbow[0], J4[0]], 'ys': [elbow[1], J4[1]],
                 'color': '#ff9f5f', 'width': 2.5, 'style': '-'})

    # 6. Comp147 passive rod (J_distal → J7)
    segs.append({'xs': [J_distal[0], J7[0]], 'ys': [J_distal[1], J7[1]],
                 'color': '#ffc89f', 'width': 1.8, 'style': '--'})

    # 7. Comp143 wrist plate (J4 → J7)
    segs.append({'xs': [J4[0], J7[0]], 'ys': [J4[1], J7[1]],
                 'color': '#ffc89f', 'width': 2.0, 'style': '-'})

    # 8. End-effector (vertical from wrist plate midpoint)
    segs.append({'xs': [ee_base[0], ee_tip[0]], 'ys': [ee_base[1], ee_tip[1]],
                 'color': '#6fdc8c', 'width': 1.5, 'style': '-'})

    return segs, {'M1': M1, 'M2': M2, 'elbow': elbow, 'J4': J4, 'J7': J7,
                  'ee_base': ee_base, 'ee_tip': ee_tip, 'hand': ee_tip}

# ── Helpers ──────────────────────────────────────────────────────────────────

def shoulder_from_bar(bar_pos, t1_grip_deg, t2_grip_deg, grip_is_left=False):
    """Given a bar position and gripping-arm Willy motor angles, return the
    shoulder (M1) position in the world frame.

    The gripper tip (top of green line) contacts the bar directly.
    shoulder = bar_pos - gripper_tip_local (with x mirrored for left arm).
    """
    tip_local = willy_gripper_tip_local(t1_grip_deg, t2_grip_deg)
    if grip_is_left:
        # Left arm: local +x maps to world -x
        shoulder_world = bar_pos - np.array([-tip_local[0], tip_local[1]])
    else:
        shoulder_world = bar_pos - tip_local
    return shoulder_world


def body_and_free_shoulder(grip_shoulder, grip_is_right):
    """From the gripping shoulder's world position, compute the body center
    and the free shoulder's world position.

    grip_is_right=True  → grip shoulder at +pivot from body center
    grip_is_right=False → grip shoulder at -pivot from body center
    """
    if grip_is_right:
        body_center = grip_shoulder - np.array([BODY_PIVOT_OFFSET, 0.0])
        free_shoulder = body_center - np.array([BODY_PIVOT_OFFSET, 0.0])
    else:
        body_center = grip_shoulder + np.array([BODY_PIVOT_OFFSET, 0.0])
        free_shoulder = body_center + np.array([BODY_PIVOT_OFFSET, 0.0])
    return body_center, free_shoulder


def free_hand_world(free_shoulder, t1_free_deg, t2_free_deg, free_is_left):
    """Free gripper-tip position in world frame.
    Uses the actual gripper tip (top of green line from parallelogram geometry).
    For the right arm, +u local = +x world.
    For the left arm, +u local = -x world (mirror)."""
    tip_local = willy_gripper_tip_local(t1_free_deg, t2_free_deg)
    if free_is_left:
        return free_shoulder + np.array([-tip_local[0], tip_local[1]])
    else:
        return free_shoulder + tip_local


def compute_com_world(t1_grip_deg, t2_grip_deg, t1_free_deg, t2_free_deg,
                      theta_tail_gui, grip_is_right, bar_pos):
    """Return (com_world_x, com_world_y) using the old mass model.

    Converts Willy motor angles to COM_Prediction's convention via K1/K2,
    then maps body-frame COM back to world coordinates.
    """
    grip_shoulder = shoulder_from_bar(bar_pos, t1_grip_deg, t2_grip_deg,
                                     grip_is_left=not grip_is_right)
    body_center, _ = body_and_free_shoulder(grip_shoulder, grip_is_right)

    # Map Willy motor angles → COM_Prediction convention:
    #   COM's theta_1 = absolute shoulder angle = motor_theta1 + K1
    #   COM's theta_2 = relative elbow angle   = motor_theta2 + K2
    if grip_is_right:
        r_theta_1 = t1_grip_deg + K1
        r_theta_2 = t2_grip_deg + K2
        l_theta_1 = t1_free_deg + K1
        l_theta_2 = t2_free_deg + K2
    else:
        r_theta_1 = t1_free_deg + K1
        r_theta_2 = t2_free_deg + K2
        l_theta_1 = t1_grip_deg + K1
        l_theta_2 = t2_grip_deg + K2

    theta_tail_internal = theta_tail_gui + TAIL_GUI_OFFSET_DEG

    x_com_in, y_com_in = COM_Prediction(
        r_theta_1, r_theta_2,
        l_theta_1, l_theta_2,
        theta_tail_internal,
    )

    # COM_Prediction returns body-frame inches.  Convert to metres.
    # Our world frame has body_center at body_center, with x mirrored.
    com_world_x = body_center[0] - x_com_in * INCH_TO_M
    com_world_y = body_center[1] + y_com_in * INCH_TO_M
    return com_world_x, com_world_y


# ── Angle ↔ flat-vector helpers ──────────────────────────────────────────────
# State per waypoint: [t1_grip, t2_grip, t1_free, t2_free, tail]
# All in RADIANS.  t1/t2 are Willy MOTOR angles (before K1/K2 offset).
STATE_DIM = 5

def pack_state(t1_grip_deg, t2_grip_deg, t1_free_deg, t2_free_deg, tail_gui_deg):
    return np.array([math.radians(t1_grip_deg), math.radians(t2_grip_deg),
                     math.radians(t1_free_deg), math.radians(t2_free_deg),
                     math.radians(tail_gui_deg)])

def unpack_state(x):
    """Returns (t1_grip_deg, t2_grip_deg, t1_free_deg, t2_free_deg, tail_gui_deg)."""
    return (math.degrees(x[0]), math.degrees(x[1]),
            math.degrees(x[2]), math.degrees(x[3]),
            math.degrees(x[4]))


def motor_angle_str(t1_deg, t2_deg):
    """Format Willy motor angles for display."""
    return f"t1={t1_deg:+6.1f}° t2={t2_deg:+6.1f}°"


# ── Optimisation ─────────────────────────────────────────────────────────────

def plan_trajectory(
    bar1_x, bar2_x, bar_y,
    grip_is_right=True,
    n_waypoints=10,
    L1=LINK1_LENGTH, L2=LINK2_LENGTH,
    right_shoulder_lim=(-70.9, 46.0),
    right_elbow_lim=(-90.4, 140.0),
    left_shoulder_lim=(-46.0, 70.9),
    left_elbow_lim=(-140.0, 90.4),
    tail_lim=(0.0, 0.0),         # tail fixed straight down
    com_tolerance=0.01,       # metres — how far COM_x may drift from bar
    smoothness_weight=0.5,
    reach_weight=10.0,
):
    """Plan a brachiation trajectory from bar 1 to bar 2.

    State per waypoint: [t1_grip, t2_grip, t1_free, t2_free, tail] in RADIANS.
    Motor angles use Willy convention (geometry angle = motor + K offset).

    Returns an (n_waypoints, STATE_DIM) array of joint-angle waypoints plus
    auxiliary info (COM positions, hand positions, etc.) for visualisation.
    """
    bar1 = np.array([bar1_x, bar_y])
    bar2 = np.array([bar2_x, bar_y])

    free_is_right = not grip_is_right

    # Determine per-arm motor limits (in degrees) based on which arm is gripping
    if grip_is_right:
        grip_t1_lim = right_shoulder_lim
        grip_t2_lim = right_elbow_lim
        free_t1_lim = left_shoulder_lim
        free_t2_lim = left_elbow_lim
    else:
        grip_t1_lim = left_shoulder_lim
        grip_t2_lim = left_elbow_lim
        free_t1_lim = right_shoulder_lim
        free_t2_lim = right_elbow_lim

    # Convert degree limits to radian bounds for the optimiser
    def deg_to_rad_bounds(lim):
        return (math.radians(lim[0]), math.radians(lim[1]))

    per_wp_bounds = [
        deg_to_rad_bounds(grip_t1_lim),
        deg_to_rad_bounds(grip_t2_lim),
        deg_to_rad_bounds(free_t1_lim),
        deg_to_rad_bounds(free_t2_lim),
        deg_to_rad_bounds(tail_lim),
    ]

    # Initial guess: arms reaching UP so the body hangs below the bar.
    # alpha1 ≈ 90° (shoulder pointing up) → t1 = 90 - K1 ≈ 62°
    # alpha2 ≈ 0° (forearm continues up) → t2 = 0 - K2 ≈ 32°
    t1_up = 90.0 - K1   # ≈ 62°
    t2_up = 0.0 - K2    # ≈ 32°
    x0_single = pack_state(t1_up, t2_up, t1_up, t2_up, 0.0)
    x0 = np.tile(x0_single, n_waypoints)

    # Build flat bounds
    bounds = per_wp_bounds * n_waypoints

    def cost_and_info(x_flat, return_info=False):
        X = x_flat.reshape(n_waypoints, STATE_DIM)
        total_cost = 0.0
        infos = []

        for k in range(n_waypoints):
            t1g, t2g, t1f, t2f, tail_deg = unpack_state(X[k])

            # Gripping arm shoulder
            grip_sh = shoulder_from_bar(bar1, t1g, t2g,
                                         grip_is_left=not grip_is_right)
            _, free_sh = body_and_free_shoulder(grip_sh, grip_is_right)

            # Free gripper tip position (top of green line = bar contact point).
            # free_hand_world now returns the actual gripper tip directly.
            fh = free_hand_world(free_sh, t1f, t2f, free_is_left=grip_is_right)
            gripper_tip = fh

            # Progress fraction — smooth hook trajectory.
            # The gripper swings across to bar 2 then approaches from
            # ~3 inches below.  Uses a single smooth cosine curve for y
            # to avoid sharp transitions between phases.
            t = k / max(n_waypoints - 1, 1)

            # x: smooth ease-in-out toward bar 2
            target_x = bar1[0] + (bar2[0] - bar1[0]) * (0.5 - 0.5 * math.cos(t * math.pi))

            # y: smooth cosine dip — peaks (bar height) at t=0 and t=1,
            # lowest point at t≈0.65 (near bar 2 x-position), dipping
            # 3 inches below bar.
            dip_depth = 8.0 * INCH_TO_M  # 8 inches below bar
            # Shift the dip center toward the end (t=0.65) so the
            # approach to bar 2 comes from below.
            dip_center = 0.50
            # Use a Gaussian-like bump centered at dip_center
            sigma = 0.35  # wider arc, smoother curve
            dip = dip_depth * math.exp(-0.5 * ((t - dip_center) / sigma) ** 2)
            target_y = bar1[1] - dip

            target = np.array([target_x, target_y])

            # Cost: distance of free gripper tip to its target at this waypoint
            tip_err = np.linalg.norm(gripper_tip - target)
            total_cost += reach_weight * tip_err ** 2

            # Final waypoint: extra penalty on bar2 miss (gripper tip must reach bar)
            if k == n_waypoints - 1:
                total_cost += 50.0 * reach_weight * np.linalg.norm(gripper_tip - bar2) ** 2

            # Smoothness: penalise big jumps between consecutive waypoints
            if k > 0:
                diff = X[k] - X[k - 1]
                total_cost += smoothness_weight * np.dot(diff, diff)

            if return_info:
                com_x, com_y = compute_com_world(
                    t1g, t2g, t1f, t2f, tail_deg,
                    grip_is_right, bar1,
                )
                infos.append({
                    "grip_shoulder": grip_sh,
                    "free_shoulder": free_sh,
                    "free_hand": fh,
                    "free_gripper_tip": gripper_tip.copy(),
                    "com": np.array([com_x, com_y]),
                    "target": target,
                    "state": X[k].copy(),
                })

        if return_info:
            return total_cost, infos
        return total_cost

    def cost(x_flat):
        return cost_and_info(x_flat, return_info=False)

    # COM constraint: allow COM to drift behind the gripped bar (toward -x,
    # safe — pendulum swings back) but limit forward drift (toward bar 2).
    #   COM_x ≤ bar1_x + com_tolerance   (don't drift too far toward bar 2)
    #   COM_x ≥ bar1_x - 3 * com_tolerance  (generous backward allowance)
    constraints = []
    for k in range(n_waypoints):
        def make_com_lo(kk):
            def con(x_flat):
                X = x_flat.reshape(n_waypoints, STATE_DIM)
                t1g, t2g, t1f, t2f, td = unpack_state(X[kk])
                cx, _ = compute_com_world(t1g, t2g, t1f, t2f, td,
                                          grip_is_right, bar1)
                return cx - bar1_x + 3.0 * com_tolerance  # generous backward
            return con

        def make_com_hi(kk):
            def con(x_flat):
                X = x_flat.reshape(n_waypoints, STATE_DIM)
                t1g, t2g, t1f, t2f, td = unpack_state(X[kk])
                cx, _ = compute_com_world(t1g, t2g, t1f, t2f, td,
                                          grip_is_right, bar1)
                return bar1_x + com_tolerance - cx  # tight forward
            return con

        constraints.append({"type": "ineq", "fun": make_com_lo(k)})
        constraints.append({"type": "ineq", "fun": make_com_hi(k)})

    # Hard equality constraint: final waypoint gripper tip must be AT bar 2.
    def final_grip_x(x_flat):
        X = x_flat.reshape(n_waypoints, STATE_DIM)
        t1g, t2g, t1f, t2f, td = unpack_state(X[-1])
        grip_sh = shoulder_from_bar(bar1, t1g, t2g,
                                    grip_is_left=not grip_is_right)
        _, free_sh = body_and_free_shoulder(grip_sh, grip_is_right)
        tip = free_hand_world(free_sh, t1f, t2f, free_is_left=grip_is_right)
        return tip[0] - bar2[0]

    def final_grip_y(x_flat):
        X = x_flat.reshape(n_waypoints, STATE_DIM)
        t1g, t2g, t1f, t2f, td = unpack_state(X[-1])
        grip_sh = shoulder_from_bar(bar1, t1g, t2g,
                                    grip_is_left=not grip_is_right)
        _, free_sh = body_and_free_shoulder(grip_sh, grip_is_right)
        tip = free_hand_world(free_sh, t1f, t2f, free_is_left=grip_is_right)
        return tip[1] - bar2[1]

    constraints.append({"type": "eq", "fun": final_grip_x})
    constraints.append({"type": "eq", "fun": final_grip_y})

    print(f"Optimising {n_waypoints} waypoints × {STATE_DIM} DOF "
          f"= {n_waypoints * STATE_DIM} variables …")

    result = minimize(
        cost,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9, "disp": True},
    )

    print(f"Optimiser exit: {result.message}  (success={result.success})")

    _, infos = cost_and_info(result.x, return_info=True)
    X_opt = result.x.reshape(n_waypoints, STATE_DIM)

    return X_opt, infos, result


# ── Visualisation ────────────────────────────────────────────────────────────

def draw_robot_at_waypoint(ax, info, grip_is_right, L1, L2, alpha=1.0,
                           bar_pos=None, show_meshes=False):
    """Draw the full robot for one waypoint.

    Uses the Willy double-parallelogram arm style from the HTML visualizer.
    If show_meshes=True, also draws translucent convex hull outlines of
    the actual STL meshes.
    """
    segments = get_robot_segments(info, grip_is_right, L1, L2, bar_pos=bar_pos)

    # Segment structure:
    #   [0]        body line
    #   [1..8]     grip arm (8 parallelogram segments from willy_arm_segments)
    #   [9..16]    free arm (8 parallelogram segments)
    #   [17]       body box
    #   [18]       tail
    #   [19]       COM point
    #
    # We draw with appropriate colors from willy_arm_segments.
    # Get the color/style info from a dummy call for reference:
    ref_segs, _ = willy_arm_segments(0, 0)
    n_arm_segs = len(ref_segs)

    for i, (xs, ys) in enumerate(segments):
        if i == len(segments) - 1:
            # COM marker (last segment is a single point)
            ax.plot(xs[0], ys[0], "D", color="#0891b2", markersize=6, alpha=alpha)
        elif i == 0:
            # Body line
            ax.plot(xs, ys, color="#6b7280", linewidth=4, alpha=alpha,
                    solid_capstyle="round")
        elif 1 <= i <= n_arm_segs:
            # Grip arm segments
            seg_info = ref_segs[i - 1]
            ax.plot(xs, ys, color=seg_info['color'],
                    linewidth=seg_info['width'], linestyle=seg_info['style'],
                    alpha=alpha)
        elif n_arm_segs + 1 <= i <= 2 * n_arm_segs:
            # Free arm segments
            seg_info = ref_segs[i - n_arm_segs - 1]
            ax.plot(xs, ys, color=seg_info['color'],
                    linewidth=seg_info['width'], linestyle=seg_info['style'],
                    alpha=alpha * 0.7)
        elif i == 2 * n_arm_segs + 1:
            # Body box
            ax.plot(xs, ys, color="#4b5563", linewidth=2, alpha=alpha)
        elif i == 2 * n_arm_segs + 2:
            # Tail
            ax.plot(xs, ys, color="#d97706", linewidth=3, alpha=alpha)

    # Draw mesh outlines if requested
    if show_meshes and alpha > 0.5:  # only for visible waypoints
        state = info["state"]
        t1g, t2g, t1f, t2f, _ = unpack_state(state)
        gs = info["grip_shoulder"]
        fs = info["free_shoulder"]
        grip_is_left = not grip_is_right

        # Grip arm meshes
        grip_outlines = mesh_outlines_2d(
            t1g, t2g, gs,
            side='left' if grip_is_left else 'right',
            is_mirrored=grip_is_left)
        for hx, hy, name in grip_outlines:
            ax.fill(hx, hy, color="#5fb3ff", alpha=0.08 * alpha, linewidth=0)
            ax.plot(hx, hy, color="#5fb3ff", linewidth=0.5, alpha=0.3 * alpha)

        # Free arm meshes
        free_is_left = not grip_is_left
        free_outlines = mesh_outlines_2d(
            t1f, t2f, fs,
            side='left' if free_is_left else 'right',
            is_mirrored=free_is_left)
        for hx, hy, name in free_outlines:
            ax.fill(hx, hy, color="#ff9f5f", alpha=0.08 * alpha, linewidth=0)
            ax.plot(hx, hy, color="#ff9f5f", linewidth=0.5, alpha=0.3 * alpha)


def visualise_trajectory(X_opt, infos, bar1_x, bar2_x, bar_y,
                         grip_is_right, L1, L2):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.canvas.manager.set_window_title("Brachiation Trajectory Planner")

    # ── Left panel: robot poses ──────────────────────────────────────────
    ax = axes[0]
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Planned Trajectory (ghost poses)")

    # Zoom to fit the trajectory tightly
    all_gt_x = [info["free_gripper_tip"][0] for info in infos]
    all_gt_y = [info["free_gripper_tip"][1] for info in infos]
    all_gs_x = [info["grip_shoulder"][0] for info in infos]
    all_gs_y = [info["grip_shoulder"][1] for info in infos]
    all_fs_x = [info["free_shoulder"][0] for info in infos]
    all_fs_y = [info["free_shoulder"][1] for info in infos]
    pad = 0.12
    x_lo = min(min(all_gt_x), min(all_gs_x), min(all_fs_x), bar1_x) - pad
    x_hi = max(max(all_gt_x), max(all_gs_x), max(all_fs_x), bar2_x) + pad
    y_lo = min(min(all_gt_y), min(all_gs_y), min(all_fs_y)) - pad
    y_hi = max(bar_y, max(all_gt_y)) + pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Bars
    bar_half = 0.04
    for bx, label in [(bar1_x, "Bar 1"), (bar2_x, "Bar 2")]:
        ax.plot([bx, bx], [bar_y - bar_half, bar_y + bar_half],
                color="brown", linewidth=6, solid_capstyle="round")
        ax.text(bx, bar_y + bar_half + 0.01, label, ha="center", fontsize=9)

    n = len(infos)
    show_meshes = False  # mesh outlines disabled for now
    for k, info in enumerate(infos):
        alpha = 0.15 + 0.85 * (k / max(n - 1, 1))
        draw_robot_at_waypoint(ax, info, grip_is_right, L1, L2, alpha=alpha,
                               bar_pos=None, show_meshes=show_meshes)
        # Mark colliding waypoints with a red X
        if not info.get("collision_safe", True):
            gt = info["free_gripper_tip"]
            ax.plot(gt[0], gt[1], "x", color="red", markersize=12,
                    markeredgewidth=2.5, alpha=alpha)

    # Draw reference hook trajectory (smooth Gaussian dip near bar 2)
    arc_t = np.linspace(0, 1, 80)
    dip_depth = 8.0 * INCH_TO_M
    dip_center = 0.50
    sigma = 0.35
    arc_xs = [bar1_x + (bar2_x - bar1_x) * (0.5 - 0.5 * math.cos(t * math.pi))
              for t in arc_t]
    arc_ys = [bar_y - dip_depth * math.exp(-0.5 * ((t - dip_center) / sigma) ** 2)
              for t in arc_t]
    ax.plot(arc_xs, arc_ys, "--", color="#a855f7", linewidth=1.5,
            alpha=0.6, label="Reference path")

    # Draw COM and gripper tip paths (tilted)
    # COM and gripper tip paths (no tilt — body stays horizontal)
    com_xs = [info["com"][0] for info in infos]
    com_ys = [info["com"][1] for info in infos]
    gt_xs = [info["free_gripper_tip"][0] for info in infos]
    gt_ys = [info["free_gripper_tip"][1] for info in infos]

    ax.plot(com_xs, com_ys, "o-", color="#0891b2", markersize=4,
            linewidth=1.5, label="COM path")
    ax.plot(gt_xs, gt_ys, "x-", color="#f59e0b", markersize=6,
            linewidth=1.5, label="Free gripper tip path")

    # COM tolerance band
    ax.axvline(bar1_x, color="red", linestyle=":", alpha=0.5, label="Bar 1 x")

    ax.legend(loc="lower left", fontsize=8)

    # ── Right panel: angle / COM plots over time ─────────────────────────
    ax2 = axes[1]
    steps = np.arange(n)

    ax2.plot(steps, [math.degrees(info["state"][0]) for info in infos],
             label="grip shoulder θ", marker=".")
    ax2.plot(steps, [math.degrees(info["state"][1]) for info in infos],
             label="grip elbow φ", marker=".")
    ax2.plot(steps, [math.degrees(info["state"][2]) for info in infos],
             label="free shoulder θ", marker=".")
    ax2.plot(steps, [math.degrees(info["state"][3]) for info in infos],
             label="free elbow φ", marker=".")
    ax2.plot(steps, [math.degrees(info["state"][4]) for info in infos],
             label="tail (gui deg)", marker=".")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(steps, [(info["com"][0] - bar1_x) / INCH_TO_M for info in infos],
                  "D-", color="#0891b2", markersize=4, label="COM_x offset (in)")
    ax2_twin.axhline(0, color="red", linestyle=":", alpha=0.5)
    ax2_twin.set_ylabel("COM_x offset from bar (in)", color="#0891b2")

    ax2.set_xlabel("Waypoint")
    ax2.set_ylabel("Angle (deg)")
    ax2.set_title("Joint Angles & COM offset")
    ax2.legend(loc="upper left", fontsize=8)
    ax2_twin.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Pendulum tilt ────────────────────────────────────────────────────────────

def rotate_points(xs, ys, cx, cy, angle):
    """Rotate arrays of x,y coords around (cx, cy) by angle (radians)."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    dx = xs - cx
    dy = ys - cy
    rx = cx + dx * cos_a - dy * sin_a
    ry = cy + dx * sin_a + dy * cos_a
    return rx.tolist(), ry.tolist()


def pendulum_tilt_angle(com_world, grip_point):
    """Compute the tilt angle (radians) to rotate the robot so COM hangs
    directly below the grip point.

    Positive angle = COM was to the right of the bar, robot tilts clockwise.
    """
    dx = com_world[0] - grip_point[0]
    dy = com_world[1] - grip_point[1]  # negative (COM below bar)
    return math.atan2(dx, -dy)


def apply_tilt_to_segments(segments, tilt_angle, grip_point):
    """Rotate every segment in the list around grip_point by -tilt_angle."""
    tilted = []
    gx, gy = grip_point
    for xs, ys in segments:
        rx, ry = rotate_points(xs, ys, gx, gy, -tilt_angle)
        tilted.append((rx, ry))
    return tilted


# ── IMU Interface & Replanning ───────────────────────────────────────────────

class IMUReading:
    """Single IMU measurement.  All angles in radians."""
    __slots__ = ("tilt_rad", "tilt_rate_rad_s", "timestamp_s")

    def __init__(self, tilt_rad, tilt_rate_rad_s=0.0, timestamp_s=0.0):
        self.tilt_rad = tilt_rad              # body tilt from vertical (+ = CW)
        self.tilt_rate_rad_s = tilt_rate_rad_s  # angular velocity
        self.timestamp_s = timestamp_s

    def tilt_deg(self):
        return math.degrees(self.tilt_rad)

    def __repr__(self):
        return (f"IMUReading(tilt={self.tilt_deg():+.1f}°, "
                f"rate={math.degrees(self.tilt_rate_rad_s):+.1f}°/s, "
                f"t={self.timestamp_s:.3f}s)")


class IMUInterface:
    """Abstract IMU interface.  Subclass and implement read() for real hardware.

    To plug in a real IMU:
      1. Subclass IMUInterface
      2. Implement read() → IMUReading
      3. Implement connect() / disconnect() for hardware lifecycle
      4. Pass the instance to ReplanningController
    """

    def connect(self):
        """Open connection to IMU hardware."""
        pass

    def disconnect(self):
        """Close connection."""
        pass

    def read(self):
        """Return the latest IMUReading.  Must not block for long."""
        raise NotImplementedError

    def is_connected(self):
        return False


class SimulatedIMU(IMUInterface):
    """Simulated IMU that computes tilt from the pendulum model.

    Uses the pre-computed trajectory info to return what the IMU *would*
    read if the robot perfectly followed the planned trajectory.
    Optionally adds Gaussian noise to simulate sensor noise.
    """

    def __init__(self, infos, grip_point, noise_std_deg=0.0):
        self.infos = infos
        self.grip_point = np.array(grip_point)
        self.noise_std_rad = math.radians(noise_std_deg)
        self._step = 0
        self._connected = False

    def connect(self):
        self._connected = True
        self._step = 0

    def disconnect(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def read(self):
        if self._step >= len(self.infos):
            self._step = len(self.infos) - 1
        info = self.infos[self._step]
        com = info["com"]
        tilt = pendulum_tilt_angle(com, self.grip_point)
        # Add noise
        if self.noise_std_rad > 0:
            tilt += np.random.normal(0, self.noise_std_rad)
        # Estimate rate from previous step
        rate = 0.0
        if self._step > 0:
            prev_com = self.infos[self._step - 1]["com"]
            prev_tilt = pendulum_tilt_angle(prev_com, self.grip_point)
            rate = tilt - prev_tilt  # per-step, not per-second
        return IMUReading(tilt_rad=tilt, tilt_rate_rad_s=rate,
                          timestamp_s=self._step * 0.1)

    def advance(self):
        """Move to the next waypoint (call after each control step)."""
        self._step += 1


class ReplanningController:
    """Closed-loop trajectory controller that replans when IMU tilt
    diverges from the planned trajectory.

    Usage:
        ctrl = ReplanningController(planner_kwargs, imu, ...)
        ctrl.plan_initial()
        while not ctrl.done:
            reading = imu.read()
            ctrl.step(reading)
            # ctrl.current_target_angles() → send to motors
            imu.advance()  # if simulated
    """

    def __init__(self, bar1_x, bar2_x, bar_y, grip_is_right,
                 imu, n_waypoints=16,
                 replan_threshold_deg=5.0,
                 planner_kwargs=None):
        self.bar1_x = bar1_x
        self.bar2_x = bar2_x
        self.bar_y = bar_y
        self.grip_is_right = grip_is_right
        self.imu = imu
        self.n_waypoints = n_waypoints
        self.replan_threshold_rad = math.radians(replan_threshold_deg)
        self.planner_kwargs = planner_kwargs or {}

        self.trajectory = None   # (n_waypoints, STATE_DIM) array
        self.infos = None        # list of info dicts
        self.current_wp = 0
        self.planned_tilts = []  # expected tilt at each waypoint
        self.measured_tilts = [] # actual tilt readings
        self.replan_count = 0
        self.done = False

    def plan_initial(self):
        """Run the initial full trajectory plan."""
        grip_point = np.array([self.bar1_x, self.bar_y])
        X_opt, infos, result = plan_trajectory(
            self.bar1_x, self.bar2_x, self.bar_y,
            grip_is_right=self.grip_is_right,
            n_waypoints=self.n_waypoints,
            **self.planner_kwargs,
        )
        self.trajectory = X_opt
        self.infos = infos
        self.current_wp = 0
        self.done = False

        # Pre-compute expected tilts
        self.planned_tilts = []
        for info in infos:
            tilt = pendulum_tilt_angle(info["com"], grip_point)
            self.planned_tilts.append(tilt)

        return X_opt, infos, result

    def step(self, imu_reading):
        """Process one IMU reading.  Returns dict with status info.

        Call this once per control cycle.  It will:
          1. Compare measured vs planned tilt
          2. Replan if error exceeds threshold
          3. Advance to next waypoint
        """
        if self.done or self.current_wp >= len(self.planned_tilts):
            self.done = True
            return {"status": "done", "wp": self.current_wp}

        measured_tilt = imu_reading.tilt_rad
        planned_tilt = self.planned_tilts[self.current_wp]
        error = measured_tilt - planned_tilt
        self.measured_tilts.append(measured_tilt)

        status = {
            "wp": self.current_wp,
            "planned_tilt_deg": math.degrees(planned_tilt),
            "measured_tilt_deg": math.degrees(measured_tilt),
            "error_deg": math.degrees(error),
            "replanned": False,
            "status": "tracking",
        }

        # Check if replan is needed
        if abs(error) > self.replan_threshold_rad:
            remaining = self.n_waypoints - self.current_wp
            if remaining > 2:
                print(f"\n[REPLAN] WP {self.current_wp}: tilt error "
                      f"{math.degrees(error):+.1f}° exceeds threshold "
                      f"({math.degrees(self.replan_threshold_rad):.1f}°). "
                      f"Replanning {remaining} remaining waypoints...")

                # Replan from current state with adjusted initial guess
                # that accounts for the actual tilt
                self._replan_from_current(measured_tilt)
                status["replanned"] = True
                self.replan_count += 1

        self.current_wp += 1
        if self.current_wp >= self.n_waypoints:
            self.done = True
            status["status"] = "done"

        return status

    def _replan_from_current(self, measured_tilt):
        """Replan remaining waypoints starting from current state."""
        remaining = self.n_waypoints - self.current_wp
        grip_point = np.array([self.bar1_x, self.bar_y])

        # Use the current waypoint's state as starting point
        current_state = self.trajectory[self.current_wp]

        X_opt, infos, result = plan_trajectory(
            self.bar1_x, self.bar2_x, self.bar_y,
            grip_is_right=self.grip_is_right,
            n_waypoints=remaining,
            **self.planner_kwargs,
        )

        if result.success:
            # Splice the new plan into the trajectory
            self.trajectory[self.current_wp:] = X_opt
            self.infos[self.current_wp:] = infos

            # Update planned tilts
            for i, info in enumerate(infos):
                wp_idx = self.current_wp + i
                if wp_idx < len(self.planned_tilts):
                    self.planned_tilts[wp_idx] = pendulum_tilt_angle(
                        info["com"], grip_point)
            print(f"[REPLAN] Success — {remaining} waypoints replanned.")
        else:
            print(f"[REPLAN] Failed: {result.message}")

    def current_target_angles(self):
        """Return the current waypoint's joint angles as a dict.

        These are the angles to send to the motors.
        """
        if self.current_wp >= len(self.trajectory):
            wp = self.trajectory[-1]
        else:
            wp = self.trajectory[self.current_wp]

        t1g, t2g, t1f, t2f, td = unpack_state(wp)
        return {
            "grip_shoulder_motor_deg": t1g,
            "grip_elbow_motor_deg": t2g,
            "free_shoulder_motor_deg": t1f,
            "free_elbow_motor_deg": t2f,
            "tail_gui_deg": td,
        }

    def summary(self):
        """Print execution summary."""
        print(f"\n── Replanning Summary ──")
        print(f"  Total waypoints: {self.n_waypoints}")
        print(f"  Replans triggered: {self.replan_count}")
        if self.measured_tilts:
            errors = [abs(m - p) for m, p in
                      zip(self.measured_tilts, self.planned_tilts[:len(self.measured_tilts)])]
            print(f"  Max tilt error: {math.degrees(max(errors)):.1f}°")
            print(f"  Mean tilt error: {math.degrees(sum(errors)/len(errors)):.1f}°")


def run_closed_loop_demo(X_opt, infos, bar1_x, bar2_x, bar_y,
                         grip_is_right, noise_deg=2.0, replan_threshold_deg=5.0,
                         planner_kwargs=None):
    """Run a simulated closed-loop execution of the planned trajectory.

    Uses SimulatedIMU with noise to test the replanning pipeline.
    """
    grip_point = (bar1_x, bar_y)

    # Create simulated IMU from the planned trajectory
    imu = SimulatedIMU(infos, grip_point, noise_std_deg=noise_deg)
    imu.connect()

    # Create controller
    ctrl = ReplanningController(
        bar1_x, bar2_x, bar_y, grip_is_right, imu,
        n_waypoints=len(infos),
        replan_threshold_deg=replan_threshold_deg,
        planner_kwargs=planner_kwargs or {},
    )
    # Use the already-planned trajectory instead of replanning from scratch
    ctrl.trajectory = X_opt.copy()
    ctrl.infos = list(infos)
    ctrl.planned_tilts = [
        pendulum_tilt_angle(info["com"], np.array(grip_point))
        for info in infos
    ]

    print(f"\n── Closed-Loop Demo (noise={noise_deg}°, "
          f"replan threshold={replan_threshold_deg}°) ──")

    step_log = []
    while not ctrl.done:
        reading = imu.read()
        status = ctrl.step(reading)
        angles = ctrl.current_target_angles()
        step_log.append({**status, **angles})

        print(f"  WP {status['wp']:2d}: "
              f"planned={status['planned_tilt_deg']:+6.1f}°  "
              f"measured={status['measured_tilt_deg']:+6.1f}°  "
              f"error={status['error_deg']:+5.1f}°  "
              f"{'REPLAN' if status['replanned'] else ''}")
        imu.advance()

    ctrl.summary()
    imu.disconnect()
    return ctrl, step_log


# ── 3D Animated Visualisation ────────────────────────────────────────────────

def get_robot_segments(info, grip_is_right, L1, L2, bar_pos=None):
    """Return a list of (xs, ys) line segments for one waypoint in world frame.

    If bar_pos is given, the entire robot is rotated around the grip point
    (bar contact) so that the COM hangs directly below it (pendulum tilt).
    Also returns body_box vertices and tail segment."""
    gs = info["grip_shoulder"]
    fs = info["free_shoulder"]
    com = info["com"]
    state = info["state"]
    t1g, t2g, t1f, t2f, tail_deg = unpack_state(state)

    if grip_is_right:
        r_sh, l_sh = gs, fs
    else:
        r_sh, l_sh = fs, gs

    segments = []

    # Body (line between shoulders)
    segments.append(([r_sh[0], l_sh[0]], [r_sh[1], l_sh[1]]))

    # Gripping arm — full Willy double-parallelogram drawing.
    # Mirror x if the gripping arm is the LEFT arm (same as HTML's scale(-1,1)).
    grip_segs, _ = willy_arm_segments(t1g, t2g)
    grip_is_left = not grip_is_right
    for seg in grip_segs:
        if grip_is_left:
            xs = [gs[0] - x for x in seg['xs']]
        else:
            xs = [gs[0] + x for x in seg['xs']]
        ys = [gs[1] + y for y in seg['ys']]
        segments.append((xs, ys))

    # Free arm — mirror x if the free arm is the LEFT arm.
    free_segs, _ = willy_arm_segments(t1f, t2f)
    free_is_right = not grip_is_right
    free_is_left = not free_is_right
    for seg in free_segs:
        if free_is_left:
            xs = [fs[0] - x for x in seg['xs']]
        else:
            xs = [fs[0] + x for x in seg['xs']]
        ys = [fs[1] + y for y in seg['ys']]
        segments.append((xs, ys))

    # Body box: rectangle centered between shoulders, dropping down
    body_cx = (r_sh[0] + l_sh[0]) / 2.0
    body_cy = (r_sh[1] + l_sh[1]) / 2.0
    body_half_w = BODY_PIVOT_OFFSET  # half-width = shoulder offset
    body_height = 4.0 * INCH_TO_M   # ~4 inches tall box
    box_xs = [body_cx - body_half_w, body_cx + body_half_w,
              body_cx + body_half_w, body_cx - body_half_w,
              body_cx - body_half_w]
    box_ys = [body_cy, body_cy,
              body_cy - body_height, body_cy - body_height,
              body_cy]
    segments.append((box_xs, box_ys))

    # Tail: hangs from bottom-center of body box
    tail_pivot_x = body_cx
    tail_pivot_y = body_cy - body_height
    tail_angle = math.radians(tail_deg + TAIL_GUI_OFFSET_DEG + TAIL_ANGLE_STOP_DEG)
    tail_end_x = tail_pivot_x + TAIL_LENGTH * math.cos(tail_angle)
    tail_end_y = tail_pivot_y - TAIL_LENGTH * math.sin(tail_angle)
    segments.append(([tail_pivot_x, tail_end_x], [tail_pivot_y, tail_end_y]))

    # COM as a tiny segment (so it can be rotated with everything else)
    segments.append(([com[0]], [com[1]]))

    # ── Pendulum tilt ────────────────────────────────────────────────
    # Rotate everything around the grip point (bar contact) so COM hangs
    # directly below.  The gripping arm stays connected to the bar.
    if bar_pos is not None:
        tilt = pendulum_tilt_angle(com, bar_pos)
        grip_point = np.array(bar_pos)
        segments = apply_tilt_to_segments(segments, tilt, grip_point)

    return segments


def animate_trajectory_3d(X_opt, infos, bar1_x, bar2_x, bar_y,
                          grip_is_right, L1, L2, interval_ms=250):
    """Animate the trajectory in 3D (bars along z-axis, robot in x-y plane)."""
    fig = plt.figure(figsize=(14, 9))
    fig.canvas.manager.set_window_title("Brachiation 3D Animation")
    ax = fig.add_subplot(111, projection="3d")

    # Bar geometry in 3D: bars run along z, at fixed (x, y)
    bar_len_z = 0.25   # 25 cm bar length in z
    z_lo, z_hi = -bar_len_z / 2, bar_len_z / 2

    # Draw bars
    for bx, color, label in [(bar1_x, "#8B4513", "Bar 1"),
                              (bar2_x, "#A0522D", "Bar 2")]:
        ax.plot([bx, bx], [bar_y, bar_y], [z_lo, z_hi],
                color=color, linewidth=6, solid_capstyle="round")
        ax.text(bx, bar_y + 0.02, 0, label, ha="center", fontsize=9)

    # COM tolerance band (translucent vertical plane at bar1_x)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tol = 0.015
    y_lo_band = bar_y - 0.6
    y_hi_band = bar_y + 0.05
    verts = [[(bar1_x - tol, y_lo_band, z_lo),
              (bar1_x + tol, y_lo_band, z_lo),
              (bar1_x + tol, y_hi_band, z_lo),
              (bar1_x - tol, y_hi_band, z_lo)]]
    band = Poly3DCollection(verts, alpha=0.08, facecolor="red", edgecolor="red",
                            linewidth=0.5)
    ax.add_collection3d(band)

    # Prepare line objects for animation (all in the z=0 plane).
    # Segment count: 1 body + 8 grip arm + 8 free arm + 1 body box + 1 tail = 19
    # (COM point is handled separately as a marker)
    ref_segs, _ = willy_arm_segments(0, 0)
    n_arm_segs = len(ref_segs)
    n_segs = 1 + n_arm_segs + n_arm_segs + 1 + 1  # 19

    # Build color/width/style arrays
    colors = ["#6b7280"]  # body
    widths = [4]
    styles = ["-"]
    for seg in ref_segs:  # grip arm
        colors.append(seg['color'])
        widths.append(seg['width'])
        styles.append(seg['style'])
    for seg in ref_segs:  # free arm
        colors.append(seg['color'])
        widths.append(seg['width'])
        styles.append(seg['style'])
    colors += ["#4b5563", "#d97706"]  # body box, tail
    widths += [2, 3]
    styles += ["-", "-"]

    lines_3d = []
    for i in range(n_segs):
        line, = ax.plot([], [], [], color=colors[i], linewidth=widths[i],
                        linestyle=styles[i])
        lines_3d.append(line)

    com_dot, = ax.plot([], [], [], "D", color="#0891b2", markersize=8)
    com_trail, = ax.plot([], [], [], "-", color="#0891b2", linewidth=1, alpha=0.5)
    fh_trail, = ax.plot([], [], [], "x-", color="#f59e0b", markersize=5,
                        linewidth=1, alpha=0.5)

    # Info text
    info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                          family="monospace", va="top",
                          bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Axis limits
    all_xs = [info["com"][0] for info in infos]
    all_xs += [info["grip_shoulder"][0] for info in infos]
    all_xs += [info["free_hand"][0] for info in infos]
    x_margin = 0.08
    ax.set_xlim(min(all_xs) - x_margin, max(all_xs) + x_margin)
    ax.set_ylim(bar_y - 0.55, bar_y + 0.08)
    ax.set_zlim(z_lo - 0.05, z_hi + 0.05)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m) — bar axis")
    ax.set_title("Brachiation Trajectory")
    ax.view_init(elev=20, azim=-60)

    com_history_x = []
    com_history_y = []
    fh_history_x = []
    fh_history_y = []

    def update(frame):
        info = infos[frame]
        segments = get_robot_segments(info, grip_is_right, L1, L2,
                                     bar_pos=None)  # body stays horizontal
        # Last segment = COM point (no tilt)
        com_x = segments[-1][0][0]
        com_y = segments[-1][1][0]
        fh = info["free_gripper_tip"]

        n_draw = len(segments) - 1  # skip COM point segment
        for i in range(min(n_draw, len(lines_3d))):
            xs, ys = segments[i]
            zs = [0.0] * len(xs)
            lines_3d[i].set_data_3d(xs, ys, zs)

        com_dot.set_data_3d([com_x], [com_y], [0.0])

        com_history_x.append(com_x)
        com_history_y.append(com_y)
        com_trail.set_data_3d(com_history_x, com_history_y,
                              [0.0] * len(com_history_x))

        fh_history_x.append(fh[0])
        fh_history_y.append(fh[1])
        fh_trail.set_data_3d(fh_history_x, fh_history_y,
                             [0.0] * len(fh_history_x))

        t1g, t2g, t1f, t2f, td = unpack_state(info["state"])
        info_text.set_text(
            f"WP {frame}/{len(infos)-1}\n"
            f"gripper tip: ({fh[0]:+.3f}, {fh[1]:+.3f}) m\n"
            f"COM_x offset: {(com_x-bar1_x)*1000:+.1f} mm\n"
            f"grip: t1={t1g:+.1f}° t2={t2g:+.1f}°\n"
            f"free: t1={t1f:+.1f}° t2={t2f:+.1f}°\n"
            f"tail: {td:+.1f}°"
        )

        return lines_3d + [com_dot, com_trail, fh_trail, info_text]

    anim = FuncAnimation(fig, update, frames=len(infos),
                         interval=interval_ms, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()
    return anim


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Brachiation trajectory planner: swing from bar 1 to bar 2"
    )
    parser.add_argument("--bar-spacing", type=float, default=12.0,
                        help="Bar spacing in inches (default 12)")
    parser.add_argument("--bar-y", type=float, default=None,
                        help="Bar y-height in metres (default: auto from arm reach)")
    parser.add_argument("--grip", choices=["left", "right"], default="left",
                        help="Which hand grips bar 1 (default: left)")
    parser.add_argument("--n-waypoints", type=int, default=12,
                        help="Number of trajectory waypoints (default 12)")
    parser.add_argument("--com-tol", type=float, default=0.05,
                        help="COM x-tolerance from bar in metres (default 0.05)")
    parser.add_argument("--smoothness", type=float, default=0.5,
                        help="Smoothness penalty weight (default 0.5)")
    parser.add_argument("--reach-weight", type=float, default=10.0,
                        help="Reach objective weight (default 10.0)")
    parser.add_argument("--closed-loop", action="store_true",
                        help="Run closed-loop demo with simulated IMU after planning")
    parser.add_argument("--imu-noise", type=float, default=2.0,
                        help="Simulated IMU noise std dev in degrees (default 2.0)")
    parser.add_argument("--replan-threshold", type=float, default=5.0,
                        help="Tilt error threshold to trigger replan in degrees (default 5.0)")
    parser.add_argument("--collision", action="store_true",
                        help="Check mesh collisions at each waypoint and show in viz")
    args = parser.parse_args()

    spacing_m = args.bar_spacing * INCH_TO_M
    bar_y = args.bar_y if args.bar_y is not None else BAR_Y
    grip_is_right = (args.grip == "right")

    # Place bar 1 at x=0 (in plot frame), bar 2 to the right.
    # If left hand grips bar 1, bar 1 is at the left shoulder's x in
    # the start pose.  For simplicity, bar 1 at x=0.
    bar1_x = 0.0
    bar2_x = bar1_x + spacing_m

    print(f"Bar 1 at x={bar1_x:.3f} m,  Bar 2 at x={bar2_x:.3f} m,  "
          f"y={bar_y:.3f} m")
    print(f"Gripping hand: {args.grip}")
    print(f"COM tolerance: ±{args.com_tol*100:.1f} cm")
    print()

    X_opt, infos, result = plan_trajectory(
        bar1_x, bar2_x, bar_y,
        grip_is_right=grip_is_right,
        n_waypoints=args.n_waypoints,
        com_tolerance=args.com_tol,
        smoothness_weight=args.smoothness,
        reach_weight=args.reach_weight,
    )

    # Print summary
    print("\n── Trajectory summary ──")
    for k, info in enumerate(infos):
        t1g, t2g, t1f, t2f, td = unpack_state(info["state"])
        gt = info["free_gripper_tip"]
        com = info["com"]
        print(f"  WP {k:2d}:  gripper_tip=({gt[0]:+.3f}, {gt[1]:+.3f}) m  "
              f"COM_x={com[0]:+.4f} m  "
              f"grip=({t1g:+6.1f}°,{t2g:+6.1f}°)  "
              f"free=({t1f:+6.1f}°,{t2f:+6.1f}°)  "
              f"tail={td:+5.1f}°")

    final_gt = infos[-1]["free_gripper_tip"]
    bar2 = np.array([bar2_x, bar_y])
    miss = np.linalg.norm(final_gt - bar2)
    print(f"\nFinal gripper-tip miss from bar 2: {miss*1000:.1f} mm")

    # ── Collision check ──────────────────────────────────────────────
    if args.collision:
        print("\n── Collision check ──")
        n_collisions = 0
        for k, info in enumerate(infos):
            t1g, t2g, t1f, t2f, td = unpack_state(info["state"])
            safe, contacts = check_collision_at_waypoint(
                t1g, t2g, t1f, t2f, td, grip_is_right=grip_is_right)
            info["collision_safe"] = safe
            info["collision_contacts"] = contacts
            status = "OK" if safe else f"COLLISION {contacts}"
            if not safe:
                n_collisions += 1
            print(f"  WP {k:2d}: {status}")
        print(f"  {n_collisions}/{len(infos)} waypoints have collisions")
    else:
        for info in infos:
            info["collision_safe"] = True
            info["collision_contacts"] = []

    # ── Closed-loop demo ──────────────────────────────────────────────
    if args.closed_loop:
        planner_kwargs = dict(
            com_tolerance=args.com_tol,
            smoothness_weight=args.smoothness,
            reach_weight=args.reach_weight,
        )
        ctrl, step_log = run_closed_loop_demo(
            X_opt, infos, bar1_x, bar2_x, bar_y, grip_is_right,
            noise_deg=args.imu_noise,
            replan_threshold_deg=args.replan_threshold,
            planner_kwargs=planner_kwargs,
        )
        # Use the (possibly replanned) trajectory for visualisation
        X_opt = ctrl.trajectory
        infos = ctrl.infos

    # ── Visualisation ────────────────────────────────────────────────
    visualise_trajectory(X_opt, infos, bar1_x, bar2_x, bar_y,
                         grip_is_right, LINK1_LENGTH, LINK2_LENGTH)

    # 3D animated playback
    animate_trajectory_3d(X_opt, infos, bar1_x, bar2_x, bar_y,
                          grip_is_right, LINK1_LENGTH, LINK2_LENGTH,
                          interval_ms=350)


if __name__ == "__main__":
    main()
