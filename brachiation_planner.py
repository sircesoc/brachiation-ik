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
import threading
import time

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

# ── Motor → joint mapping ────────────────────────────────────────────────────
# Each entry: motor_id -> (side, kind, sign)
#   side:  "right" | "left" | "tail"
#   kind:  "shoulder" | "elbow" | None  (None for tail)
#   sign:  +1 if motor's positive rotation matches the planner's positive
#          angular direction, -1 if the motor is mounted mirrored.
# The left arm motors are mirrored relative to the right (mounting convention),
# so they get sign=-1.
MOTOR_TO_JOINT = {
    1:  ("right", "shoulder", +1),
    2:  ("right", "elbow",    +1),
    16: ("left",  "shoulder", -1),
    18: ("left",  "elbow",    -1),
    20: ("tail",  None,       +1),
}

# What planner-state value each joint should have at the user's "zero" pose.
# Default: arms pointing straight up to the bar.
#   Shoulder vertical: alpha1 = +90° → t1 = 90 − K1
#   Elbow straight:    alpha2 =  0°  → t2 =  0 − K2 = −K2
import willy_ik as _willy_ik  # already imported above; re-bind for clarity
T1_AT_VERTICAL = 90.0 - _willy_ik.K1   # ~+62.04°
T2_AT_VERTICAL = -_willy_ik.K2          # ~+31.97°
ZERO_POSE_REFERENCE = {
    ("right", "shoulder"): T1_AT_VERTICAL,
    ("right", "elbow"):    T2_AT_VERTICAL,
    ("left",  "shoulder"): T1_AT_VERTICAL,
    ("left",  "elbow"):    T2_AT_VERTICAL,
    ("tail",  None):       0.0,
}

# Per-joint offset (degrees) added to the motor reading before mapping into
# the planner's state. Initialized to ZERO_POSE_REFERENCE so that motor=0
# (the user's physical zero pose = vertical arms) renders as vertical.
# Press 'z' in --motors-animate / --live-animate to recapture at any pose.
JOINT_OFFSETS_DEG = dict(ZERO_POSE_REFERENCE)


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


# ── Geometric collision dimensions ───────────────────────────────────────────
# Distances in metres. Used both for the planner's collision constraints
# and for the visualisation overlay.
INCH_M            = 0.0254
LINK_HALF_WIDTH_M = 0.625 * INCH_M    # parallelogram links: 1.25" wide
LINK_RADIUS_M     = LINK_HALF_WIDTH_M  # alias for capsule-capsule distances
DISC_RADIUS_M     = 2.0  * INCH_M     # 4" diameter wrist-plate disc → 2" radius
BAR_RADIUS_M      = 0.5  * INCH_M     # treat bars as 1" diameter cylinders


def _inflate_segment_polygon(p1, p2, half_width):
    """Return 4 vertices of a rectangle around segment p1→p2, half_width on
    each side perpendicular to the segment. p1, p2 are (x, y) tuples or
    1D arrays."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x2 - x1, y2 - y1
    L = math.sqrt(dx * dx + dy * dy)
    if L < 1e-9:
        return [(x1, y1), (x1, y1), (x2, y2), (x2, y2)]
    nx, ny = -dy / L, dx / L          # unit normal
    hw = half_width
    return [(x1 + hw * nx, y1 + hw * ny),
            (x2 + hw * nx, y2 + hw * ny),
            (x2 - hw * nx, y2 - hw * ny),
            (x1 - hw * nx, y1 - hw * ny)]


def arm_collision_polygons(t1_deg, t2_deg, shoulder_world, side_is_left,
                            half_width_m=LINK_HALF_WIDTH_M,
                            disc_radius_m=DISC_RADIUS_M):
    """Return a list of (polygon_vertices, kind) tuples representing the
    physical extents of one arm in the world frame. `kind` is one of
    {'link', 'triangle', 'plate', 'disc'} for visual styling.

    polygon_vertices is a list of (x, y) world-frame points.
    """
    a1  = (t1_deg + K1) * _D2R
    a12 = a1 + (t2_deg + K2) * _D2R

    # All positions in arm-local frame.
    M1 = np.array([0.0, 0.0])
    M2 = _M2_OFFSET.copy()
    elbow = np.array([A_M * math.cos(a1), A_M * math.sin(a1)])
    J_orange = elbow.copy()
    J_distal = elbow + _COMP151_DISTAL
    J_purple = elbow + _COMP151_PURPLE
    J4 = elbow + np.array([_FOREARM_DRAW * math.cos(a12),
                            _FOREARM_DRAW * math.sin(a12)])
    J7 = J4 + np.array([_WRIST_PLATE, 0.0])
    ee_base = (J4 + J7) / 2.0

    # Map each local point into the world frame (mirror x for left arm).
    sx = -1.0 if side_is_left else 1.0
    sh = np.asarray(shoulder_world, dtype=float)
    def W(p):
        return (sh[0] + sx * float(p[0]), sh[1] + float(p[1]))

    polys = []

    # 1. Comp139 upper arm (M1 → elbow).
    polys.append((_inflate_segment_polygon(W(M1), W(elbow), half_width_m),
                  "link"))
    # 2. Comp148 passive rod (M2 → J_purple).
    polys.append((_inflate_segment_polygon(W(M2), W(J_purple), half_width_m),
                  "link"))
    # 3. Comp151 triangle — filled.
    polys.append(([W(J_orange), W(J_purple), W(J_distal)], "triangle"))
    # 4. Comp146 forearm (elbow → J4).
    polys.append((_inflate_segment_polygon(W(elbow), W(J4), half_width_m),
                  "link"))
    # 5. Comp147 passive rod (J_distal → J7).
    polys.append((_inflate_segment_polygon(W(J_distal), W(J7), half_width_m),
                  "link"))
    # 6. Comp143 wrist plate (J4 → J7).
    polys.append((_inflate_segment_polygon(W(J4), W(J7), half_width_m),
                  "plate"))
    # 7. Disc at ee_base — runs PERPENDICULAR to the gripper end-effector
    #    (which is vertical in body frame), so in 2D it's a horizontal line
    #    of length 2*R centered on ee_base.
    eb = W(ee_base)
    disc_left  = (eb[0] - disc_radius_m, eb[1])
    disc_right = (eb[0] + disc_radius_m, eb[1])
    polys.append((_inflate_segment_polygon(disc_left, disc_right,
                                            0.005),  # 1 cm visual thickness
                  "disc"))

    return polys


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


# COM_Prediction segment masses & geometry, mirrored here so we can
# differentiate analytically without re-running the function 6× per waypoint.
# Keep these in sync with COM Prediction.py if that file ever changes.
_COMP_L1_IN          = 3.4
_COMP_L2_IN          = 11.8
_COMP_BPO_IN         = 5.99606299
_COMP_M_BODY         = 9.68960433
_COMP_M_TAIL         = 3.9471154
_COMP_M_LINK1        = 2.78627841
_COMP_M_LINK2        = 1.16620275
_COMP_M_GRIPPER      = 4.08915
_COMP_X_GRIP_OFF_IN  = 0.37758314
_COMP_TAIL_OFF_IN    = 15.3379
_COMP_TAIL_STOP_RAD  = math.radians(8.73)
# Polynomial fits for link-2 mass-offset, in DEGREES → inches.
_COMP_AX = (0.0, -0.0001699867194, 0.03098551081, -0.1653118738)        # cubic→constant for x-offset
_COMP_BY = (0.0000009035335573, -0.0002522954645, 0.00184756179, 2.727258043)
_DEG_PER_RAD = 180.0 / math.pi
_COMP_M_TOTAL = (_COMP_M_BODY + _COMP_M_TAIL
                 + 2.0 * (_COMP_M_LINK1 + _COMP_M_LINK2 + _COMP_M_GRIPPER))


def _com_arm_x_and_grads(a1_rad, a12_rad, side):
    """Return (x_link1, x_link2, x_gripper) and their (∂/∂a1, ∂/∂a12)
    derivatives (rad) for one arm — matches COM_Prediction's x-component
    formulas. side ∈ {'right','left'}; left mirrors signs about the body.
    All x-positions in INCHES (and inches/rad for derivatives).
    """
    sgn = -1.0 if side == 'right' else +1.0
    s1, c1   = math.sin(a1_rad),  math.cos(a1_rad)
    s12, c12 = math.sin(a12_rad), math.cos(a12_rad)

    # x_link1 = sgn*BPO + sgn*L1*c1   (right: -BPO - L1*c1; left: +BPO + L1*c1)
    x1       = sgn * _COMP_BPO_IN + sgn * _COMP_L1_IN * c1
    dx1_da1  = -sgn * _COMP_L1_IN * s1

    # Polynomial offsets evaluated in DEGREES; chain to rad via _DEG_PER_RAD.
    a12_deg = a12_rad * _DEG_PER_RAD
    ax3, ax2, ax1, ax0 = _COMP_AX
    by3, by2, by1, by0 = _COMP_BY
    x_off = ((ax3 * a12_deg + ax2) * a12_deg + ax1) * a12_deg + ax0
    y_off = ((by3 * a12_deg + by2) * a12_deg + by1) * a12_deg + by0
    dx_off_da12 = (3*ax3*a12_deg*a12_deg + 2*ax2*a12_deg + ax1) * _DEG_PER_RAD
    dy_off_da12 = (3*by3*a12_deg*a12_deg + 2*by2*a12_deg + by1) * _DEG_PER_RAD

    # x_link2 = sgn*BPO + sgn*L1*c1 + sgn*(L2/2 + y_off)*c12 + sgn*x_off*sin(a12)
    half_L2 = _COMP_L2_IN / 2.0
    x2 = sgn * (_COMP_BPO_IN + _COMP_L1_IN * c1
                + (half_L2 + y_off) * c12 + x_off * s12)
    dx2_da1  = -sgn * _COMP_L1_IN * s1
    dx2_da12 = sgn * (
        dy_off_da12 * c12 - (half_L2 + y_off) * s12
        + dx_off_da12 * s12 + x_off * c12
    )

    # x_gripper = sgn*BPO + sgn*L1*c1 + sgn*L2*c12 + sgn*x_grip_off
    xg       = sgn * (_COMP_BPO_IN + _COMP_L1_IN * c1
                      + _COMP_L2_IN * c12 + _COMP_X_GRIP_OFF_IN)
    dxg_da1  = -sgn * _COMP_L1_IN * s1
    dxg_da12 = -sgn * _COMP_L2_IN * s12

    return (x1, x2, xg), (dx1_da1, dx2_da1, dxg_da1), (dx2_da12, dxg_da12)


def compute_com_world_x_and_grad(state_rad, grip_is_right, bar_pos):
    """Closed-form com_world_x (m) and its 5-vector gradient (m/rad) wrt the
    planner state [t1_grip, t2_grip, t1_free, t2_free, tail] in radians.

    Replaces the 6-call finite-difference Jacobian at the QP linearization
    point. Matches compute_com_world to ~1e-12 m at typical configurations.
    """
    t1g_r, t2g_r, t1f_r, t2f_r, ttail_r = (float(s) for s in state_rad)

    K1_r = math.radians(K1)
    K2_r = math.radians(K2)

    # --- body_center_x via grip-arm gripper-tip kinematics ------------------
    a1g  = t1g_r + K1_r
    a12g = a1g + t2g_r + K2_r
    s1g, c1g   = math.sin(a1g),  math.cos(a1g)
    s12g, c12g = math.sin(a12g), math.cos(a12g)
    ee_x_grip = A_M * c1g + _FOREARM_DRAW * c12g + _WRIST_PLATE / 2.0
    dee_x_dt1g = -A_M * s1g - _FOREARM_DRAW * s12g
    dee_x_dt2g = -_FOREARM_DRAW * s12g
    if grip_is_right:
        body_center_x = float(bar_pos[0]) - ee_x_grip - BODY_PIVOT_OFFSET
        d_bcx_dt1g = -dee_x_dt1g
        d_bcx_dt2g = -dee_x_dt2g
    else:
        body_center_x = float(bar_pos[0]) + ee_x_grip + BODY_PIVOT_OFFSET
        d_bcx_dt1g = +dee_x_dt1g
        d_bcx_dt2g = +dee_x_dt2g

    # --- COM_Prediction inner sum ------------------------------------------
    # COM theta_1 = motor t1 + K1 (absolute shoulder); theta_2 = motor t2 + K2
    # (relative elbow). a12 = a1 + a2_rel.
    if grip_is_right:
        Rt1, Rt2 = t1g_r, t2g_r
        Lt1, Lt2 = t1f_r, t2f_r
    else:
        Rt1, Rt2 = t1f_r, t2f_r
        Lt1, Lt2 = t1g_r, t2g_r
    R_a1, R_a12 = Rt1 + K1_r, Rt1 + K1_r + Rt2 + K2_r
    L_a1, L_a12 = Lt1 + K1_r, Lt1 + K1_r + Lt2 + K2_r

    (xR1, xR2, xRg), (dxR1_da1, dxR2_da1, dxRg_da1), (dxR2_da12, dxRg_da12) = \
        _com_arm_x_and_grads(R_a1, R_a12, side='right')
    (xL1, xL2, xLg), (dxL1_da1, dxL2_da1, dxLg_da1), (dxL2_da12, dxLg_da12) = \
        _com_arm_x_and_grads(L_a1, L_a12, side='left')

    # Tail (body x_com is 0).
    tail_arg = ttail_r + math.radians(TAIL_GUI_OFFSET_DEG) + _COMP_TAIL_STOP_RAD
    x_tail        = _COMP_TAIL_OFF_IN * math.cos(tail_arg)
    dxtail_dttail = -_COMP_TAIL_OFF_IN * math.sin(tail_arg)

    x_moment = (
        _COMP_M_TAIL    * x_tail
        + _COMP_M_LINK1   * xR1 + _COMP_M_LINK2 * xR2 + _COMP_M_GRIPPER * xRg
        + _COMP_M_LINK1   * xL1 + _COMP_M_LINK2 * xL2 + _COMP_M_GRIPPER * xLg
    )
    x_com_in = x_moment / _COMP_M_TOTAL

    # ∂x_moment/∂(motor angles, rad). a12 = a1 + a2_rel, so ∂a12/∂a1 = 1, ∂a12/∂a2 = 1.
    dxm_dRt1 = (_COMP_M_LINK1 * dxR1_da1
                + _COMP_M_LINK2 * (dxR2_da1 + dxR2_da12)
                + _COMP_M_GRIPPER * (dxRg_da1 + dxRg_da12))
    dxm_dRt2 = _COMP_M_LINK2 * dxR2_da12 + _COMP_M_GRIPPER * dxRg_da12
    dxm_dLt1 = (_COMP_M_LINK1 * dxL1_da1
                + _COMP_M_LINK2 * (dxL2_da1 + dxL2_da12)
                + _COMP_M_GRIPPER * (dxLg_da1 + dxLg_da12))
    dxm_dLt2 = _COMP_M_LINK2 * dxL2_da12 + _COMP_M_GRIPPER * dxLg_da12
    dxm_dttail = _COMP_M_TAIL * dxtail_dttail

    inv_M = 1.0 / _COMP_M_TOTAL
    # Map (R, L) → (grip, free) state slots.
    if grip_is_right:
        dxin_dt1g, dxin_dt2g = dxm_dRt1 * inv_M, dxm_dRt2 * inv_M
        dxin_dt1f, dxin_dt2f = dxm_dLt1 * inv_M, dxm_dLt2 * inv_M
    else:
        dxin_dt1g, dxin_dt2g = dxm_dLt1 * inv_M, dxm_dLt2 * inv_M
        dxin_dt1f, dxin_dt2f = dxm_dRt1 * inv_M, dxm_dRt2 * inv_M
    dxin_dttail = dxm_dttail * inv_M

    com_world_x = body_center_x - x_com_in * INCH_TO_M
    grad = np.array([
        d_bcx_dt1g - INCH_TO_M * dxin_dt1g,
        d_bcx_dt2g - INCH_TO_M * dxin_dt2g,
                   - INCH_TO_M * dxin_dt1f,
                   - INCH_TO_M * dxin_dt2f,
                   - INCH_TO_M * dxin_dttail,
    ], dtype=float)
    return com_world_x, grad


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
    start_state_deg=None,    # (t1g, t2g, t1f, t2f, tail) in degrees, or None
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

    # Initial guess: either the supplied start state, or arms reaching UP
    # so the body hangs below the bar.
    # alpha1 ≈ 90° (shoulder pointing up) → t1 = 90 - K1 ≈ 62°
    # alpha2 ≈ 0° (forearm continues up) → t2 = 0 - K2 ≈ 32°
    if start_state_deg is not None:
        t1g0, t2g0, t1f0, t2f0, tail0 = start_state_deg
        x0_single = pack_state(t1g0, t2g0, t1f0, t2f0, tail0)
    else:
        t1_up = 90.0 - K1   # ≈ 62°
        t2_up = 0.0 - K2    # ≈ 32°
        x0_single = pack_state(t1_up, t2_up, t1_up, t2_up, 0.0)
    x0 = np.tile(x0_single, n_waypoints)

    # Build flat bounds. If a start state is supplied, pin WP0 by tightening
    # its per-variable bounds to a degenerate interval — SLSQP handles this
    # much better than redundant equality constraints alongside box bounds.
    # If the measured start violates a per-joint bound (e.g. motor drifted
    # past its logged limit), CLAMP into bounds rather than producing an
    # infeasible problem.
    bounds = list(per_wp_bounds * n_waypoints)
    clamped_start = None
    if start_state_deg is not None:
        x0_pin = pack_state(*start_state_deg)
        clamped = list(x0_pin)
        any_clamp = False
        for j in range(STATE_DIM):
            lo, hi = per_wp_bounds[j]
            if clamped[j] < lo:
                any_clamp = True; clamped[j] = lo
            elif clamped[j] > hi:
                any_clamp = True; clamped[j] = hi
            bounds[j] = (float(clamped[j]), float(clamped[j]))
        if any_clamp:
            clamped_start = tuple(math.degrees(v) for v in clamped)
            orig = start_state_deg
            print(f"[plan] start state clamped to bounds: "
                  f"{tuple(round(o, 2) for o in orig)} → "
                  f"{tuple(round(c, 2) for c in clamped_start)}")
        # Use the (possibly clamped) values for the initial guess too.
        x0_single = np.array(clamped)
        x0 = np.tile(x0_single, n_waypoints)

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
    # When the first waypoint is pinned to the robot's measured state, skip
    # COM constraints on WP0 — the robot's actual state is what it is.
    constraints = []
    com_constrained_wps = (range(1, n_waypoints)
                           if start_state_deg is not None
                           else range(n_waypoints))
    for k in com_constrained_wps:
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

    # WP0 is pinned via degenerate box bounds (above) when start_state_deg
    # is given — no extra equality constraints needed here.

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


def _point_seg_dist_sq(p, q1, q2):
    """Squared min distance from 2D point p to segment q1→q2."""
    d = q2 - q1
    L2 = float(d[0] * d[0] + d[1] * d[1])
    if L2 < 1e-18:
        dx, dy = p[0] - q1[0], p[1] - q1[1]
        return dx * dx + dy * dy
    t = float((p[0] - q1[0]) * d[0] + (p[1] - q1[1]) * d[1]) / L2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = q1[0] + t * d[0]
    cy = q1[1] + t * d[1]
    dx, dy = p[0] - cx, p[1] - cy
    return dx * dx + dy * dy


def _seg_seg_dist_sq(p1, p2, q1, q2):
    """Squared min distance between two 2D segments."""
    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = q2[0] - q1[0], q2[1] - q1[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) > 1e-12:
        rx, ry = q1[0] - p1[0], q1[1] - p1[1]
        s = (rx * d2y - ry * d2x) / denom
        t = (rx * d1y - ry * d1x) / denom
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            return 0.0
    return min(
        _point_seg_dist_sq(p1, q1, q2),
        _point_seg_dist_sq(p2, q1, q2),
        _point_seg_dist_sq(q1, p1, p2),
        _point_seg_dist_sq(q2, p1, p2),
    )


def _arm_landmarks(state_rad, grip_is_right, bar1):
    """Return key kinematic points (world frame) for both arms, given
    a planner state vector in radians.

    Returns dict with arrays for: grip_shoulder, grip_elbow, grip_J4,
    grip_J7, grip_disc, free_shoulder, free_elbow, free_J4, free_J7,
    free_disc. Positions follow the same conventions used elsewhere in
    the planner (right arm: +x = world +x; left arm mirrored).
    """
    t1g_r, t2g_r, t1f_r, t2f_r, _ttail = (float(s) for s in state_rad)
    t1g, t2g = math.degrees(t1g_r), math.degrees(t2g_r)
    t1f, t2f = math.degrees(t1f_r), math.degrees(t2f_r)

    grip_sh = shoulder_from_bar(bar1, t1g, t2g,
                                grip_is_left=not grip_is_right)
    _, free_sh = body_and_free_shoulder(grip_sh, grip_is_right)

    def _local_arm_points(t1_deg, t2_deg):
        a1 = math.radians(t1_deg + K1)
        a12 = a1 + math.radians(t2_deg + K2)
        elbow = np.array([A_M * math.cos(a1), A_M * math.sin(a1)])
        j4 = elbow + np.array([_FOREARM_DRAW * math.cos(a12),
                                _FOREARM_DRAW * math.sin(a12)])
        j7 = j4 + np.array([_WRIST_PLATE, 0.0])
        disc = (j4 + j7) / 2.0
        m2       = _M2_OFFSET.copy()
        j_purple = elbow + _COMP151_PURPLE
        j_distal = elbow + _COMP151_DISTAL
        return elbow, j4, j7, disc, m2, j_purple, j_distal

    (g_elbow, g_j4, g_j7, g_disc,
     g_m2, g_j_purple, g_j_distal) = _local_arm_points(t1g, t2g)
    (f_elbow, f_j4, f_j7, f_disc,
     f_m2, f_j_purple, f_j_distal) = _local_arm_points(t1f, t2f)

    # Mirror the left arm's local x → world: left x_local = -x_world_offset.
    def _to_world(local, shoulder_world, side_is_left):
        if side_is_left:
            return shoulder_world + np.array([-local[0], local[1]])
        return shoulder_world + local

    grip_is_left = not grip_is_right
    free_is_left = grip_is_right

    return {
        "grip_shoulder": np.asarray(grip_sh, dtype=float),
        "grip_M2":       _to_world(g_m2,       grip_sh, grip_is_left),
        "grip_elbow":    _to_world(g_elbow,    grip_sh, grip_is_left),
        "grip_J_purple": _to_world(g_j_purple, grip_sh, grip_is_left),
        "grip_J_distal": _to_world(g_j_distal, grip_sh, grip_is_left),
        "grip_J4":       _to_world(g_j4,       grip_sh, grip_is_left),
        "grip_J7":       _to_world(g_j7,       grip_sh, grip_is_left),
        "grip_disc":     _to_world(g_disc,     grip_sh, grip_is_left),
        "free_shoulder": np.asarray(free_sh, dtype=float),
        "free_M2":       _to_world(f_m2,       free_sh, free_is_left),
        "free_elbow":    _to_world(f_elbow,    free_sh, free_is_left),
        "free_J_purple": _to_world(f_j_purple, free_sh, free_is_left),
        "free_J_distal": _to_world(f_j_distal, free_sh, free_is_left),
        "free_J4":       _to_world(f_j4,       free_sh, free_is_left),
        "free_J7":       _to_world(f_j7,       free_sh, free_is_left),
        "free_disc":     _to_world(f_disc,     free_sh, free_is_left),
    }


# ── Minimal planner: no path prescription, tail unlocked ─────────────────────

def plan_trajectory_minimal(
    bar1_x, bar2_x, bar_y,
    grip_is_right=True,
    n_waypoints=12,
    L1=LINK1_LENGTH, L2=LINK2_LENGTH,
    right_shoulder_lim=(-70.9, 46.0),
    right_elbow_lim=(-90.4, 140.0),
    left_shoulder_lim=(-46.0, 70.9),
    left_elbow_lim=(-140.0, 90.4),
    tail_lim=(-60.0, 60.0),       # tail UNLOCKED — counterweight is free
    smoothness_weight=1.0,
    monotone_x=True,
    start_state_deg=None,
    bar_clearance_m=0.02,         # min y-clearance below bar2 except at catch
    com_band_m=0.03,              # |COM_x - bar1_x| during the direct phase
    com_band_late_m=0.05,         # |COM_x - bar1_x| during the dip/catch phase
    com_relax_t=0.70,             # 0..1; t at which COM band starts widening
    pin_wp0=False,                # pin WP0 to start_state_deg (or default)
    track_weight=20.0,            # soft tracking cost weight on Cartesian target
    dip_depth_m=0.10,             # peak dip depth (m below bar)
    dip_skew_k=4.0,               # >1 shifts the dip peak late (sin(π·t^k))
    enable_collision=True,        # geometric collision constraints (link/disc/bar)
):
    """U-shape brachiation reach: dip below bars, traverse, rise to catch.

    Free hand follows a Cartesian U-curve from start to bar2:
        x(t) = bar1_x + (bar2_x - bar1_x) · smoothstep(t)
        y(t) = bar_y - dip_depth · sin(π·t)^p
    Tracking is a SOFT cost (weight `track_weight`); the optimizer is free
    to deviate when joint geometry or COM band makes exact tracking
    infeasible.

    Hard constraints:
      • Final waypoint: free gripper tip exactly at bar 2.
      • Every waypoint: COM_x within ±com_band_m of bar1_x.
      • Every waypoint except final: free_hand_y ≤ bar_y − bar_clearance_m
        (gripper catches from below, so it must stay under the bars).
      • Monotone free_hand_x progression (no backtracking).
      • Joint bounds (incl. unlocked tail).
    """
    bar1 = np.array([bar1_x, bar_y])
    bar2 = np.array([bar2_x, bar_y])

    free_is_right = not grip_is_right

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

    def deg_to_rad_bounds(lim):
        return (math.radians(lim[0]), math.radians(lim[1]))

    per_wp_bounds = [
        deg_to_rad_bounds(grip_t1_lim),
        deg_to_rad_bounds(grip_t2_lim),
        deg_to_rad_bounds(free_t1_lim),
        deg_to_rad_bounds(free_t2_lim),
        deg_to_rad_bounds(tail_lim),
    ]

    # Two-stage solve when collisions are enabled: stage 1 solves without
    # collision constraints to get a smooth U trajectory, then stage 2
    # warm-starts from it with the full collision set. WP 0 is pinned to
    # a known non-colliding starting pose so stage 1 cannot drift into a
    # collision at the start.
    NON_COLLIDING_WP0 = (70.0, 65.0, -6.0, 136.0, 0.0)
    if enable_collision:
        wp0_for_pin = (start_state_deg if start_state_deg is not None
                        else NON_COLLIDING_WP0)
        print(f"[minimal] Stage 1: warm start with WP0 pinned to "
              f"{tuple(round(v,1) for v in wp0_for_pin)} (non-colliding) …")
        X_warm, _infos_w, _res_w = plan_trajectory_minimal(
            bar1_x, bar2_x, bar_y,
            grip_is_right=grip_is_right,
            n_waypoints=n_waypoints,
            L1=L1, L2=L2,
            right_shoulder_lim=right_shoulder_lim,
            right_elbow_lim=right_elbow_lim,
            left_shoulder_lim=left_shoulder_lim,
            left_elbow_lim=left_elbow_lim,
            tail_lim=tail_lim,
            smoothness_weight=smoothness_weight,
            monotone_x=monotone_x,
            start_state_deg=wp0_for_pin,
            bar_clearance_m=bar_clearance_m,
            com_band_m=com_band_m,
            com_band_late_m=com_band_late_m,
            com_relax_t=com_relax_t,
            pin_wp0=True,                # pin WP0 to wp0_for_pin
            track_weight=track_weight,
            dip_depth_m=dip_depth_m,
            dip_skew_k=dip_skew_k,
            enable_collision=False,
        )
        x0 = X_warm.reshape(-1).copy()
        x0_single = x0[:STATE_DIM]
        # Carry the WP0 pin into stage 2 so collisions can never re-enter
        # at the start.
        if start_state_deg is None:
            start_state_deg = wp0_for_pin
        pin_wp0 = True
    else:
        # Cold start: linearly interpolate in joint space between a
        # "swing-start" pose and a "catch" pose. Both verified to satisfy
        # all collision constraints (free arm clear of grip arm) and are
        # near the COM band centre.
        swing_start = (70.0, 65.0, -6.0, 136.0, 0.0)
        catch_pose  = (69.0, 84.0, -50.0, 109.0, -25.0)
        if start_state_deg is not None:
            swing_start = tuple(start_state_deg)
        x0 = np.zeros(n_waypoints * STATE_DIM)
        for k in range(n_waypoints):
            s = k / max(n_waypoints - 1, 1)
            wp = tuple(swing_start[j] * (1 - s) + catch_pose[j] * s
                       for j in range(STATE_DIM))
            x0[k * STATE_DIM:(k + 1) * STATE_DIM] = pack_state(*wp)
        x0_single = x0[:STATE_DIM]

    bounds = list(per_wp_bounds * n_waypoints)
    if pin_wp0 and start_state_deg is not None:
        # User pinned the start — pin WP0 to it exactly.
        x0_pin = pack_state(*start_state_deg)
        clamped = list(x0_pin)
        for j in range(STATE_DIM):
            lo, hi = per_wp_bounds[j]
            clamped[j] = max(lo, min(hi, clamped[j]))
            bounds[j] = (float(clamped[j]), float(clamped[j]))

    def free_tip(state):
        t1g, t2g, t1f, t2f, _td = unpack_state(state)
        grip_sh = shoulder_from_bar(bar1, t1g, t2g,
                                    grip_is_left=not grip_is_right)
        _, free_sh = body_and_free_shoulder(grip_sh, grip_is_right)
        return free_hand_world(free_sh, t1f, t2f, free_is_left=grip_is_right)

    # Cartesian target schedule for the free hand:
    #  • x progresses smoothstep from bar1 to bar2.
    #  • y stays just below the bar (at bar_clearance_m) for the direct
    #    phase, then dips deeper near the catch via sin(π·t^k). Larger k
    #    pushes the dip peak later (k=4 → peak near t≈0.84).
    targets = []
    for k in range(n_waypoints):
        t = k / max(n_waypoints - 1, 1)
        s = t * t * (3.0 - 2.0 * t)                       # smoothstep(t)
        x_t = bar1_x + (bar2_x - bar1_x) * s
        bump = max(math.sin(math.pi * (t ** dip_skew_k)), 0.0)
        # Baseline = clearance below bar; bump adds extra dip (max=dip_depth_m).
        extra = max(dip_depth_m - bar_clearance_m, 0.0) * bump
        y_t = bar_y - bar_clearance_m - extra
        targets.append((x_t, y_t))
    targets[-1] = (bar2_x, bar_y)                         # exact catch

    def cost(x_flat):
        X = x_flat.reshape(n_waypoints, STATE_DIM)
        diffs = X[1:] - X[:-1]
        c = smoothness_weight * float(np.sum(diffs * diffs))
        for k in range(n_waypoints):
            tip = free_tip(X[k])
            tx, ty = targets[k]
            c += track_weight * ((tip[0] - tx) ** 2 + (tip[1] - ty) ** 2)
        return float(c)

    constraints = []

    # Per-waypoint COM band: tight in the direct phase, loose during the
    # dip/catch phase. Linearly ramps from com_band_m to com_band_late_m
    # for t ∈ [com_relax_t, 1]; equality (eq, no band) when com_band_m == 0
    # AND the late band is also 0.
    def com_band_at(k):
        t = k / max(n_waypoints - 1, 1)
        if t <= com_relax_t:
            return com_band_m
        s = (t - com_relax_t) / max(1.0 - com_relax_t, 1e-9)
        return com_band_m + s * (com_band_late_m - com_band_m)

    com_wps = (range(1, n_waypoints) if (pin_wp0 and start_state_deg is not None)
               else range(n_waypoints))
    use_eq = (com_band_m is None or com_band_m <= 0.0) and \
             (com_band_late_m is None or com_band_late_m <= 0.0)
    if use_eq:
        for k in com_wps:
            def make_com_eq(kk):
                def con(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    cx, _g = compute_com_world_x_and_grad(X[kk], grip_is_right, bar1)
                    return cx - bar1_x
                def jac(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    _cx, g = compute_com_world_x_and_grad(X[kk], grip_is_right, bar1)
                    J = np.zeros((n_waypoints, STATE_DIM))
                    J[kk] = g
                    return J.reshape(-1)
                return con, jac
            c, j = make_com_eq(k)
            constraints.append({"type": "eq", "fun": c, "jac": j})
    else:
        for k in com_wps:
            band = com_band_at(k)
            def make_com_lo(kk, b=band):
                def con(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    cx, _g = compute_com_world_x_and_grad(X[kk], grip_is_right, bar1)
                    return cx - (bar1_x - b)
                return con
            def make_com_hi(kk, b=band):
                def con(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    cx, _g = compute_com_world_x_and_grad(X[kk], grip_is_right, bar1)
                    return (bar1_x + b) - cx
                return con
            constraints.append({"type": "ineq", "fun": make_com_lo(k)})
            constraints.append({"type": "ineq", "fun": make_com_hi(k)})

    # Final tip exactly at bar2 (both x and y).
    def final_tip_x(x_flat):
        X = x_flat.reshape(n_waypoints, STATE_DIM)
        return float(free_tip(X[-1])[0] - bar2[0])

    def final_tip_y(x_flat):
        X = x_flat.reshape(n_waypoints, STATE_DIM)
        return float(free_tip(X[-1])[1] - bar2[1])

    constraints.append({"type": "eq", "fun": final_tip_x})
    constraints.append({"type": "eq", "fun": final_tip_y})

    # Monotone progress in free-hand x: prevent backtracking.
    if monotone_x:
        for k in range(n_waypoints - 1):
            def make_mono(kk):
                def con(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    return float(free_tip(X[kk + 1])[0] - free_tip(X[kk])[0])
                return con
            constraints.append({"type": "ineq", "fun": make_mono(k)})

    # Bar clearance: gripper points up and catches bar2 from below, so the
    # free-hand tip must stay at least `bar_clearance_m` below bar height
    # at every waypoint except the catch. Forces the swing arc to dip
    # under the bars instead of arcing over them.
    if bar_clearance_m is not None and bar_clearance_m > 0.0:
        for k in range(n_waypoints - 1):
            def make_clearance(kk):
                def con(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    return float((bar_y - bar_clearance_m) - free_tip(X[kk])[1])
                return con
            constraints.append({"type": "ineq", "fun": make_clearance(k)})

    # Collision constraints — only the three pairs that matter in 3D:
    #   1. Same-arm parallel link pair within the parallelogram
    #      (Comp139 ∥ Comp148, Comp146 ∥ Comp147 — width 1.25" each).
    #   2. Body line (between shoulders) vs bar1, bar2.
    #   3. Each arm's links vs the body box hanging from the shoulders.
    # Inter-arm collisions are not included: the 11.7" bilateral spacing
    # makes them physically impossible.
    if enable_collision:
        d_link_link = 2.0 * LINK_RADIUS_M           # 1.25" (sum of half-widths)
        d2_ll = d_link_link ** 2
        d_body_bar = BAR_RADIUS_M + LINK_HALF_WIDTH_M  # body line ≈ link width
        d2_body_bar = d_body_bar ** 2
        d_arm_body = LINK_HALF_WIDTH_M             # arm capsule half-width
        d2_arm_body = d_arm_body ** 2
        bar1_pt = np.asarray(bar1, dtype=float)
        bar2_pt = np.asarray(bar2, dtype=float)
        body_height_m = 4.0 * INCH_TO_M             # body box height (~4")

        def _body_box_corners(L):
            """Return (top_left, top_right, bottom_left, bottom_right) in
            world frame given the landmarks dict."""
            gs = L["grip_shoulder"]; fs = L["free_shoulder"]
            x_lo = min(gs[0], fs[0]); x_hi = max(gs[0], fs[0])
            y_top = (gs[1] + fs[1]) / 2.0
            y_bot = y_top - body_height_m
            return (np.array([x_lo, y_top]), np.array([x_hi, y_top]),
                    np.array([x_lo, y_bot]), np.array([x_hi, y_bot]))

        for k in range(n_waypoints):
            is_catch = (k == n_waypoints - 1)

            # 1. Same-arm parallel link pairs — the two real parallelograms:
            #    Comp139 (M1 → elbow)         ∥ Comp148 (M2 → J_purple)
            #    Comp146 (elbow → J4 forearm) ∥ Comp147 (J_distal → J7)
            # Both arms are checked. Parallelogram geometry keeps these
            # parallel; the check enforces centre-line distance ≥ 1.25".
            def make_parallel_pair(kk, arm, p1k, p2k, q1k, q2k):
                def con(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    L = _arm_landmarks(X[kk], grip_is_right, bar1)
                    d2 = _seg_seg_dist_sq(L[p1k], L[p2k], L[q1k], L[q2k])
                    return float(d2 - d2_ll)
                return con

            for arm in ("grip", "free"):
                # Comp139 (shoulder→elbow) ∥ Comp148 (M2→J_purple)
                constraints.append({"type": "ineq",
                                    "fun": make_parallel_pair(
                                        k, arm,
                                        f"{arm}_shoulder", f"{arm}_elbow",
                                        f"{arm}_M2",       f"{arm}_J_purple")})
                # Comp146 (elbow→J4) ∥ Comp147 (J_distal→J7)
                constraints.append({"type": "ineq",
                                    "fun": make_parallel_pair(
                                        k, arm,
                                        f"{arm}_elbow",    f"{arm}_J4",
                                        f"{arm}_J_distal", f"{arm}_J7")})

            # 2. Body line vs bars. The "body line" is the segment between
            #    the two shoulders (top of the body box).
            def make_body_vs_bar(kk, bar_pt):
                def con(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    L = _arm_landmarks(X[kk], grip_is_right, bar1)
                    d2 = _point_seg_dist_sq(bar_pt,
                                             L["grip_shoulder"],
                                             L["free_shoulder"])
                    return float(d2 - d2_body_bar)
                return con

            constraints.append({"type": "ineq",
                                "fun": make_body_vs_bar(k, bar1_pt)})
            if not is_catch:
                constraints.append({"type": "ineq",
                                    "fun": make_body_vs_bar(k, bar2_pt)})

            # 3. Arm vs body box. Each arm has 3 segments (upper, forearm,
            #    wrist plate). The body box has 4 edges. Skip the upper-arm
            #    constraint that's connected at the shoulder (zero distance
            #    by construction). Check forearm and wrist-plate vs the
            #    body box's four edges.
            def make_seg_vs_body_edge(kk, arm, p_key, q_key, edge_idx):
                def con(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    L = _arm_landmarks(X[kk], grip_is_right, bar1)
                    tl, tr, bl, br = _body_box_corners(L)
                    edges = [(tl, tr), (tr, br), (br, bl), (bl, tl)]
                    e1, e2 = edges[edge_idx]
                    d2 = _seg_seg_dist_sq(L[p_key], L[q_key], e1, e2)
                    return float(d2 - d2_arm_body)
                return con

            for arm in ("grip", "free"):
                for (p, q) in [(f"{arm}_elbow", f"{arm}_J4"),    # forearm
                                (f"{arm}_J4",    f"{arm}_J7")]:    # wrist plate
                    for edge_idx in range(4):
                        constraints.append({
                            "type": "ineq",
                            "fun": make_seg_vs_body_edge(k, arm, p, q, edge_idx),
                        })

            # 4. Inter-arm collisions — the two arms must never intersect in
            #    the swing plane, even though the bilateral spacing makes 3D
            #    collision impossible. Disc treated as a horizontal line of
            #    length 2*DISC_RADIUS at ee_base (perpendicular to the ee
            #    line, as the user specified).
            d2_disc_link = (LINK_HALF_WIDTH_M) ** 2
            d2_disc_disc = (0.005) ** 2   # 5 mm minimum separation

            def _disc_segment(L, side):
                eb = L[f"{side}_disc"]
                return (np.array([eb[0] - DISC_RADIUS_M, eb[1]]),
                        np.array([eb[0] + DISC_RADIUS_M, eb[1]]))

            def make_inter_seg(kk, fp1, fp2, gp1, gp2, thr_sq):
                def con(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    L = _arm_landmarks(X[kk], grip_is_right, bar1)
                    d2 = _seg_seg_dist_sq(L[fp1], L[fp2], L[gp1], L[gp2])
                    return float(d2 - thr_sq)
                return con

            def make_inter_disc_seg(kk, disc_side, op1, op2, thr_sq):
                def con(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    L = _arm_landmarks(X[kk], grip_is_right, bar1)
                    da, db = _disc_segment(L, disc_side)
                    d2 = _seg_seg_dist_sq(da, db, L[op1], L[op2])
                    return float(d2 - thr_sq)
                return con

            def make_inter_disc_disc(kk, thr_sq):
                def con(x_flat):
                    X = x_flat.reshape(n_waypoints, STATE_DIM)
                    L = _arm_landmarks(X[kk], grip_is_right, bar1)
                    fa, fb = _disc_segment(L, "free")
                    ga, gb = _disc_segment(L, "grip")
                    d2 = _seg_seg_dist_sq(fa, fb, ga, gb)
                    return float(d2 - thr_sq)
                return con

            # Link-vs-link pairs (forearm + upper, both arms)
            link_pairs = [
                ("free_shoulder", "free_elbow", "grip_shoulder", "grip_elbow"),
                ("free_shoulder", "free_elbow", "grip_elbow",    "grip_J4"),
                ("free_elbow",    "free_J4",    "grip_shoulder", "grip_elbow"),
                ("free_elbow",    "free_J4",    "grip_elbow",    "grip_J4"),
            ]
            for (a, b, c, d) in link_pairs:
                constraints.append({"type": "ineq",
                                    "fun": make_inter_seg(k, a, b, c, d, d2_ll)})

            # Disc-vs-other-arm-link pairs (disc as horizontal segment)
            for disc_side, other in [("free", "grip"), ("grip", "free")]:
                for (p, q) in [(f"{other}_shoulder", f"{other}_elbow"),
                                (f"{other}_elbow",    f"{other}_J4"),
                                (f"{other}_J4",       f"{other}_J7")]:
                    constraints.append({
                        "type": "ineq",
                        "fun": make_inter_disc_seg(k, disc_side, p, q, d2_disc_link),
                    })

            # Disc-vs-disc (skip at catch — discs end up at the catch bar)
            if not is_catch:
                constraints.append({"type": "ineq",
                                    "fun": make_inter_disc_disc(k, d2_disc_disc)})

    print(f"[minimal] Optimising {n_waypoints} waypoints × {STATE_DIM} DOF "
          f"= {n_waypoints * STATE_DIM} variables …")

    result = minimize(
        cost,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9, "disp": True},
    )
    print(f"[minimal] Optimiser exit: {result.message}  "
          f"(success={result.success})")

    X_opt = result.x.reshape(n_waypoints, STATE_DIM)

    # Build infos to match plan_trajectory's output shape.
    infos = []
    for k in range(n_waypoints):
        t1g, t2g, t1f, t2f, td = unpack_state(X_opt[k])
        grip_sh = shoulder_from_bar(bar1, t1g, t2g,
                                    grip_is_left=not grip_is_right)
        _, free_sh = body_and_free_shoulder(grip_sh, grip_is_right)
        fh = free_hand_world(free_sh, t1f, t2f, free_is_left=grip_is_right)
        com_x, com_y = compute_com_world(t1g, t2g, t1f, t2f, td,
                                         grip_is_right, bar1)
        infos.append({
            "grip_shoulder": grip_sh,
            "free_shoulder": free_sh,
            "free_hand": fh,
            "free_gripper_tip": fh.copy(),
            "com": np.array([com_x, com_y]),
            "target": fh.copy(),
            "state": X_opt[k].copy(),
        })

    return X_opt, infos, result


# ── Visualisation ────────────────────────────────────────────────────────────

def draw_robot_at_waypoint(ax, info, grip_is_right, L1, L2, alpha=1.0,
                           bar_pos=None, show_meshes=False,
                           show_widths=True):
    """Draw the full robot for one waypoint.

    Uses the Willy double-parallelogram arm style from the HTML visualizer.
    If show_meshes=True, also draws translucent convex hull outlines of
    the actual STL meshes. If show_widths=True (default), overlays the
    physical extents of each parallelogram link (1.25" wide), the triangle,
    and the wrist-plate disc (4" diameter, edge-on).
    """
    if show_widths:
        # Filled polygons: link extents + triangle + disc.
        from matplotlib.patches import Polygon as _MplPolygon
        t1g, t2g, t1f, t2f, _td = unpack_state(info["state"])
        gs = info["grip_shoulder"]
        fs = info["free_shoulder"]
        grip_is_left = not grip_is_right

        def _draw_arm(t1, t2, sh, side_is_left, link_color, disc_color):
            polys = arm_collision_polygons(t1, t2, sh, side_is_left)
            for verts, kind in polys:
                if kind == "triangle":
                    face = link_color; ec = link_color
                elif kind == "plate":
                    face = link_color; ec = link_color
                elif kind == "disc":
                    face = disc_color; ec = disc_color
                else:
                    face = link_color; ec = link_color
                p = _MplPolygon(verts, closed=True,
                                facecolor=face, edgecolor=ec,
                                alpha=0.18 * alpha, linewidth=0.6,
                                zorder=1)
                ax.add_patch(p)

        _draw_arm(t1g, t2g, gs, grip_is_left,
                  link_color="#5fb3ff", disc_color="#0891b2")
        _draw_arm(t1f, t2f, fs, not grip_is_left,
                  link_color="#ff9f5f", disc_color="#d97706")

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


class WitIMU(IMUInterface):
    """Real WIT-Motion IMU (e.g. JY-901, WT901) over USB-serial."""

    def __init__(self, port=None, baud=9600, axis="roll",
                 offset_deg=0.0, invert=False):
        from imu_reader import IMUReader, autodetect_port
        if port is None:
            port = autodetect_port() or "/dev/ttyUSB0"
        self._reader = IMUReader(port=port, baud=baud)
        self._axis = axis
        self._offset_deg = offset_deg
        self._invert = invert
        self._connected = False
        self._port = port

    def connect(self):
        self._reader.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and self._reader.last_update == 0.0:
            time.sleep(0.01)
        self._connected = True

    def disconnect(self):
        self._reader.stop()
        self._connected = False

    def is_connected(self):
        return self._connected

    def read(self):
        r, p, y = self._reader.angle_degrees
        wx, wy, wz = self._reader.angular_velocity
        if self._axis == "roll":
            tilt_deg, rate = r, wx
        elif self._axis == "pitch":
            tilt_deg, rate = p, wy
        elif self._axis == "yaw":
            tilt_deg, rate = y, wz
        else:
            raise ValueError(f"unknown imu axis: {self._axis!r}")
        if self._invert:
            tilt_deg = -tilt_deg
            rate = -rate
        tilt_deg += self._offset_deg
        return IMUReading(
            tilt_rad=math.radians(tilt_deg),
            tilt_rate_rad_s=rate,
            timestamp_s=time.monotonic(),
        )

    def advance(self):
        pass


class InnerStabilizer:
    """Fast inner-loop body-tilt stabilizer.

    Runs at the motor command rate (200-500 Hz). Reads IMU tilt + rate,
    computes a tail-angle correction with a PD law, and adds it to whatever
    tail reference the planner provides. The outer planner / MPC stays
    unchanged; this just rides on top of its commanded angles.

    Sign convention: positive tilt = COM drifted to +x of the gripped bar
    (see pendulum_tilt_angle). To restore, COM must move toward -x. Whether
    that means increasing or decreasing the tail angle depends on the
    tail's mass-offset geometry (see compute_com_world: increasing tail_gui
    rotates the tail mass, shifting world COM_x). Default tail_sign=-1.0
    assumes increasing tail moves world COM_x in +x; flip to +1.0 if your
    geometry is mirrored.

    Usage:
        stab = InnerStabilizer(kp_deg_per_rad=30.0, kd_deg_s_per_rad=5.0)
        ...
        q_ref = controller.current_target_angles()
        q_cmd = stab.apply(q_ref, imu.read(), dt=0.005)
        send_to_motors(q_cmd)
    """

    def __init__(self,
                 kp_deg_per_rad=30.0,
                 kd_deg_s_per_rad=5.0,
                 tail_sign=-1.0,
                 max_correction_deg=20.0,
                 tail_limits_deg=(-90.0, 90.0),
                 tilt_lowpass_alpha=1.0,
                 use_imu_rate=True):
        self.kp = float(kp_deg_per_rad)
        self.kd = float(kd_deg_s_per_rad)
        self.tail_sign = float(tail_sign)
        self.max_correction_deg = float(max_correction_deg)
        self.tail_lo, self.tail_hi = (float(tail_limits_deg[0]),
                                      float(tail_limits_deg[1]))
        self.alpha = float(tilt_lowpass_alpha)
        self.use_imu_rate = bool(use_imu_rate)

        self._tilt_filt_rad = None
        self._prev_tilt_rad = None
        self._last_correction_deg = 0.0
        self._last_tilt_err_rad = 0.0
        self._last_rate_rad_s = 0.0

    def reset(self):
        self._tilt_filt_rad = None
        self._prev_tilt_rad = None
        self._last_correction_deg = 0.0
        self._last_tilt_err_rad = 0.0
        self._last_rate_rad_s = 0.0

    def apply(self, q_ref, imu_reading, dt=None, tilt_ref_rad=0.0):
        """Return a corrected joint-angle dict.

        q_ref: dict from ReplanningController.current_target_angles().
               Must contain key 'tail_gui_deg'. Other joints pass through
               unchanged.
        imu_reading: IMUReading. Uses tilt_rad and (optionally)
               tilt_rate_rad_s.
        dt: timestep in seconds. Required when use_imu_rate=False so the
            stabilizer can finite-difference the tilt itself.
        tilt_ref_rad: reference tilt (rad). Set to 0 for the static-reach
            plan; pass the planner's expected tilt if you want the
            stabilizer to track a swinging reference.
        """
        tilt = float(imu_reading.tilt_rad)

        # Optional first-order low-pass to suppress IMU noise.
        if self._tilt_filt_rad is None:
            self._tilt_filt_rad = tilt
        else:
            self._tilt_filt_rad = (self.alpha * tilt
                                   + (1.0 - self.alpha) * self._tilt_filt_rad)
        tilt_used = self._tilt_filt_rad

        if self.use_imu_rate:
            rate = float(imu_reading.tilt_rate_rad_s)
        else:
            if self._prev_tilt_rad is None or dt is None or dt <= 0.0:
                rate = 0.0
            else:
                rate = (tilt_used - self._prev_tilt_rad) / float(dt)
        self._prev_tilt_rad = tilt_used

        tilt_err = tilt_used - float(tilt_ref_rad)
        delta_tail_deg = self.tail_sign * (self.kp * tilt_err
                                            + self.kd * rate)

        # Clamp correction magnitude so a single spike can't slam the tail.
        if delta_tail_deg > self.max_correction_deg:
            delta_tail_deg = self.max_correction_deg
        elif delta_tail_deg < -self.max_correction_deg:
            delta_tail_deg = -self.max_correction_deg

        out = dict(q_ref)
        tail_ref = float(out.get("tail_gui_deg", 0.0))
        tail_cmd = tail_ref + delta_tail_deg
        if tail_cmd < self.tail_lo:
            tail_cmd = self.tail_lo
        elif tail_cmd > self.tail_hi:
            tail_cmd = self.tail_hi
        out["tail_gui_deg"] = tail_cmd

        self._last_correction_deg = delta_tail_deg
        self._last_tilt_err_rad = tilt_err
        self._last_rate_rad_s = rate
        return out

    def diagnostics(self):
        """Return last-step diagnostics for logging / plotting."""
        return {
            "tilt_err_deg": math.degrees(self._last_tilt_err_rad),
            "tilt_rate_deg_s": math.degrees(self._last_rate_rad_s),
            "tail_correction_deg": self._last_correction_deg,
        }


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
                 planner_kwargs=None,
                 motor_reader=None,
                 joint_replan_threshold_deg=8.0,
                 mpc_mode=False,
                 qp_mpc=False,
                 qp_slack_weight=1000.0,
                 qp_track_weight=10.0):
        self.bar1_x = bar1_x
        self.bar2_x = bar2_x
        self.bar_y = bar_y
        self.grip_is_right = grip_is_right
        self.imu = imu
        self.n_waypoints = n_waypoints
        self.replan_threshold_rad = math.radians(replan_threshold_deg)
        self.planner_kwargs = planner_kwargs or {}
        self.motor_reader = motor_reader
        self.joint_replan_threshold_deg = joint_replan_threshold_deg
        self.mpc_mode = mpc_mode
        self.qp_mpc = qp_mpc
        self.qp_slack_weight = qp_slack_weight
        self.qp_track_weight = qp_track_weight
        # Cache QPReplanner instances by remaining-horizon size. Setup is
        # ~20ms per size; once built, each replan is sub-millisecond.
        self._qp_replanners = {}
        self._last_qp_solve_ms = 0.0
        # Optional fixed reference trajectory (e.g. the SLSQP cold-start
        # arc). When set, the QP tracks a sliding window into this anchor
        # rather than tracking its own previous solution — keeps the
        # trajectory shape (arc under bar1 then up to bar2) consistent.
        self.anchor = None
        self.anchor_idx = 0
        # Current measured IMU tilt (rad), updated each step before replanning.
        # Passed to QPReplanner.replan() via current_tilt_rad. The goal
        # constraint rotates the body-frame tip by this angle around bar1 so
        # the QP picks a final joint config whose rendered (in-world) tip
        # lands on bar2. The COM constraint stays in body frame on purpose
        # (gravity restores tilt when the commanded body-COM is at bar1).
        self.current_imu_tilt_rad = 0.0

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
          1. Compare measured tilt + (optional) measured joint angles vs plan
          2. Replan if either error exceeds its threshold
          3. Advance to next waypoint
        """
        if self.done or self.current_wp >= len(self.planned_tilts):
            self.done = True
            return {"status": "done", "wp": self.current_wp}

        measured_tilt = imu_reading.tilt_rad
        planned_tilt = self.planned_tilts[self.current_wp]
        error = measured_tilt - planned_tilt
        self.measured_tilts.append(measured_tilt)
        # Make the COM/goal constraints aware of the current body tilt so
        # the planner adapts joint configs to put the world-frame COM under
        # bar1 (and the world-frame final tip on bar2).
        self.current_imu_tilt_rad = measured_tilt

        # Joint-level feedback (if a motor reader is wired up).
        measured_state = None
        max_joint_err_deg = None
        if self.motor_reader is not None:
            snap = self.motor_reader.snapshot()
            measured_state = measured_planner_state(snap, self.grip_is_right)
            joints_complete = all(
                v is not None for k, v in measured_state.items()
                if k != "tail_gui_deg"
            )
            if joints_complete:
                planned_wp = self.trajectory[self.current_wp]
                errs = joint_error_deg(measured_state, planned_wp)
                joint_errs_only = [abs(v) for k, v in errs.items()
                                   if v is not None and k != "tail_gui_deg"]
                if joint_errs_only:
                    max_joint_err_deg = max(joint_errs_only)

        status = {
            "wp": self.current_wp,
            "planned_tilt_deg": math.degrees(planned_tilt),
            "measured_tilt_deg": math.degrees(measured_tilt),
            "error_deg": math.degrees(error),
            "max_joint_err_deg": max_joint_err_deg,
            "replanned": False,
            "replan_reason": None,
            "status": "tracking",
        }

        tilt_trigger  = abs(error) > self.replan_threshold_rad
        joint_trigger = (max_joint_err_deg is not None
                         and max_joint_err_deg > self.joint_replan_threshold_deg)

        if self.mpc_mode or tilt_trigger or joint_trigger:
            remaining = self.n_waypoints - self.current_wp
            if remaining > 2:
                reason = []
                if self.mpc_mode:
                    reason.append("mpc")
                if tilt_trigger:
                    reason.append(f"tilt err {math.degrees(error):+.1f}° "
                                  f"> {math.degrees(self.replan_threshold_rad):.1f}°")
                if joint_trigger:
                    reason.append(f"joint err {max_joint_err_deg:+.1f}° "
                                  f"> {self.joint_replan_threshold_deg:.1f}°")
                reason_str = "; ".join(reason)
                if not self.mpc_mode:
                    print(f"\n[REPLAN] WP {self.current_wp}: {reason_str}. "
                          f"Replanning {remaining} remaining waypoints...")

                self._replan_from_current(measured_tilt,
                                           measured_state=measured_state)
                status["replanned"] = True
                status["replan_reason"] = reason_str
                self.replan_count += 1

        self.current_wp += 1
        if self.current_wp >= self.n_waypoints:
            self.done = True
            status["status"] = "done"

        return status

    def _per_joint_bounds_rad(self):
        """Per-joint (lo, hi) radian bounds in planner state order
        (t1_grip, t2_grip, t1_free, t2_free, tail), pulled from
        self.planner_kwargs and the grip side."""
        kw = self.planner_kwargs
        rsh = kw.get("right_shoulder_lim", (-70.9, 46.0))
        rel = kw.get("right_elbow_lim",   (-90.4, 140.0))
        lsh = kw.get("left_shoulder_lim", (-46.0, 70.9))
        lel = kw.get("left_elbow_lim",    (-140.0, 90.4))
        tail = kw.get("tail_lim", (0.0, 0.0))
        if self.grip_is_right:
            grip_t1, grip_t2 = rsh, rel
            free_t1, free_t2 = lsh, lel
        else:
            grip_t1, grip_t2 = lsh, lel
            free_t1, free_t2 = rsh, rel
        return [(math.radians(grip_t1[0]), math.radians(grip_t1[1])),
                (math.radians(grip_t2[0]), math.radians(grip_t2[1])),
                (math.radians(free_t1[0]), math.radians(free_t1[1])),
                (math.radians(free_t2[0]), math.radians(free_t2[1])),
                (math.radians(tail[0]),    math.radians(tail[1]))]

    def _make_com_constraint(self):
        """Build a com_constraint for QPReplanner.

        Constrains the **plan-frame** COM (joint-only, before any IMU
        rotation) to lie under the gripping bar. This is the classical
        pendulum setup: under any disturbance tilt theta, world-frame COM
        offset is L·sin(theta), gravity creates a restoring torque, and
        IMU tilt naturally decays toward zero. (Goal constraint stays in
        world frame so the rendered tip lands at bar2.)
        """
        com_tol = self.planner_kwargs.get("com_tolerance", 0.01)
        bar1 = np.array([self.bar1_x, self.bar_y])
        grip_is_right = self.grip_is_right

        def eval_fn(state_rad, tilt_rad=0.0, _bar1=bar1, _gir=grip_is_right):
            # COM constraint stays in body frame regardless of tilt — see
            # docstring above. tilt_rad accepted for API uniformity.
            del tilt_rad  # intentionally unused
            return compute_com_world_x_and_grad(state_rad, _gir, _bar1)

        return {
            "lo": self.bar1_x - 3.0 * com_tol,
            "hi": self.bar1_x + com_tol,
            "eval_fn": eval_fn,
        }

    def _make_goal_constraint(self):
        """Build a goal_constraint for QPReplanner: hard equality that the
        WORLD-frame free-arm gripper tip (= plan-frame tip rotated by the
        current IMU tilt around bar1) lands at bar2. So the planner picks
        a final joint config whose RENDERED tip hits bar2 even when the
        body is tilted by IMU-reported disturbance."""
        bar1 = np.array([self.bar1_x, self.bar_y])
        target_xy = np.array([self.bar2_x, self.bar_y])
        grip_is_right = self.grip_is_right
        eps = math.radians(0.05)

        def _free_tip_plan(state_rad):
            t1g = math.degrees(state_rad[0]); t2g = math.degrees(state_rad[1])
            t1f = math.degrees(state_rad[2]); t2f = math.degrees(state_rad[3])
            grip_sh = shoulder_from_bar(bar1, t1g, t2g,
                                         grip_is_left=not grip_is_right)
            _, free_sh = body_and_free_shoulder(grip_sh, grip_is_right)
            return np.asarray(free_hand_world(free_sh, t1f, t2f,
                                              free_is_left=grip_is_right))

        def _rotate_around_bar1(p, cos_t, sin_t):
            dx = p[0] - bar1[0]; dy = p[1] - bar1[1]
            return np.array([bar1[0] + dx * cos_t - dy * sin_t,
                             bar1[1] + dx * sin_t + dy * cos_t])

        def eval_fn(state_rad, tilt_rad=0.0, _eps=eps):
            # Rotate the body-frame tip by the explicitly-supplied IMU
            # tilt around bar1 to get its world-frame position. This is
            # the critical bit: the QP picks a final joint config whose
            # RENDERED tip (after the body has tilted) lands on bar2.
            cos_t = math.cos(tilt_rad); sin_t = math.sin(tilt_rad)
            tip_p0 = _free_tip_plan(state_rad)
            tip_w0 = _rotate_around_bar1(tip_p0, cos_t, sin_t)
            J = np.zeros((2, STATE_DIM), dtype=float)
            for j in range(STATE_DIM):
                s = np.array(state_rad, dtype=float)
                s[j] += _eps
                tip_p = _free_tip_plan(s)
                tip_w = _rotate_around_bar1(tip_p, cos_t, sin_t)
                J[:, j] = (tip_w - tip_w0) / _eps
            return tip_w0, J

        return {"target_xy": target_xy, "eval_fn": eval_fn}

    def _qp_replan(self, x_ref, current_state_rad):
        """Sub-millisecond OSQP-based replan. Tracks `x_ref`, or, if
        `self.anchor` is set, a sliding window into the anchor trajectory.
        Passes the latest measured IMU tilt explicitly to the replanner so
        the goal constraint enforces the rendered (in-world) tip on bar2.
        """
        from qp_replanner import QPReplanner
        n = x_ref.shape[0]
        # Sliding-window anchor reference (keeps the arc shape stable).
        if self.anchor is not None and len(self.anchor) > 0:
            last = len(self.anchor) - 1
            x_ref = np.array([
                self.anchor[min(self.anchor_idx + k, last)] for k in range(n)
            ], dtype=float)
        replanner = self._qp_replanners.get(n)
        if replanner is None:
            replanner = QPReplanner(
                n_waypoints=n,
                per_wp_bounds_rad=self._per_joint_bounds_rad(),
                w_track=self.qp_track_weight,
                slack_weight=self.qp_slack_weight,
                com_constraint=self._make_com_constraint(),
                goal_constraint=self._make_goal_constraint() if n >= 2 else None,
            )
            self._qp_replanners[n] = replanner
        X_new, status = replanner.replan(
            x_ref, current_state_rad,
            current_tilt_rad=float(self.current_imu_tilt_rad),
        )
        self._last_qp_solve_ms = replanner.last_solve_ms
        return X_new, status

    def _replan_from_current(self, measured_tilt, measured_state=None):
        """Replan remaining waypoints. If `measured_state` is given (and
        complete), it is used as the pinned start state for the replan."""
        remaining = self.n_waypoints - self.current_wp
        grip_point = np.array([self.bar1_x, self.bar_y])

        start_state_deg = None
        if measured_state is not None and all(
            v is not None for k, v in measured_state.items()
            if k != "tail_gui_deg"
        ):
            start_state_deg = (
                measured_state["t1_grip_deg"], measured_state["t2_grip_deg"],
                measured_state["t1_free_deg"], measured_state["t2_free_deg"],
                0.0,  # tail not connected; held straight
            )

        # Fast path: OSQP-tracked QP replan. Tracks the existing trajectory
        # (which came from the SLSQP cold-start) with smoothness + bounds
        # + WP0 pin. Sub-millisecond on this hardware.
        if (self.qp_mpc
                and start_state_deg is not None
                and self.trajectory is not None
                and remaining >= 2):
            x_ref = self.trajectory[self.current_wp:].copy()
            current_rad = np.array([math.radians(v) for v in start_state_deg])
            X_new, qp_status = self._qp_replan(x_ref, current_rad)
            if X_new is not None:
                self.trajectory[self.current_wp:] = X_new
                # Update planned tilts using cheap COM compute on the new wps.
                for i in range(remaining):
                    wp_idx = self.current_wp + i
                    if wp_idx >= len(self.planned_tilts):
                        break
                    t1g, t2g, t1f, t2f, td = unpack_state(X_new[i])
                    cx, cy = compute_com_world(t1g, t2g, t1f, t2f, td,
                                               self.grip_is_right,
                                               np.array([self.bar1_x, self.bar_y]))
                    self.planned_tilts[wp_idx] = pendulum_tilt_angle(
                        np.array([cx, cy]), grip_point)
                if not self.mpc_mode:
                    print(f"[REPLAN-QP] {remaining} wps in "
                          f"{self._last_qp_solve_ms:.2f} ms ({qp_status})")
                return
            # If the QP fell over (very unlikely with feasible bounds),
            # fall through to the SLSQP path below.
            print(f"[REPLAN-QP] failed ({qp_status}); falling back to SLSQP")

        X_opt, infos, result = plan_trajectory(
            self.bar1_x, self.bar2_x, self.bar_y,
            grip_is_right=self.grip_is_right,
            n_waypoints=remaining,
            start_state_deg=start_state_deg,
            **self.planner_kwargs,
        )

        # Fallback: if a hard pin made the problem infeasible, retry without
        # the pin (use measured state only as initial guess).
        if not result.success and start_state_deg is not None:
            if not self.mpc_mode:
                print(f"[REPLAN] hard-pin failed ({result.message}); "
                      f"retrying without pin...")
            X_opt, infos, result = plan_trajectory(
                self.bar1_x, self.bar2_x, self.bar_y,
                grip_is_right=self.grip_is_right,
                n_waypoints=remaining,
                start_state_deg=None,
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
            if not self.mpc_mode:
                print(f"[REPLAN] Success — {remaining} waypoints replanned"
                      + (" from measured state." if start_state_deg else "."))
        else:
            print(f"[REPLAN] Failed (both attempts): {result.message}")

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
                         planner_kwargs=None, imu=None, step_period_s=0.0,
                         motor_reader=None):
    """Run closed-loop execution of the planned trajectory.

    If `imu` is None, a SimulatedIMU is built from the planned trajectory
    (with optional Gaussian noise). Pass a real IMUInterface (e.g. WitIMU)
    to drive it from hardware. `step_period_s` paces the loop in wall-clock
    time when running with real hardware. If `motor_reader` is provided,
    measured joint angles are sampled at each step and the per-joint error
    vs planned is printed (read-only, no behavior change).
    """
    grip_point = (bar1_x, bar_y)

    if imu is None:
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
    next_tick = time.monotonic()
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

        if motor_reader is not None:
            wp_idx = min(status["wp"], len(X_opt) - 1)
            planned = X_opt[wp_idx]
            measured = measured_planner_state(motor_reader.snapshot(),
                                              grip_is_right)
            errs = joint_error_deg(measured, planned)
            def _e(k):
                v = errs[k]
                return "  ----" if v is None else f"{v:+6.2f}"
            print(
                f"        joint err°:  "
                f"t1_grip={_e('t1_grip_deg')}  "
                f"t2_grip={_e('t2_grip_deg')}  "
                f"t1_free={_e('t1_free_deg')}  "
                f"t2_free={_e('t2_free_deg')}  "
                f"tail={_e('tail_gui_deg')}"
            )

        imu.advance()
        if step_period_s > 0:
            next_tick += step_period_s
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

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


# ── Motor feedback helpers (Robstride RS-02/RS-03) ───────────────────────────

def measured_planner_state(snapshot, grip_is_right):
    """Convert a MotorReader snapshot into the planner's 5-vector
    (t1_grip_deg, t2_grip_deg, t1_free_deg, t2_free_deg, tail_gui_deg).

    Joints not yet received are returned as None.
    """
    by_joint = {}
    for mid, (side, kind, sign) in MOTOR_TO_JOINT.items():
        s = snapshot.get(mid)
        if s is None or s.last_update == 0.0:
            by_joint[(side, kind)] = None
        else:
            by_joint[(side, kind)] = (sign * s.joint_deg
                                      + JOINT_OFFSETS_DEG.get((side, kind), 0.0))

    grip_side = "right" if grip_is_right else "left"
    free_side = "left"  if grip_is_right else "right"

    t1_grip = by_joint.get((grip_side, "shoulder"))
    t2_grip = by_joint.get((grip_side, "elbow"))
    t1_free = by_joint.get((free_side, "shoulder"))
    t2_free = by_joint.get((free_side, "elbow"))
    tail    = by_joint.get(("tail", None))

    return {
        "t1_grip_deg": t1_grip,
        "t2_grip_deg": t2_grip,
        "t1_free_deg": t1_free,
        "t2_free_deg": t2_free,
        "tail_gui_deg": tail,
    }


def set_zero_from_snapshot(snapshot):
    """Capture current motor readings as the new zero pose.

    Updates JOINT_OFFSETS_DEG so that "motor at this pose" maps to the
    planner-state values in ZERO_POSE_REFERENCE (default: vertical arms,
    tail straight). Returns the captured offsets.
    """
    captured = {}
    for mid, (side, kind, sign) in MOTOR_TO_JOINT.items():
        s = snapshot.get(mid)
        if s is None or s.last_update == 0.0:
            continue
        target = ZERO_POSE_REFERENCE.get((side, kind), 0.0)
        # We want: sign * joint_deg + offset == target at the zero pose.
        offset = target - sign * s.joint_deg
        JOINT_OFFSETS_DEG[(side, kind)] = offset
        captured[(side, kind)] = offset
    return captured


def joint_error_deg(measured, planned_state):
    """Per-joint error (measured - planned) in degrees. None if not yet received."""
    t1g, t2g, t1f, t2f, td = unpack_state(planned_state)
    out = {}
    for key, planned_deg in [
        ("t1_grip_deg", t1g),
        ("t2_grip_deg", t2g),
        ("t1_free_deg", t1f),
        ("t2_free_deg", t2f),
    ]:
        m = measured[key]
        out[key] = None if m is None else (m - planned_deg)
    m_td = measured["tail_gui_deg"]
    out["tail_gui_deg"] = None if m_td is None else (m_td - td)
    return out


class SegmentPool:
    """Lazy pool of matplotlib Line2D objects so we can render an arbitrary
    number of robot segments per frame. The first call creates the lines;
    subsequent frames reuse and hide any extras."""
    def __init__(self, ax):
        self._ax = ax
        self._lines = []

    def draw(self, segments_xy):
        """Update lines to match `segments_xy` (list of (xs, ys) tuples).
        Returns the list of Line2D objects actually used this frame."""
        for i, (xs, ys) in enumerate(segments_xy):
            if i >= len(self._lines):
                line, = self._ax.plot([], [], color="#374151", linewidth=1.6)
                self._lines.append(line)
            self._lines[i].set_data(xs, ys)
            self._lines[i].set_visible(True)
        # Hide any leftovers from previous frames.
        for j in range(len(segments_xy), len(self._lines)):
            self._lines[j].set_visible(False)
        return self._lines


def run_motors_test(config_path, log_file, only_buses, rate_hz, grip_is_right):
    """Stream motor readings + planner-state conversion until Ctrl+C."""
    from motor_reader import MotorReader
    reader = MotorReader(config_path=config_path, log_file=log_file,
                         only_buses=only_buses)
    print(f"motors: {reader.motor_ids}  (grip_is_right={grip_is_right})")
    reader.start()
    period = 1.0 / max(rate_hz, 0.1)
    try:
        while True:
            snap = reader.snapshot()
            cols = []
            for mid in reader.motor_ids:
                s = snap[mid]
                side, kind, _ = MOTOR_TO_JOINT.get(mid, ("?", "?", 1))
                stale = "*" if s.is_stale(0.5) else " "
                fault = (" " + ",".join(s.fault)) if s.fault else ""
                label = f"{side[:1].upper()}{kind[:2] if kind else 'tl'}"
                cols.append(
                    f"M{mid:>2}({label}){stale}"
                    f"{s.joint_deg:+7.2f}° v={s.vel_rad_s:+5.2f} "
                    f"t={s.tor_nm:+5.2f}Nm T={s.temp_c:4.1f}°{fault}"
                )
            ms = measured_planner_state(snap, grip_is_right)
            def fmt(d):
                return "  ----" if d is None else f"{d:+7.2f}"
            print("  |  ".join(cols))
            print(
                f"        planner-state°:  "
                f"t1_grip={fmt(ms['t1_grip_deg'])}  "
                f"t2_grip={fmt(ms['t2_grip_deg'])}  "
                f"t1_free={fmt(ms['t1_free_deg'])}  "
                f"t2_free={fmt(ms['t2_free_deg'])}  "
                f"tail={fmt(ms['tail_gui_deg'])}"
            )
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        reader.stop()


def _build_pose_info(t1g, t2g, t1f, t2f, tail_deg, grip_is_right, bar1):
    """Build the info dict (grip_shoulder, free_shoulder, com, state) from
    a measured/known set of motor angles in degrees."""
    grip_shoulder = shoulder_from_bar(bar1, t1g, t2g,
                                      grip_is_left=not grip_is_right)
    _, free_shoulder = body_and_free_shoulder(grip_shoulder, grip_is_right)
    com_x, com_y = compute_com_world(t1g, t2g, t1f, t2f, tail_deg,
                                     grip_is_right, bar1)
    return {
        "grip_shoulder": grip_shoulder,
        "free_shoulder": free_shoulder,
        "com": np.array([com_x, com_y]),
        "state": pack_state(t1g, t2g, t1f, t2f, tail_deg),
        "collision_safe": True,
        "collision_contacts": [],
    }


def run_motors_animate(config_path, log_file, only_buses, grip_is_right,
                       bar_y=BAR_Y, fps=20, apply_pendulum_tilt=False):
    """Live 2D animation that mirrors the measured motor joint angles."""
    from motor_reader import MotorReader
    reader = MotorReader(config_path=config_path, log_file=log_file,
                         only_buses=only_buses)
    reader.start()

    bar1_x = 0.0
    bar1 = np.array([bar1_x, bar_y])
    L1, L2 = LINK1_LENGTH, LINK2_LENGTH

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title(
        "Motor Feedback — Live  (press 'z' to set zero)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Live motor feedback   grip={'right' if grip_is_right else 'left'}"
                 f"   tilt={'on' if apply_pendulum_tilt else 'off'}   "
                 f"(press 'z' to set zero)")

    def on_key(event):
        if event.key == 'z':
            captured = set_zero_from_snapshot(reader.snapshot())
            if not captured:
                print("[set zero] no motor data yet — pose the robot and try again.")
                return
            print("\n[set zero] captured at current pose:")
            for (side, kind), off in captured.items():
                label = f"{side}/{kind}" if kind else side
                print(f"  {label:>16s}: offset = {off:+7.2f}°")
    fig.canvas.mpl_connect('key_press_event', on_key)

    bar_half = 0.04
    ax.plot([bar1_x, bar1_x], [bar_y - bar_half, bar_y + bar_half],
            color="brown", linewidth=6, solid_capstyle="round")

    reach = L1 + L2 + GRIPPER_DRAW_LENGTH + 0.05
    ax.set_xlim(bar1_x - reach - 0.1, bar1_x + reach + 0.1)
    ax.set_ylim(bar_y - reach - 0.05, bar_y + 0.15)

    pool = SegmentPool(ax)
    com_dot, = ax.plot([], [], "D", color="#0891b2", markersize=9)

    info_text = ax.text(0.02, 0.98, "waiting for motor data…",
                        transform=ax.transAxes, fontsize=9,
                        family="monospace", va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    def update(_frame):
        snap = reader.snapshot()
        ms = measured_planner_state(snap, grip_is_right)

        missing = [k for k, v in ms.items()
                   if v is None and k != "tail_gui_deg"]
        if missing:
            info_text.set_text(
                "waiting for motor data:\n  " +
                "\n  ".join(f"{k}: ----" for k in missing)
            )
            return pool._lines + [com_dot, info_text]

        t1g = ms["t1_grip_deg"]; t2g = ms["t2_grip_deg"]
        t1f = ms["t1_free_deg"]; t2f = ms["t2_free_deg"]
        # Tail held straight (motor 20 not connected).
        tail_deg = 0.0

        info = _build_pose_info(t1g, t2g, t1f, t2f, tail_deg, grip_is_right, bar1)

        bar_for_tilt = (bar1_x, bar_y) if apply_pendulum_tilt else None
        segments = get_robot_segments(info, grip_is_right, L1, L2,
                                      bar_pos=bar_for_tilt)

        # Last segment is the COM single-point; everything else are robot lines.
        pool.draw(segments[:-1])
        com_xs, com_ys = segments[-1]
        com_dot.set_data([com_xs[0]], [com_ys[0]])

        stale = [mid for mid, s in snap.items() if s.is_stale(0.5)]
        stale_str = f"  STALE: {stale}" if stale else ""
        info_text.set_text(
            f"grip shoulder t1: {t1g:+7.1f}°\n"
            f"grip elbow    t2: {t2g:+7.1f}°\n"
            f"free shoulder t1: {t1f:+7.1f}°\n"
            f"free elbow    t2: {t2f:+7.1f}°\n"
            f"tail:             {tail_deg:+7.1f}° (fixed)\n"
            f"COM offset from bar: {(info['com'][0] - bar1_x)*1000:+.1f} mm{stale_str}"
        )
        return pool._lines + [com_dot, info_text]

    interval_ms = max(20, int(1000.0 / max(fps, 1)))
    anim = FuncAnimation(fig, update, interval=interval_ms, blit=False,
                         cache_frame_data=False)
    try:
        plt.show()
    finally:
        reader.stop()
    return anim


def run_imu_test(port=None, baud=9600, axis="roll", offset_deg=0.0,
                 invert=False, rate_hz=10.0):
    """Stream live IMU tilt readings until Ctrl+C."""
    imu = WitIMU(port=port, baud=baud, axis=axis,
                 offset_deg=offset_deg, invert=invert)
    imu.connect()
    print(f"IMU test mode — port={imu._port}  axis={axis}  "
          f"offset={offset_deg:+.2f}°  invert={invert}")
    print("Hold the robot at known orientations. Ctrl+C to exit.\n")
    print(f"{'time(s)':>8}  {'roll':>8}  {'pitch':>8}  {'yaw':>8}  "
          f"{'tilt':>8}  {'rate(°/s)':>10}")
    period = 1.0 / max(rate_hz, 0.1)
    t0 = time.monotonic()
    try:
        while True:
            r, p, y = imu._reader.angle_degrees
            reading = imu.read()
            tilt_deg = math.degrees(reading.tilt_rad)
            rate_deg_s = math.degrees(reading.tilt_rate_rad_s)
            t = time.monotonic() - t0
            print(f"{t:8.2f}  {r:+8.2f}  {p:+8.2f}  {y:+8.2f}  "
                  f"{tilt_deg:+8.2f}  {rate_deg_s:+10.2f}")
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        imu.disconnect()


def run_imu_animate(port, baud, axis, offset_deg, invert, grip_is_right,
                    bar_y=BAR_Y, fps=20):
    """Live animation: rest pose rotated around grip point by IMU tilt."""
    imu = WitIMU(port=port, baud=baud, axis=axis,
                 offset_deg=offset_deg, invert=invert)
    imu.connect()

    bar1_x = 0.0
    bar1 = np.array([bar1_x, bar_y])
    L1, L2 = LINK1_LENGTH, LINK2_LENGTH

    # Rest pose: arms vertical (pointing up to the bar), tail straight.
    info = _build_pose_info(T1_AT_VERTICAL, T2_AT_VERTICAL,
                            T1_AT_VERTICAL, T2_AT_VERTICAL, 0.0,
                            grip_is_right, bar1)
    base_segments = get_robot_segments(info, grip_is_right, L1, L2, bar_pos=None)
    grip_point = np.array([bar1_x, bar_y])

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title("IMU Tilt — Live")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Body tilt from IMU   axis={axis}"
                 f"{'  (inverted)' if invert else ''}"
                 f"   offset={offset_deg:+.2f}°")

    bar_half = 0.04
    ax.plot([bar1_x, bar1_x], [bar_y - bar_half, bar_y + bar_half],
            color="brown", linewidth=6, solid_capstyle="round")
    ax.axvline(bar1_x, color="red", linestyle=":", alpha=0.4)

    reach = L1 + L2 + GRIPPER_DRAW_LENGTH + 0.05
    ax.set_xlim(bar1_x - reach - 0.2, bar1_x + reach + 0.2)
    ax.set_ylim(bar_y - reach - 0.2, bar_y + 0.2)

    pool = SegmentPool(ax)
    com_dot, = ax.plot([], [], "D", color="#0891b2", markersize=9)

    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        fontsize=11, family="monospace", va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    def update(_frame):
        reading = imu.read()
        tilt_rad = reading.tilt_rad
        # apply_tilt_to_segments rotates by -angle. Negate to *visualise* lean
        # rather than correct it.
        rotated = apply_tilt_to_segments(base_segments, -tilt_rad, grip_point)
        pool.draw(rotated[:-1])
        com_xs, com_ys = rotated[-1]
        com_dot.set_data([com_xs[0]], [com_ys[0]])
        info_text.set_text(
            f"tilt:  {math.degrees(tilt_rad):+7.2f}°\n"
            f"rate:  {math.degrees(reading.tilt_rate_rad_s):+7.2f}°/s"
        )
        return pool._lines + [com_dot, info_text]

    interval_ms = max(20, int(1000.0 / max(fps, 1)))
    anim = FuncAnimation(fig, update, interval=interval_ms, blit=False,
                         cache_frame_data=False)
    try:
        plt.show()
    finally:
        imu.disconnect()
    return anim


def run_live_animate(config_path, log_file, only_buses, grip_is_right,
                     imu_kwargs, bar_y=BAR_Y, fps=20):
    """Live animation driven by BOTH motors (pose) and IMU (body tilt)."""
    from motor_reader import MotorReader
    reader = MotorReader(config_path=config_path, log_file=log_file,
                         only_buses=only_buses)
    imu = WitIMU(**imu_kwargs)

    reader.start()
    imu.connect()

    bar1_x = 0.0
    bar1 = np.array([bar1_x, bar_y])
    grip_point = np.array([bar1_x, bar_y])
    L1, L2 = LINK1_LENGTH, LINK2_LENGTH

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title(
        "Motors + IMU — Live  (press 'z' to set zero)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Live: motors → pose, IMU → tilt   "
                 f"grip={'right' if grip_is_right else 'left'}   "
                 f"axis={imu_kwargs.get('axis', 'roll')}   "
                 f"(press 'z' to set zero)")

    def on_key(event):
        if event.key == 'z':
            captured = set_zero_from_snapshot(reader.snapshot())
            if not captured:
                print("[set zero] no motor data yet — pose the robot and try again.")
                return
            print("\n[set zero] captured at current pose:")
            for (side, kind), off in captured.items():
                label = f"{side}/{kind}" if kind else side
                print(f"  {label:>16s}: offset = {off:+7.2f}°")
    fig.canvas.mpl_connect('key_press_event', on_key)

    bar_half = 0.04
    ax.plot([bar1_x, bar1_x], [bar_y - bar_half, bar_y + bar_half],
            color="brown", linewidth=6, solid_capstyle="round")
    ax.axvline(bar1_x, color="red", linestyle=":", alpha=0.4)

    reach = L1 + L2 + GRIPPER_DRAW_LENGTH + 0.05
    ax.set_xlim(bar1_x - reach - 0.2, bar1_x + reach + 0.2)
    ax.set_ylim(bar_y - reach - 0.2, bar_y + 0.2)

    pool = SegmentPool(ax)
    com_dot, = ax.plot([], [], "D", color="#0891b2", markersize=9)

    info_text = ax.text(0.02, 0.98, "waiting…", transform=ax.transAxes,
                        fontsize=10, family="monospace", va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    def update(_frame):
        snap = reader.snapshot()
        ms = measured_planner_state(snap, grip_is_right)
        reading = imu.read()
        tilt_rad = reading.tilt_rad

        missing = [k for k, v in ms.items()
                   if v is None and k != "tail_gui_deg"]
        if missing:
            info_text.set_text(
                f"tilt: {math.degrees(tilt_rad):+7.2f}°  (IMU live)\n"
                "waiting for motors:\n  " +
                "\n  ".join(f"{k}: ----" for k in missing)
            )
            return pool._lines + [com_dot, info_text]

        t1g = ms["t1_grip_deg"]; t2g = ms["t2_grip_deg"]
        t1f = ms["t1_free_deg"]; t2f = ms["t2_free_deg"]
        # Tail held straight (motor 20 not connected).
        tail_deg = 0.0

        info = _build_pose_info(t1g, t2g, t1f, t2f, tail_deg, grip_is_right, bar1)
        base_segs = get_robot_segments(info, grip_is_right, L1, L2, bar_pos=None)
        rotated = apply_tilt_to_segments(base_segs, -tilt_rad, grip_point)

        pool.draw(rotated[:-1])
        com_xs, com_ys = rotated[-1]
        com_dot.set_data([com_xs[0]], [com_ys[0]])

        stale = [mid for mid, s in snap.items() if s.is_stale(0.5)]
        stale_str = f"  STALE: {stale}" if stale else ""
        info_text.set_text(
            f"tilt:  {math.degrees(tilt_rad):+7.2f}°   "
            f"rate: {math.degrees(reading.tilt_rate_rad_s):+6.1f}°/s\n"
            f"grip shoulder t1: {t1g:+7.1f}°\n"
            f"grip elbow    t2: {t2g:+7.1f}°\n"
            f"free shoulder t1: {t1f:+7.1f}°\n"
            f"free elbow    t2: {t2f:+7.1f}°\n"
            f"tail:             {tail_deg:+7.1f}° (fixed)\n"
            f"COM offset from bar: {(info['com'][0] - bar1_x)*1000:+.1f} mm{stale_str}"
        )
        return pool._lines + [com_dot, info_text]

    interval_ms = max(20, int(1000.0 / max(fps, 1)))
    anim = FuncAnimation(fig, update, interval=interval_ms, blit=False,
                         cache_frame_data=False)
    try:
        plt.show()
    finally:
        reader.stop()
        imu.disconnect()
    return anim


# ── MPC replan visualization (no motors) ────────────────────────────────────

def run_mpc_viz(grip_is_right, imu_kwargs, bar1_x, bar2_x, bar_y, n_waypoints,
                planner_kwargs,
                qp_mpc=True, qp_slack_weight=1000.0, qp_track_weight=10.0,
                replan_threshold_deg=5.0,
                step_period_s=0.1, fps=20, advance_wp=4):
    """Live viz of MPC replanning driven by a real IMU. No motors.

    Cold-starts an SLSQP trajectory, then steps through it at
    `step_period_s`/wp. At each step, reads the IMU; if tilt error vs the
    planned tilt exceeds the threshold (or `qp_mpc` forces it every step),
    replans from the current planned waypoint as the start state. The
    body is rendered at the current planned waypoint, rotated around the
    grip point by the measured IMU tilt.
    """
    imu = WitIMU(**imu_kwargs)
    imu.connect()

    L1, L2 = LINK1_LENGTH, LINK2_LENGTH
    bar1 = np.array([bar1_x, bar_y])
    grip_point = np.array([bar1_x, bar_y])

    print("Planning initial trajectory…")
    pkw = dict(planner_kwargs or {})
    X_opt, infos, result = plan_trajectory(
        bar1_x, bar2_x, bar_y,
        grip_is_right=grip_is_right,
        n_waypoints=n_waypoints,
        **pkw,
    )
    print(f"  success={result.success}; final tip miss = "
          f"{np.linalg.norm(infos[-1]['free_gripper_tip']-np.array([bar2_x,bar_y]))*1000:.1f} mm")

    ctrl = ReplanningController(
        bar1_x, bar2_x, bar_y, grip_is_right, imu,
        n_waypoints=n_waypoints,
        replan_threshold_deg=replan_threshold_deg,
        planner_kwargs=pkw,
        motor_reader=None,
        mpc_mode=qp_mpc,
        qp_mpc=qp_mpc,
        qp_slack_weight=qp_slack_weight,
        qp_track_weight=qp_track_weight,
    )
    ctrl.trajectory = X_opt.copy()
    ctrl.infos = list(infos)
    ctrl.planned_tilts = [pendulum_tilt_angle(info["com"], grip_point)
                          for info in infos]
    # Pin the QP MPC's tracking reference to the SLSQP cold-start (the
    # arc-under-bar trajectory). The anchor slides forward by `advance_wp`
    # each iter so the QP tracks the *future* of the original arc, not
    # whatever its own previous solution drifted into.
    ctrl.anchor = X_opt.copy()
    ctrl.anchor_idx = 0

    stop_event = threading.Event()
    target_xy = np.array([bar2_x, bar_y])
    arrive_tol_m = 0.005   # 5 mm: stop when the free-tip is within this of bar 2

    # Shared "current sim pose" that the viz thread reads. Starts at
    # WP 0 of the initial plan (= the planner's start state).
    sim_pose_box = [tuple(unpack_state(ctrl.trajectory[0]))]
    last_status = {"replanned": False, "iter": 0, "tilt_err_deg": 0.0,
                   "replan_count": 0, "qp_solve_ms": 0.0,
                   "tip_miss_mm": 0.0}

    def _free_tip_at(state_deg):
        t1g, t2g, t1f, t2f, _ = state_deg
        gs = shoulder_from_bar(bar1, t1g, t2g,
                                 grip_is_left=not grip_is_right)
        _, fs = body_and_free_shoulder(gs, grip_is_right)
        return np.asarray(free_hand_world(fs, t1f, t2f,
                                           free_is_left=grip_is_right))

    def step_loop():
        sim_pose_deg = sim_pose_box[0]
        iters = 0
        max_iters = 500
        try:
            while not stop_event.is_set() and iters < max_iters:
                reading = imu.read()
                tilt_rad = reading.tilt_rad

                # The "measured state" for the replan is wherever we are
                # in the simulated execution. (No motors → assume perfect
                # tracking; the only real-world feedback is IMU tilt.)
                measured_state = {
                    "t1_grip_deg": sim_pose_deg[0], "t2_grip_deg": sim_pose_deg[1],
                    "t1_free_deg": sim_pose_deg[2], "t2_free_deg": sim_pose_deg[3],
                    "tail_gui_deg": sim_pose_deg[4],
                }

                # Tell the controller about the current IMU tilt so the
                # COM/goal constraints rotate into world frame around bar1.
                ctrl.current_imu_tilt_rad = tilt_rad
                # Always plan the full horizon — never shrink. WP0 of the
                # new plan is sim_pose, WP1 is the next "step", WP_N-1 is
                # the bar.
                ctrl.current_wp = 0
                ctrl._replan_from_current(tilt_rad,
                                           measured_state=measured_state)
                ctrl.replan_count += 1

                planned_tilt = ctrl.planned_tilts[0]
                err = tilt_rad - planned_tilt

                # Advance: take WP `advance_wp` of the freshly planned
                # trajectory as our new state. Larger `advance_wp` → faster
                # progress per iter, less Zeno-style asymptotic crawl.
                jump = min(max(advance_wp, 1), len(ctrl.trajectory) - 1)
                sim_pose_deg = tuple(unpack_state(ctrl.trajectory[jump]))
                sim_pose_box[0] = sim_pose_deg
                # Slide the anchor reference forward by the same amount so
                # next iter's QP tracks the *future* of the cold-start arc.
                if ctrl.anchor is not None:
                    ctrl.anchor_idx = min(ctrl.anchor_idx + jump,
                                          len(ctrl.anchor) - 1)

                tip = _free_tip_at(sim_pose_deg)
                tip_miss = float(np.linalg.norm(tip - target_xy))

                last_status.update({
                    "iter": iters,
                    "tilt_err_deg": math.degrees(err),
                    "replanned": True,
                    "replan_count": ctrl.replan_count,
                    "qp_solve_ms": ctrl._last_qp_solve_ms,
                    "tip_miss_mm": tip_miss * 1000.0,
                })

                if tip_miss < arrive_tol_m:
                    print(f"\n[mpc-viz] reached bar 2 in {iters+1} replans "
                          f"(tip miss {tip_miss*1000:.1f} mm).")
                    ctrl.done = True
                    break

                iters += 1
                t_end = time.monotonic() + step_period_s
                while time.monotonic() < t_end and not stop_event.is_set():
                    time.sleep(0.02)
            else:
                if iters >= max_iters:
                    print(f"\n[mpc-viz] hit max_iters={max_iters} without arriving.")
        finally:
            print(f"[mpc-viz] {ctrl.replan_count} replans total.")

    step_thread = threading.Thread(target=step_loop, daemon=True,
                                    name="mpc_viz_step")
    step_thread.start()

    # ── Live matplotlib viz ──
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.canvas.manager.set_window_title("MPC replan viz (close to abort)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"MPC replan viz   {'qp-mpc' if qp_mpc else 'mpc'}   "
                 f"step={step_period_s:.1f}s")

    # Bars
    bar_half = 0.04
    for bx, label in [(bar1_x, "Bar 1"), (bar2_x, "Bar 2")]:
        ax.plot([bx, bx], [bar_y - bar_half, bar_y + bar_half],
                color="brown", linewidth=6, solid_capstyle="round")
        ax.text(bx, bar_y + bar_half + 0.01, label, ha="center", fontsize=9)
    ax.axvline(bar1_x, color="red", linestyle=":", alpha=0.4)

    reach = L1 + L2 + GRIPPER_DRAW_LENGTH + 0.1
    ax.set_xlim(min(bar1_x, bar2_x) - reach,
                max(bar1_x, bar2_x) + reach)
    ax.set_ylim(bar_y - reach, bar_y + 0.2)

    pool = SegmentPool(ax)            # current body
    ghost_lines = []                  # remaining-trajectory ghosts
    com_dot, = ax.plot([], [], "D", color="#0891b2", markersize=10)
    info_text = ax.text(0.02, 0.98, "starting…", transform=ax.transAxes,
                        fontsize=10, family="monospace", va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    def _ghost_line(alpha):
        line, = ax.plot([], [], color="#9ca3af", linewidth=0.9,
                        alpha=alpha, linestyle="-")
        return line

    def update(_frame):
        reading = imu.read()
        tilt_rad = reading.tilt_rad
        sim_pose_deg = sim_pose_box[0]

        t1g, t2g, t1f, t2f, td = sim_pose_deg
        info = _build_pose_info(t1g, t2g, t1f, t2f, td, grip_is_right, bar1)
        base_segs = get_robot_segments(info, grip_is_right, L1, L2,
                                        bar_pos=None)
        rotated = apply_tilt_to_segments(base_segs, -tilt_rad, grip_point)

        pool.draw(rotated[:-1])
        com_xs, com_ys = rotated[-1]
        com_dot.set_data([com_xs[0]], [com_ys[0]])

        # Ghost the planned future free-tip positions (WP 1 .. N-1).
        traj = ctrl.trajectory
        n_rem = len(traj) - 1
        while len(ghost_lines) < n_rem:
            ghost_lines.append(_ghost_line(alpha=0.4))
        # Apply IMU tilt to ghost tips so they share the body's frame.
        cos_a = math.cos(-tilt_rad); sin_a = math.sin(-tilt_rad)
        gx, gy = grip_point
        for i, gline in enumerate(ghost_lines):
            if i < n_rem:
                state_g = traj[i + 1]
                t1g_g, t2g_g, t1f_g, t2f_g, _ = unpack_state(state_g)
                tip_g = _free_tip_at((t1g_g, t2g_g, t1f_g, t2f_g, 0.0))
                dx = tip_g[0] - gx; dy = tip_g[1] - gy
                rx = gx + dx * cos_a - dy * sin_a
                ry = gy + dx * sin_a + dy * cos_a
                gline.set_data([rx], [ry])
                gline.set_marker("x")
                gline.set_visible(True)
            else:
                gline.set_visible(False)

        info_text.set_text(
            f"iter:     {last_status['iter']}\n"
            f"tilt:     {math.degrees(tilt_rad):+6.2f}° "
            f"(planned {math.degrees(ctrl.planned_tilts[0]):+6.2f}°)\n"
            f"err:      {last_status['tilt_err_deg']:+6.2f}°\n"
            f"replans:  {last_status['replan_count']}\n"
            f"qp solve: {last_status['qp_solve_ms']:.2f} ms\n"
            f"tip miss: {last_status['tip_miss_mm']:.1f} mm"
        )
        return pool._lines + ghost_lines + [com_dot, info_text]

    interval_ms = max(20, int(1000.0 / max(fps, 1)))
    anim = FuncAnimation(fig, update, interval=interval_ms, blit=False,
                         cache_frame_data=False)

    def on_close(event):
        stop_event.set()
    fig.canvas.mpl_connect('close_event', on_close)

    try:
        plt.show()
    finally:
        stop_event.set()
        step_thread.join(timeout=2.0)
        imu.disconnect()


# ── Trajectory execution on real motors ─────────────────────────────────────

def planner_state_to_motor_targets(state_radians, grip_is_right, motor_reader):
    """Map a planner state vector to per-motor absolute encoder targets.

    Returns {motor_id: target_pos_rad} for motors present in the snapshot.
    Motors not connected (e.g. tail motor 20) are silently skipped.
    """
    t1g, t2g, t1f, t2f, td = unpack_state(state_radians)
    grip_side = "right" if grip_is_right else "left"
    free_side = "left"  if grip_is_right else "right"
    by_joint = {
        (grip_side, "shoulder"): t1g,
        (grip_side, "elbow"):    t2g,
        (free_side, "shoulder"): t1f,
        (free_side, "elbow"):    t2f,
        ("tail", None):          td,
    }
    snap = motor_reader.snapshot()
    targets = {}
    for mid, (side, kind, sign) in MOTOR_TO_JOINT.items():
        s = snap.get(mid)
        if s is None or s.last_update == 0.0:
            continue
        target_state = by_joint.get((side, kind))
        if target_state is None:
            continue
        offset = JOINT_OFFSETS_DEG.get((side, kind), 0.0)
        # state = sign * joint_deg + offset  ⇒  joint_deg = (state - offset) / sign
        joint_deg = (target_state - offset) / sign
        targets[mid] = s.zero_rad + math.radians(joint_deg)
    return targets


def motor_range_limits_deg(motor_reader):
    """Per-(side,kind) (lo, hi) state-deg bounds derived from each motor's
    logged min_from_zero_deg / max_from_zero_deg."""
    snap = motor_reader.snapshot()
    bounds = {}
    for mid, (side, kind, sign) in MOTOR_TO_JOINT.items():
        s = snap.get(mid)
        if s is None:
            continue
        offset = JOINT_OFFSETS_DEG.get((side, kind), 0.0)
        a = sign * s.min_from_zero_deg + offset
        b = sign * s.max_from_zero_deg + offset
        bounds[(side, kind)] = (min(a, b), max(a, b))
    return bounds


def _intersect_lim(default, motor_lim):
    if motor_lim is None:
        return default
    return (max(default[0], motor_lim[0]),
            min(default[1], motor_lim[1]))


def run_execute(config_path, log_file, only_buses, grip_is_right,
                imu_kwargs, bar1_x, bar2_x, bar_y, n_waypoints,
                kp, kd, tor, step_period_s, planner_kwargs,
                replan_threshold_deg=5.0,
                joint_replan_threshold_deg=8.0,
                mpc_mode=False,
                qp_mpc=False,
                qp_slack_weight=1000.0,
                qp_track_weight=10.0,
                fps=20, confirm=True):
    """Plan + execute the brachiation trajectory on the real motors at low
    torque, with a live matplotlib display of the measured pose + IMU tilt.

    The MotorReader must be the sole owner of the gs_usb buses (stop the
    GUI before running). Motors are enabled at low Kp/Kd so the arm is
    soft-tracked; on Ctrl+C, exception, or window close, all motors are
    disabled.
    """
    from motor_reader import MotorReader

    reader = MotorReader(config_path=config_path, log_file=log_file,
                         only_buses=only_buses)
    imu = WitIMU(**imu_kwargs)

    reader.start()
    imu.connect()

    # Wait for first motor readings on the connected arm joints.
    print("Waiting for first motor readings...")
    t_deadline = time.monotonic() + 3.0
    while time.monotonic() < t_deadline:
        ms = measured_planner_state(reader.snapshot(), grip_is_right)
        joints = [v for k, v in ms.items() if k != "tail_gui_deg"]
        if all(v is not None for v in joints):
            break
        time.sleep(0.1)

    ms = measured_planner_state(reader.snapshot(), grip_is_right)
    missing = [k for k, v in ms.items() if v is None and k != "tail_gui_deg"]
    if missing:
        print(f"ERROR: motor data still missing for {missing}. Aborting.")
        reader.stop(); imu.disconnect()
        return

    start_state_deg = (
        ms["t1_grip_deg"], ms["t2_grip_deg"],
        ms["t1_free_deg"], ms["t2_free_deg"],
        0.0,   # tail not connected
    )

    # Per-arm limits from logged motor ranges (intersected with planner defaults).
    motor_bnds = motor_range_limits_deg(reader)
    rsh = _intersect_lim((-70.9, 46.0), motor_bnds.get(("right", "shoulder")))
    rel = _intersect_lim((-90.4, 140.0), motor_bnds.get(("right", "elbow")))
    lsh = _intersect_lim((-46.0, 70.9), motor_bnds.get(("left", "shoulder")))
    lel = _intersect_lim((-140.0, 90.4), motor_bnds.get(("left", "elbow")))

    print("\nLimits (state-deg):")
    print(f"  right shoulder: {rsh}")
    print(f"  right elbow:    {rel}")
    print(f"  left shoulder:  {lsh}")
    print(f"  left elbow:     {lel}")
    print(f"\nStarting state (current measured pose):")
    print(f"  t1_grip={start_state_deg[0]:+6.2f}°  "
          f"t2_grip={start_state_deg[1]:+6.2f}°  "
          f"t1_free={start_state_deg[2]:+6.2f}°  "
          f"t2_free={start_state_deg[3]:+6.2f}°")

    print("\nPlanning trajectory from current pose...")
    pkw = dict(planner_kwargs or {})
    X_opt, infos, result = plan_trajectory(
        bar1_x, bar2_x, bar_y,
        grip_is_right=grip_is_right,
        n_waypoints=n_waypoints,
        right_shoulder_lim=rsh, right_elbow_lim=rel,
        left_shoulder_lim=lsh,  left_elbow_lim=lel,
        start_state_deg=start_state_deg,
        **pkw,
    )

    total_s = n_waypoints * step_period_s
    print(f"\nTrajectory: {n_waypoints} waypoints, "
          f"step={step_period_s:.2f}s, total ~{total_s:.1f}s")
    print(f"Gains: Kp={kp} Kd={kd} TorqueFF={tor}")

    arm_motors = [mid for mid, (_, kind, _) in MOTOR_TO_JOINT.items()
                  if kind in ("shoulder", "elbow")
                  and mid in reader.snapshot()
                  and reader.snapshot()[mid].last_update > 0]
    print(f"Motors to drive: {arm_motors}")

    if confirm:
        resp = input("\nEnable motors and execute? [y/N]: ").strip().lower()
        if resp != "y":
            print("Aborted.")
            reader.stop(); imu.disconnect()
            return

    print(f"Enabling motors {arm_motors}...")
    for mid in arm_motors:
        reader.enable(mid)
        time.sleep(0.05)
    # Hold current pose at very low gain before stepping the trajectory, so the
    # motors don't lurch when the first WP is sent.
    initial_targets = planner_state_to_motor_targets(
        pack_state(*start_state_deg), grip_is_right, reader)
    for mid, target in initial_targets.items():
        if mid in arm_motors:
            try:
                reader.command(mid, target, kp=kp, kd=kd, tor=tor)
            except Exception:
                pass

    # Build a ReplanningController so each step re-checks IMU tilt + joint
    # error and replans (from the measured state) if either exceeds threshold.
    # In --mpc mode the controller replans every step regardless of error.
    ctrl = ReplanningController(
        bar1_x, bar2_x, bar_y, grip_is_right, imu,
        n_waypoints=n_waypoints,
        replan_threshold_deg=replan_threshold_deg,
        joint_replan_threshold_deg=joint_replan_threshold_deg,
        planner_kwargs=dict(
            **(planner_kwargs or {}),
            right_shoulder_lim=rsh, right_elbow_lim=rel,
            left_shoulder_lim=lsh,  left_elbow_lim=lel,
        ),
        motor_reader=reader,
        mpc_mode=mpc_mode,
        qp_mpc=qp_mpc,
        qp_slack_weight=qp_slack_weight,
        qp_track_weight=qp_track_weight,
    )
    ctrl.trajectory = X_opt.copy()
    ctrl.infos = list(infos)
    grip_point_arr = np.array([bar1_x, bar_y])
    ctrl.planned_tilts = [pendulum_tilt_angle(info["com"], grip_point_arr)
                          for info in infos]

    stop_event = threading.Event()
    current_wp = [0]

    def cmd_loop():
        try:
            while not stop_event.is_set() and not ctrl.done:
                current_wp[0] = ctrl.current_wp
                # 1) Read IMU + (inside ctrl.step) motor feedback, replan if
                #    tilt or joint error exceeds threshold.
                reading = imu.read()
                status = ctrl.step(reading)
                # 2) Send the (possibly replanned) target for the just-acted
                #    waypoint to all connected arm motors.
                wp_just_acted = status["wp"]
                state_rad = ctrl.trajectory[wp_just_acted]
                targets = planner_state_to_motor_targets(
                    state_rad, grip_is_right, reader)
                for mid, target in targets.items():
                    if mid not in arm_motors:
                        continue
                    try:
                        reader.command(mid, target, kp=kp, kd=kd, tor=tor)
                    except Exception as exc:
                        print(f"[cmd] motor {mid} send failed: {exc}")
                # Per-step diagnostic line.
                jerr = status.get("max_joint_err_deg")
                jerr_s = "  ----" if jerr is None else f"{jerr:+5.1f}"
                tag = "REPLAN" if status["replanned"] else ""
                print(f"  WP {wp_just_acted:>2d}: "
                      f"tilt err={status['error_deg']:+5.1f}°  "
                      f"max joint err={jerr_s}°  {tag}")
                # 3) Wait the step period (interruptible).
                t_end = time.monotonic() + step_period_s
                while time.monotonic() < t_end and not stop_event.is_set():
                    time.sleep(0.01)
            print("\n[cmd] trajectory complete." if ctrl.done
                  else "\n[cmd] aborted.")
            ctrl.summary()
        finally:
            print("[cmd] disabling motors...")
            for mid in arm_motors:
                reader.disable(mid)
            print("[cmd] motors disabled.")

    cmd_thread = threading.Thread(target=cmd_loop, daemon=True, name="cmd_loop")
    cmd_thread.start()

    # Live display window (main thread).
    bar1 = np.array([bar1_x, bar_y])
    L1, L2 = LINK1_LENGTH, LINK2_LENGTH

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.canvas.manager.set_window_title("Executing trajectory (close window to abort)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"EXECUTE  Kp={kp} Kd={kd} step={step_period_s:.2f}s  "
                 f"({n_waypoints} wp, ~{total_s:.0f}s total)")

    bar_half = 0.04
    for bx, label in [(bar1_x, "Bar 1"), (bar2_x, "Bar 2")]:
        ax.plot([bx, bx], [bar_y - bar_half, bar_y + bar_half],
                color="brown", linewidth=6, solid_capstyle="round")
        ax.text(bx, bar_y + bar_half + 0.01, label, ha="center", fontsize=9)

    reach = L1 + L2 + GRIPPER_DRAW_LENGTH + 0.1
    ax.set_xlim(min(bar1_x, bar2_x) - reach,
                max(bar1_x, bar2_x) + reach)
    ax.set_ylim(bar_y - reach, bar_y + 0.2)

    pool_meas = SegmentPool(ax)
    com_dot, = ax.plot([], [], "D", color="#0891b2", markersize=9)
    info_text = ax.text(0.02, 0.98, "starting…", transform=ax.transAxes,
                        fontsize=10, family="monospace", va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    def update(_frame):
        snap = reader.snapshot()
        ms = measured_planner_state(snap, grip_is_right)
        reading = imu.read()
        tilt_rad = reading.tilt_rad
        wp = min(current_wp[0], len(X_opt) - 1)
        planned = X_opt[wp]
        errs = joint_error_deg(ms, planned)

        if all(v is not None for k, v in ms.items() if k != "tail_gui_deg"):
            info_meas = _build_pose_info(
                ms["t1_grip_deg"], ms["t2_grip_deg"],
                ms["t1_free_deg"], ms["t2_free_deg"],
                0.0, grip_is_right, bar1)
            base_segs = get_robot_segments(info_meas, grip_is_right, L1, L2,
                                            bar_pos=None)
            rotated = apply_tilt_to_segments(base_segs, -tilt_rad,
                                             np.array([bar1_x, bar_y]))
            pool_meas.draw(rotated[:-1])
            com_xs, com_ys = rotated[-1]
            com_dot.set_data([com_xs[0]], [com_ys[0]])

        def _e(k):
            v = errs[k]
            return "  ----" if v is None else f"{v:+5.1f}"
        info_text.set_text(
            f"WP {wp:>2d}/{len(X_opt)-1}   "
            f"tilt={math.degrees(tilt_rad):+5.1f}°   "
            f"rate={math.degrees(reading.tilt_rate_rad_s):+5.1f}°/s\n"
            f"err°: t1g={_e('t1_grip_deg')} t2g={_e('t2_grip_deg')}  "
            f"t1f={_e('t1_free_deg')} t2f={_e('t2_free_deg')}"
        )
        return pool_meas._lines + [com_dot, info_text]

    interval_ms = max(20, int(1000.0 / max(fps, 1)))
    anim = FuncAnimation(fig, update, interval=interval_ms, blit=False,
                         cache_frame_data=False)

    def on_close(event):
        stop_event.set()
    fig.canvas.mpl_connect('close_event', on_close)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cmd_thread.join(timeout=2.0)
        # Belt-and-braces — ensure motors are disabled even if cmd_loop raced.
        for mid in arm_motors:
            try:
                reader.disable(mid)
            except Exception:
                pass
        imu.disconnect()
        reader.stop()


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
    parser.add_argument("--imu", action="store_true",
                        help="Use real WIT-Motion IMU over USB-serial (implies --closed-loop)")
    parser.add_argument("--imu-port", default=None,
                        help="IMU serial port (default: auto-detect /dev/ttyUSB*)")
    parser.add_argument("--imu-baud", type=int, default=9600)
    parser.add_argument("--imu-axis", choices=["roll", "pitch", "yaw"], default="roll")
    parser.add_argument("--imu-offset-deg", type=float, default=0.0)
    parser.add_argument("--imu-invert", action="store_true")
    parser.add_argument("--imu-step-period", type=float, default=0.1,
                        help="Wall-clock seconds per control step when --imu is set")
    parser.add_argument("--imu-test", action="store_true",
                        help="Stream live IMU tilt readings and exit")
    parser.add_argument("--imu-test-rate", type=float, default=10.0)
    parser.add_argument("--imu-animate", action="store_true",
                        help="Live animation of body tilt driven by IMU only")
    parser.add_argument("--imu-animate-fps", type=float, default=20.0)
    parser.add_argument("--motors", action="store_true",
                        help="Read Robstride motor feedback during closed-loop "
                             "(implies --closed-loop). Read-only; never commands.")
    parser.add_argument("--motors-config-path", default="/home/orin/robstridedebug")
    parser.add_argument("--motors-log", default="positions_log.json")
    parser.add_argument("--motors-bus", action="append", default=None,
                        help="Restrict to specific CAN bus name(s); repeat for multiple")
    parser.add_argument("--motors-test", action="store_true",
                        help="Stream live motor readings and exit")
    parser.add_argument("--motors-test-rate", type=float, default=5.0)
    parser.add_argument("--motors-animate", action="store_true",
                        help="Live 2D animation that mirrors motor positions")
    parser.add_argument("--motors-animate-fps", type=float, default=20.0)
    parser.add_argument("--motors-animate-tilt", action="store_true",
                        help="Apply pendulum tilt from COM (only meaningful when "
                             "the robot is hanging from a bar)")
    parser.add_argument("--live-animate", action="store_true",
                        help="Live animation driven by BOTH motors (pose) and "
                             "IMU (body tilt). Read-only on both.")
    parser.add_argument("--live-animate-fps", type=float, default=20.0)
    parser.add_argument("--execute", action="store_true",
                        help="Plan + execute the trajectory on the real motors "
                             "at low torque. Stop the GUI first; this script "
                             "owns the buses. Always disables motors on exit.")
    parser.add_argument("--kp", type=float, default=5.0,
                        help="MIT-mode Kp (default 5; gentle tracking)")
    parser.add_argument("--kd", type=float, default=0.5,
                        help="MIT-mode Kd (default 0.5)")
    parser.add_argument("--torque-ff", type=float, default=0.0,
                        help="MIT-mode torque feedforward (default 0)")
    parser.add_argument("--speed", type=float, default=0.1,
                        help="Speed multiplier for execution (default 0.1 = 10x slower)")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip the y/N confirmation prompt before enabling motors")
    parser.add_argument("--joint-replan-threshold", type=float, default=8.0,
                        help="Max joint error (deg) before triggering replan during --execute")
    parser.add_argument("--mpc", action="store_true",
                        help="MPC mode: replan from current measured state at every "
                             "step, regardless of error thresholds")
    parser.add_argument("--qp-mpc", action="store_true",
                        help="Use OSQP-based linearized-QP replan (sub-millisecond) "
                             "instead of full SLSQP. Requires an initial SLSQP "
                             "trajectory as the tracking reference.")
    parser.add_argument("--qp-slack-weight", type=float, default=1000.0,
                        help="Quadratic penalty on COM-band slack in QP MPC "
                             "(higher → stricter COM constraint; default 1000)")
    parser.add_argument("--qp-track-weight", type=float, default=10.0,
                        help="Tracking weight on x_ref in QP MPC (default 10)")
    parser.add_argument("--mpc-viz", action="store_true",
                        help="Live viz of MPC replanning driven by real IMU "
                             "(no motors). Tilt the IMU → watch the plan adapt.")
    parser.add_argument("--mpc-viz-step", type=float, default=0.1,
                        help="Wall-clock seconds per replan iteration (default 0.1)")
    parser.add_argument("--mpc-viz-fps", type=float, default=20.0,
                        help="Animation FPS for --mpc-viz (default 20)")
    parser.add_argument("--minimal", action="store_true",
                        help="Use plan_trajectory_minimal: no hand-crafted "
                             "arc, tail unlocked, COM hard-equality.")
    parser.add_argument("--tail-min", type=float, default=-60.0,
                        help="Tail GUI angle lower limit (deg, --minimal only)")
    parser.add_argument("--tail-max", type=float, default=60.0,
                        help="Tail GUI angle upper limit (deg, --minimal only)")
    parser.add_argument("--dip-depth", type=float, default=0.10,
                        help="Peak dip depth in m below bar (default 0.10)")
    parser.add_argument("--dip-skew", type=float, default=4.0,
                        help="Dip skew k: >1 pushes dip peak late "
                             "(default 4.0, peak at t≈0.84)")
    parser.add_argument("--com-band", type=float, default=0.03,
                        help="|COM_x - bar1_x| during direct phase (m)")
    parser.add_argument("--com-band-late", type=float, default=0.05,
                        help="|COM_x - bar1_x| during dip/catch phase (m)")
    parser.add_argument("--no-collision", action="store_true",
                        help="Disable geometric collision constraints "
                             "(--minimal only)")
    parser.add_argument("--com-relax-t", type=float, default=0.70,
                        help="t at which COM band starts widening (0..1)")
    parser.add_argument("--bar-clearance", type=float, default=0.02,
                        help="Min clearance below bar in m (except at catch)")
    parser.add_argument("--track-weight", type=float, default=20.0,
                        help="Soft tracking cost on Cartesian U-target")
    parser.add_argument("--mpc-viz-advance", type=int, default=4,
                        help="How many waypoints of the fresh plan to advance "
                             "per iter (default 4). Higher = faster progress "
                             "per step, less asymptotic crawl near the goal. "
                             "Set close to --n-waypoints for nearly-direct execution.")
    args = parser.parse_args()

    if args.imu or args.motors:
        args.closed_loop = True

    if args.motors_test:
        from pathlib import Path
        run_motors_test(
            config_path=Path(args.motors_config_path),
            log_file=args.motors_log,
            only_buses=args.motors_bus,
            rate_hz=args.motors_test_rate,
            grip_is_right=(args.grip == "right"),
        )
        return

    if args.motors_animate:
        from pathlib import Path
        run_motors_animate(
            config_path=Path(args.motors_config_path),
            log_file=args.motors_log,
            only_buses=args.motors_bus,
            grip_is_right=(args.grip == "right"),
            fps=args.motors_animate_fps,
            apply_pendulum_tilt=args.motors_animate_tilt,
        )
        return

    if args.live_animate:
        from pathlib import Path
        run_live_animate(
            config_path=Path(args.motors_config_path),
            log_file=args.motors_log,
            only_buses=args.motors_bus,
            grip_is_right=(args.grip == "right"),
            imu_kwargs=dict(
                port=args.imu_port,
                baud=args.imu_baud,
                axis=args.imu_axis,
                offset_deg=args.imu_offset_deg,
                invert=args.imu_invert,
            ),
            fps=args.live_animate_fps,
        )
        return

    if args.imu_test:
        run_imu_test(
            port=args.imu_port,
            baud=args.imu_baud,
            axis=args.imu_axis,
            offset_deg=args.imu_offset_deg,
            invert=args.imu_invert,
            rate_hz=args.imu_test_rate,
        )
        return

    if args.imu_animate:
        run_imu_animate(
            port=args.imu_port,
            baud=args.imu_baud,
            axis=args.imu_axis,
            offset_deg=args.imu_offset_deg,
            invert=args.imu_invert,
            grip_is_right=(args.grip == "right"),
            fps=args.imu_animate_fps,
        )
        return

    if args.mpc_viz:
        spacing_m = args.bar_spacing * INCH_TO_M
        bar_y = args.bar_y if args.bar_y is not None else BAR_Y
        bar1_x = 0.0
        bar2_x = bar1_x + spacing_m
        run_mpc_viz(
            grip_is_right=(args.grip == "right"),
            imu_kwargs=dict(
                port=args.imu_port, baud=args.imu_baud,
                axis=args.imu_axis, offset_deg=args.imu_offset_deg,
                invert=args.imu_invert,
            ),
            bar1_x=bar1_x, bar2_x=bar2_x, bar_y=bar_y,
            n_waypoints=args.n_waypoints,
            planner_kwargs=dict(
                com_tolerance=args.com_tol,
                smoothness_weight=args.smoothness,
                reach_weight=args.reach_weight,
            ),
            qp_mpc=args.qp_mpc,
            qp_slack_weight=args.qp_slack_weight,
            qp_track_weight=args.qp_track_weight,
            replan_threshold_deg=args.replan_threshold,
            step_period_s=args.mpc_viz_step,
            fps=args.mpc_viz_fps,
            advance_wp=args.mpc_viz_advance,
        )
        return

    if args.execute:
        from pathlib import Path
        spacing_m = args.bar_spacing * INCH_TO_M
        bar_y = args.bar_y if args.bar_y is not None else BAR_Y
        bar1_x = 0.0
        bar2_x = bar1_x + spacing_m
        # 0.1× speed → multiply step period by 1/speed.
        step_period_s = args.imu_step_period / max(args.speed, 1e-6)
        run_execute(
            config_path=Path(args.motors_config_path),
            log_file=args.motors_log,
            only_buses=args.motors_bus,
            grip_is_right=(args.grip == "right"),
            imu_kwargs=dict(
                port=args.imu_port, baud=args.imu_baud,
                axis=args.imu_axis, offset_deg=args.imu_offset_deg,
                invert=args.imu_invert,
            ),
            bar1_x=bar1_x, bar2_x=bar2_x, bar_y=bar_y,
            n_waypoints=args.n_waypoints,
            kp=args.kp, kd=args.kd, tor=args.torque_ff,
            step_period_s=step_period_s,
            planner_kwargs=dict(
                com_tolerance=args.com_tol,
                smoothness_weight=args.smoothness,
                reach_weight=args.reach_weight,
            ),
            replan_threshold_deg=args.replan_threshold,
            joint_replan_threshold_deg=args.joint_replan_threshold,
            mpc_mode=args.mpc or args.qp_mpc,
            qp_mpc=args.qp_mpc,
            qp_slack_weight=args.qp_slack_weight,
            qp_track_weight=args.qp_track_weight,
            fps=args.live_animate_fps,
            confirm=not args.no_confirm,
        )
        return

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

    if args.minimal:
        X_opt, infos, result = plan_trajectory_minimal(
            bar1_x, bar2_x, bar_y,
            grip_is_right=grip_is_right,
            n_waypoints=args.n_waypoints,
            tail_lim=(args.tail_min, args.tail_max),
            smoothness_weight=args.smoothness,
            dip_depth_m=args.dip_depth,
            dip_skew_k=args.dip_skew,
            com_band_m=args.com_band,
            com_band_late_m=args.com_band_late,
            com_relax_t=args.com_relax_t,
            bar_clearance_m=args.bar_clearance,
            track_weight=args.track_weight,
            enable_collision=not args.no_collision,
        )
    else:
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

        if args.imu:
            imu = WitIMU(
                port=args.imu_port, baud=args.imu_baud,
                axis=args.imu_axis, offset_deg=args.imu_offset_deg,
                invert=args.imu_invert,
            )
            print(f"\nUsing real IMU on {imu._port} (axis={args.imu_axis}, "
                  f"offset={args.imu_offset_deg:+.2f}°, invert={args.imu_invert})")
            step_period = args.imu_step_period
        else:
            imu = None
            step_period = 0.0

        motor_reader = None
        if args.motors:
            from pathlib import Path
            from motor_reader import MotorReader
            motor_reader = MotorReader(
                config_path=Path(args.motors_config_path),
                log_file=args.motors_log,
                only_buses=args.motors_bus,
            )
            print(f"\nStarting motor feedback (motors: "
                  f"{motor_reader.motor_ids}) — read-only")
            motor_reader.start()
            time.sleep(0.3)

        try:
            ctrl, step_log = run_closed_loop_demo(
                X_opt, infos, bar1_x, bar2_x, bar_y, grip_is_right,
                noise_deg=args.imu_noise,
                replan_threshold_deg=args.replan_threshold,
                planner_kwargs=planner_kwargs,
                imu=imu,
                step_period_s=step_period,
                motor_reader=motor_reader,
            )
        finally:
            if motor_reader is not None:
                motor_reader.stop()

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
