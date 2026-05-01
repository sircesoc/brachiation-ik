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
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# Import everything we need from the existing combined module
from InvserseKinematic_COM_Combined import (
    BODY_PIVOT_OFFSET,
    GRIPPER_DRAW_LENGTH,
    INCH_TO_M,
    LINK1_LENGTH,
    LINK2_LENGTH,
    TAIL_ANGLE_STOP_DEG,
    TAIL_GUI_OFFSET_DEG,
    TAIL_PIVOT_Y,
    TAIL_LENGTH,
    COM_Prediction,
    forward_kinematics_2link,
    ik_2link,
    to_com_right_arm_angles_deg,
)

# ── Bar geometry ─────────────────────────────────────────────────────────────
BAR_SPACING = 12.0 * INCH_TO_M          # 12 in -> metres
BAR_Y       = (LINK1_LENGTH + LINK2_LENGTH) + GRIPPER_DRAW_LENGTH + 0.02
# Bars positioned so that bar 1 is at x=0 (body-centered) and bar 2 is to
# the right (+x in plot frame).

# ── Helpers ──────────────────────────────────────────────────────────────────

def fk_hand_local(phi, theta, L1, L2):
    """Hand position in the shoulder-local frame."""
    x_coords, y_coords = forward_kinematics_2link(phi, theta, L1, L2)
    return np.array([x_coords[-1], y_coords[-1]])


def shoulder_from_bar(bar_pos, phi_grip, theta_grip, L1, L2):
    """Given a bar position and gripping-arm IK angles, return the shoulder
    position in the world frame.

    The gripper hangs straight down from the bar (parallelogram), so:
        hand_world = bar_pos - (0, GRIPPER_LEN)
        shoulder   = hand_world - hand_local
    """
    hand_local = fk_hand_local(phi_grip, theta_grip, L1, L2)
    hand_world = bar_pos - np.array([0.0, GRIPPER_DRAW_LENGTH])
    shoulder_world = hand_world - hand_local
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


def free_hand_world(free_shoulder, phi_free, theta_free, L1, L2, free_is_right):
    """Free hand position in world frame.  For the right arm, +x local is
    +x world.  For the left arm, +x local maps to -x world (mirror)."""
    hand_local = fk_hand_local(phi_free, theta_free, L1, L2)
    if free_is_right:
        return free_shoulder + hand_local
    else:
        return free_shoulder + np.array([-hand_local[0], hand_local[1]])


def compute_com_world(phi_grip, theta_grip, phi_free, theta_free,
                      theta_tail_gui, grip_is_right, bar_pos, L1, L2):
    """Return (com_world_x, com_world_y) using COM_Prediction.

    Internally converts everything into the body frame that COM_Prediction
    expects, runs it, then maps the result back to world coordinates.
    """
    grip_shoulder = shoulder_from_bar(bar_pos, phi_grip, theta_grip, L1, L2)
    body_center, free_shoulder = body_and_free_shoulder(grip_shoulder, grip_is_right)

    # Convert IK angles to COM_Prediction naming
    # Right arm: theta_1 = degrees(theta), theta_2 = degrees(phi) - 180
    if grip_is_right:
        r_theta_1, r_theta_2, _ = to_com_right_arm_angles_deg(phi_grip, theta_grip)
        l_theta_1, l_theta_2, _ = to_com_right_arm_angles_deg(phi_free, theta_free)
    else:
        r_theta_1, r_theta_2, _ = to_com_right_arm_angles_deg(phi_free, theta_free)
        l_theta_1, l_theta_2, _ = to_com_right_arm_angles_deg(phi_grip, theta_grip)

    theta_tail_internal = theta_tail_gui + TAIL_GUI_OFFSET_DEG

    x_com_in, y_com_in = COM_Prediction(
        r_theta_1, r_theta_2,
        l_theta_1, l_theta_2,
        theta_tail_internal,
    )

    # COM_Prediction returns body-frame inches.  Convert to metres.
    # Our world frame has body_center at body_center, with x mirrored
    # (plot +x = body-frame -x).
    com_world_x = body_center[0] - x_com_in * INCH_TO_M
    com_world_y = body_center[1] + y_com_in * INCH_TO_M
    return com_world_x, com_world_y


# ── Angle ↔ flat-vector helpers ──────────────────────────────────────────────
# State per waypoint: [shoulder_grip, elbow_grip, shoulder_free, elbow_free, tail]
# All in RADIANS (IK convention: shoulder = theta, elbow_inner = phi).
# We store (theta_grip, phi_grip, theta_free, phi_free, tail_gui_rad).
STATE_DIM = 5

def pack_state(theta_grip, phi_grip, theta_free, phi_free, tail_gui_deg):
    return np.array([theta_grip, phi_grip, theta_free, phi_free,
                     math.radians(tail_gui_deg)])

def unpack_state(x):
    theta_grip = x[0]
    phi_grip   = x[1]
    theta_free = x[2]
    phi_free   = x[3]
    tail_gui_deg = math.degrees(x[4])
    return theta_grip, phi_grip, theta_free, phi_free, tail_gui_deg


# ── Joint limits (motor-space, same convention as the GUI) ───────────────────
# shoulder_motor = dir * (theta_deg - zero_deg)
# We use zero=90, dir=1 → motor = theta_deg - 90.
SHOULDER_ZERO = 90.0
ELBOW_ZERO    = 0.0

def theta_to_motor(theta_rad):
    """Shoulder geometry angle (rad) → motor angle (deg)."""
    return math.degrees(theta_rad) - SHOULDER_ZERO

def phi_to_motor(phi_rad):
    """Elbow inner angle (rad) → motor angle (deg).  motor = 180 - phi."""
    return 180.0 - math.degrees(phi_rad) - ELBOW_ZERO


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
    tail_lim=(-60.0, 60.0),
    com_tolerance=0.01,       # metres — how far COM_x may drift from bar
    smoothness_weight=0.5,
    reach_weight=10.0,
):
    """Plan a brachiation trajectory from bar 1 to bar 2.

    Returns an (n_waypoints, STATE_DIM) array of joint-angle waypoints plus
    auxiliary info (COM positions, hand positions, etc.) for visualisation.
    """
    bar1 = np.array([bar1_x, bar_y])
    bar2 = np.array([bar2_x, bar_y])

    free_is_right = not grip_is_right

    # Determine per-arm motor limits based on which arm is gripping
    if grip_is_right:
        grip_sh_lim  = right_shoulder_lim
        grip_el_lim  = right_elbow_lim
        free_sh_lim  = left_shoulder_lim
        free_el_lim  = left_elbow_lim
    else:
        grip_sh_lim  = left_shoulder_lim
        grip_el_lim  = left_elbow_lim
        free_sh_lim  = right_shoulder_lim
        free_el_lim  = right_elbow_lim

    def motor_to_theta_bounds(sh_lim):
        """Motor limits → geometry theta bounds (rad)."""
        lo = math.radians(sh_lim[0] + SHOULDER_ZERO)
        hi = math.radians(sh_lim[1] + SHOULDER_ZERO)
        return (lo, hi)

    def motor_to_phi_bounds(el_lim):
        """Motor limits → geometry phi bounds (rad).  phi = 180 - motor."""
        lo = math.radians(180.0 - el_lim[1])   # max motor → min phi
        hi = math.radians(180.0 - el_lim[0])   # min motor → max phi
        return (lo, hi)

    grip_theta_bnd = motor_to_theta_bounds(grip_sh_lim)
    grip_phi_bnd   = motor_to_phi_bounds(grip_el_lim)
    free_theta_bnd = motor_to_theta_bounds(free_sh_lim)
    free_phi_bnd   = motor_to_phi_bounds(free_el_lim)
    tail_bnd       = (math.radians(tail_lim[0]), math.radians(tail_lim[1]))

    per_wp_bounds = [grip_theta_bnd, grip_phi_bnd,
                     free_theta_bnd, free_phi_bnd,
                     tail_bnd]

    # Initial guess: everything straight up (theta=90°, phi=180°=straight)
    # and tail at 0 (hanging).
    x0_single = pack_state(
        theta_grip=math.radians(90),
        phi_grip=math.radians(180),
        theta_free=math.radians(90),
        phi_free=math.radians(180),
        tail_gui_deg=0.0,
    )
    x0 = np.tile(x0_single, n_waypoints)

    # Build flat bounds
    bounds = per_wp_bounds * n_waypoints

    def cost_and_info(x_flat, return_info=False):
        X = x_flat.reshape(n_waypoints, STATE_DIM)
        total_cost = 0.0
        infos = []

        for k in range(n_waypoints):
            theta_g, phi_g, theta_f, phi_f, tail_deg = unpack_state(X[k])

            # Gripping arm shoulder
            grip_sh = shoulder_from_bar(bar1, phi_g, theta_g, L1, L2)
            _, free_sh = body_and_free_shoulder(grip_sh, grip_is_right)

            # Free hand position — the gripper tip is what touches the bar,
            # not the hand itself.  Gripper extends vertically above the hand.
            fh = free_hand_world(free_sh, phi_f, theta_f, L1, L2, free_is_right)
            gripper_tip = fh + np.array([0.0, GRIPPER_DRAW_LENGTH])

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
            dip_depth = 3.0 * INCH_TO_M  # 3 inches below bar
            # Shift the dip center toward the end (t=0.65) so the
            # approach to bar 2 comes from below.
            dip_center = 0.65
            # Use a Gaussian-like bump centered at dip_center
            sigma = 0.25  # controls width / smoothness
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
                    phi_g, theta_g, phi_f, theta_f, tail_deg,
                    grip_is_right, bar1, L1, L2,
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
                tg, pg, tf, pf, td = unpack_state(X[kk])
                cx, _ = compute_com_world(pg, tg, pf, tf, td,
                                          grip_is_right, bar1, L1, L2)
                return cx - bar1_x + 3.0 * com_tolerance  # generous backward
            return con

        def make_com_hi(kk):
            def con(x_flat):
                X = x_flat.reshape(n_waypoints, STATE_DIM)
                tg, pg, tf, pf, td = unpack_state(X[kk])
                cx, _ = compute_com_world(pg, tg, pf, tf, td,
                                          grip_is_right, bar1, L1, L2)
                return bar1_x + com_tolerance - cx  # tight forward
            return con

        constraints.append({"type": "ineq", "fun": make_com_lo(k)})
        constraints.append({"type": "ineq", "fun": make_com_hi(k)})

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
                           arm_color_grip="#0f766e", arm_color_free="#16a34a",
                           bar_pos=None):
    """Draw the full robot for one waypoint with pendulum tilt applied."""
    segments = get_robot_segments(info, grip_is_right, L1, L2, bar_pos=bar_pos)

    # Segment order: body, grip_arm, grip_gripper, free_arm, free_gripper,
    #                body_box, tail, com_point
    seg_colors = ["#6b7280", arm_color_grip, "#7c3aed",
                  arm_color_free, "#7c3aed",
                  "#4b5563", "#d97706", "#0891b2"]
    seg_widths = [4, 3, 2, 3, 2, 2, 3, 0]
    seg_styles = ["-", "-", "--", "-", "--", "-", "-", ""]

    for i, (xs, ys) in enumerate(segments):
        if i == len(segments) - 1:
            # COM marker (last segment is a single point)
            ax.plot(xs[0], ys[0], "D", color="#0891b2", markersize=6, alpha=alpha)
        else:
            ax.plot(xs, ys, color=seg_colors[i], linewidth=seg_widths[i],
                    linestyle=seg_styles[i], alpha=alpha)


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

    # Bars
    bar_half = 0.04
    for bx, label in [(bar1_x, "Bar 1"), (bar2_x, "Bar 2")]:
        ax.plot([bx, bx], [bar_y - bar_half, bar_y + bar_half],
                color="brown", linewidth=6, solid_capstyle="round")
        ax.text(bx, bar_y + bar_half + 0.01, label, ha="center", fontsize=9)

    n = len(infos)
    for k, info in enumerate(infos):
        alpha = 0.15 + 0.85 * (k / max(n - 1, 1))
        draw_robot_at_waypoint(ax, info, grip_is_right, L1, L2, alpha=alpha,
                               bar_pos=(bar1_x, bar_y))

    # Draw reference hook trajectory (smooth Gaussian dip near bar 2)
    arc_t = np.linspace(0, 1, 80)
    dip_depth = 3.0 * INCH_TO_M
    dip_center = 0.65
    sigma = 0.25
    arc_xs = [bar1_x + (bar2_x - bar1_x) * (0.5 - 0.5 * math.cos(t * math.pi))
              for t in arc_t]
    arc_ys = [bar_y - dip_depth * math.exp(-0.5 * ((t - dip_center) / sigma) ** 2)
              for t in arc_t]
    ax.plot(arc_xs, arc_ys, "--", color="#a855f7", linewidth=1.5,
            alpha=0.6, label="Reference path")

    # Draw COM and gripper tip paths (tilted)
    grip_point = (bar1_x, bar_y)
    com_xs_t = []
    com_ys_t = []
    gt_xs_t = []
    gt_ys_t = []
    for info in infos:
        segs = get_robot_segments(info, grip_is_right, L1, L2, bar_pos=grip_point)
        # Last segment = COM point
        com_xs_t.append(segs[-1][0][0])
        com_ys_t.append(segs[-1][1][0])
        # Segment 4 = free gripper, last point = tip
        gt_xs_t.append(segs[4][0][-1])
        gt_ys_t.append(segs[4][1][-1])

    ax.plot(com_xs_t, com_ys_t, "o-", color="#0891b2", markersize=4,
            linewidth=1.5, label="COM path (tilted)")
    ax.plot(gt_xs_t, gt_ys_t, "x-", color="#f59e0b", markersize=6,
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

        tg, pg, tf, pf, td = unpack_state(wp)
        return {
            "grip_shoulder_motor_deg": theta_to_motor(tg),
            "grip_elbow_motor_deg": phi_to_motor(pg),
            "free_shoulder_motor_deg": theta_to_motor(tf),
            "free_elbow_motor_deg": phi_to_motor(pf),
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
    theta_g, phi_g, theta_f, phi_f, tail_deg = unpack_state(state)

    if grip_is_right:
        r_sh, l_sh = gs, fs
    else:
        r_sh, l_sh = fs, gs

    segments = []

    # Body (line between shoulders)
    segments.append(([r_sh[0], l_sh[0]], [r_sh[1], l_sh[1]]))

    # Gripping arm
    gx, gy = forward_kinematics_2link(phi_g, theta_g, L1, L2)
    gx_w = gx + gs[0]
    gy_w = gy + gs[1]
    segments.append((gx_w.tolist(), gy_w.tolist()))
    # Grip gripper (vertical)
    segments.append(([gx_w[-1], gx_w[-1]],
                     [gy_w[-1], gy_w[-1] + GRIPPER_DRAW_LENGTH]))

    # Free arm
    fx_local, fy_local = forward_kinematics_2link(phi_f, theta_f, L1, L2)
    free_is_right = not grip_is_right
    if free_is_right:
        fx_w = fx_local + fs[0]
    else:
        fx_w = -fx_local + fs[0]
    fy_w = fy_local + fs[1]
    segments.append((list(fx_w), list(fy_w)))
    # Free gripper (vertical)
    segments.append(([fx_w[-1], fx_w[-1]],
                     [fy_w[-1], fy_w[-1] + GRIPPER_DRAW_LENGTH]))

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

    # Prepare line objects for animation (all in the z=0 plane)
    # Segments: body, grip arm, grip gripper, free arm, free gripper, body box, tail
    colors = ["#6b7280", "#0f766e", "#7c3aed", "#16a34a", "#7c3aed", "#4b5563", "#d97706"]
    widths = [4, 3, 2, 3, 2, 2, 3]
    styles = ["-", "-", "--", "-", "--", "-", "-"]
    n_segs = 7

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
        grip_point = (bar1_x, bar_y)
        segments = get_robot_segments(info, grip_is_right, L1, L2,
                                     bar_pos=grip_point)
        # Last segment is the tilted COM point
        tilted_com_x = segments[-1][0][0]
        tilted_com_y = segments[-1][1][0]
        # Free gripper tip (segment 4 = free gripper, last point)
        fh_x = segments[4][0][-1]
        fh_y = segments[4][1][-1]
        fh = np.array([fh_x, fh_y])

        n_draw = len(segments) - 1  # skip COM point segment
        for i in range(min(n_draw, len(lines_3d))):
            xs, ys = segments[i]
            zs = [0.0] * len(xs)
            lines_3d[i].set_data_3d(xs, ys, zs)

        com_dot.set_data_3d([tilted_com_x], [tilted_com_y], [0.0])

        com_history_x.append(com[0])
        com_history_y.append(com[1])
        com_trail.set_data_3d(com_history_x, com_history_y,
                              [0.0] * len(com_history_x))

        fh_history_x.append(fh[0])
        fh_history_y.append(fh[1])
        fh_trail.set_data_3d(fh_history_x, fh_history_y,
                             [0.0] * len(fh_history_x))

        tg, pg, tf, pf, td = unpack_state(info["state"])
        info_text.set_text(
            f"WP {frame}/{len(infos)-1}\n"
            f"gripper tip: ({fh[0]:+.3f}, {fh[1]:+.3f}) m\n"
            f"COM_x offset: {(com[0]-bar1_x)*1000:+.1f} mm\n"
            f"grip_sh: {theta_to_motor(tg):+.1f}°  "
            f"grip_el: {phi_to_motor(pg):+.1f}°\n"
            f"free_sh: {theta_to_motor(tf):+.1f}°  "
            f"free_el: {phi_to_motor(pf):+.1f}°\n"
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
        tg, pg, tf, pf, td = unpack_state(info["state"])
        gt = info["free_gripper_tip"]
        com = info["com"]
        print(f"  WP {k:2d}:  gripper_tip=({gt[0]:+.3f}, {gt[1]:+.3f}) m  "
              f"COM_x={com[0]:+.4f} m  "
              f"grip_sh={theta_to_motor(tg):+6.1f}°  "
              f"grip_el={phi_to_motor(pg):+6.1f}°  "
              f"free_sh={theta_to_motor(tf):+6.1f}°  "
              f"free_el={phi_to_motor(pf):+6.1f}°  "
              f"tail={td:+5.1f}°")

    final_gt = infos[-1]["free_gripper_tip"]
    bar2 = np.array([bar2_x, bar_y])
    miss = np.linalg.norm(final_gt - bar2)
    print(f"\nFinal gripper-tip miss from bar 2: {miss*1000:.1f} mm")

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
