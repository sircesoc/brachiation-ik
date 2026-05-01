"""
Interactive 2-link arm IK with live target sliders and COM prediction.

The right arm uses the same conventions as visualize_2link_arm.py:
- +X points outward from the shoulder
- -Y points downward
- shoulder motor is measured in the base frame
- elbow motor is measured relative to link 2
- a straight arm is 0 deg / 0 deg in motor space
"""

import argparse
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


INCH_TO_M = 0.0254

# Robot geometry from COM Prediction.py (converted to meters)
BODY_PIVOT_OFFSET = 5.99606299 * INCH_TO_M   # shoulder distance from body center
LINK1_LENGTH = 3.4 * INCH_TO_M               # upper arm
LINK2_LENGTH = 11.8 * INCH_TO_M              # forearm
TAIL_PIVOT_Y = -10.0 * INCH_TO_M             # tail pivot y below body center
TAIL_LENGTH = 15.3379 * INCH_TO_M            # tail mass offset
TAIL_ANGLE_STOP_DEG = 8.73                   # tail mechanical stop angle
# GUI convention: theta_tail_gui = 0 means the tail is hanging straight
# down (-y) from its pivot, which is the natural resting pose. This offset
# converts GUI degrees into the value that COM_Prediction expects so its
# internal math stays untouched.
# Desired internal angle when gui=0: +(90 - TAIL_ANGLE_STOP_DEG) = 81.27,
# so that (theta + TAIL_ANGLE_STOP_DEG) = 90 -> -y in (cos, -sin) frame.
TAIL_GUI_OFFSET_DEG = 90.0 - TAIL_ANGLE_STOP_DEG
GRIPPER_X_OFFSET = 0.37758314 * INCH_TO_M
GRIPPER_Y_OFFSET = 3.87674 * INCH_TO_M
# The gripper is held vertical by a parallelogram linkage, so the drawn
# line is always straight up from the hand. Length = physical offset hypot.
GRIPPER_DRAW_LENGTH = math.hypot(GRIPPER_X_OFFSET, GRIPPER_Y_OFFSET)
BODY_COM_Y = -2.22013148 * INCH_TO_M


def to_com_right_arm_angles_deg(phi, theta):
    """
    Convert IK geometry angles into COM Prediction naming.

    theta_1 is the right shoulder absolute angle.
    theta_2 is the right elbow relative angle so that:
    theta_2_absolute = theta_1 + theta_2.
    """
    theta_1_deg = math.degrees(theta)
    theta_2_deg = math.degrees(phi) - 180.0
    theta_2_absolute_deg = theta_1_deg + theta_2_deg
    return theta_1_deg, theta_2_deg, theta_2_absolute_deg


def COM_Prediction(theta_1_degrees, theta_2_degrees, theta_11_degrees, theta_12_degrees, theta_tail_degrees):
    theta_2_absolute_degrees = theta_1_degrees + theta_2_degrees
    theta_1 = math.radians(theta_1_degrees)
    theta_2_absolute = math.radians(theta_2_absolute_degrees)

    theta_12_absolute_degrees = theta_11_degrees + theta_12_degrees
    theta_11 = math.radians(theta_11_degrees)
    theta_12_absolute = math.radians(theta_12_absolute_degrees)

    theta_tail = math.radians(theta_tail_degrees)

    link1_length = 3.4  # inches
    link2_length = 11.8  # inches
    body_pivot1_offset = 5.99606299  # inches

    # Body
    m_body = 9.68960433  # pounds
    x_com_body = 0.0  # inches
    y_com_body = -2.22013148  # inches

    # Tail
    m_tail = 3.9471154  # pounds
    tail_mass_offset = 15.3379  # inches
    tail_angle_stop_degrees = 8.73
    tail_angle_stop = math.radians(tail_angle_stop_degrees)
    x_com_tail = tail_mass_offset * math.cos(theta_tail + tail_angle_stop)
    y_com_tail = -10.0 - tail_mass_offset * math.sin(theta_tail + tail_angle_stop)

    # Right arm - link 1
    m_link1 = 2.78627841  # pounds
    x_com_link1 = -body_pivot1_offset - link1_length * math.cos(theta_1)
    y_com_link1 = link1_length * math.sin(theta_1)

    # Right arm - link 2
    m_link2 = 1.16620275  # pounds
    x_offset_link2_cubed_coefficient = 0.0
    x_offset_link2_squared_coefficient = -0.0001699867194
    x_offset_link2_linear_coefficient = 0.03098551081
    x_offset_link2_constant_coefficient = -0.1653118738

    y_offset_link2_cubed_coefficient = 0.0000009035335573
    y_offset_link2_squared_coefficient = -0.0002522954645
    y_offset_link2_linear_coefficient = 0.00184756179
    y_offset_link2_constant_coefficient = 2.727258043

    x_offset_link2 = (
        x_offset_link2_cubed_coefficient * (theta_2_absolute_degrees**3)
        + x_offset_link2_squared_coefficient * (theta_2_absolute_degrees**2)
        + x_offset_link2_linear_coefficient * theta_2_absolute_degrees
        + x_offset_link2_constant_coefficient
    )
    y_offset_link2 = (
        y_offset_link2_cubed_coefficient * (theta_2_absolute_degrees**3)
        + y_offset_link2_squared_coefficient * (theta_2_absolute_degrees**2)
        + y_offset_link2_linear_coefficient * theta_2_absolute_degrees
        + y_offset_link2_constant_coefficient
    )

    x_com_link2 = (
        -body_pivot1_offset
        - link1_length * math.cos(theta_1)
        - (link2_length / 2 + y_offset_link2) * math.cos(theta_2_absolute)
        - x_offset_link2 * math.cos(math.pi / 2 - theta_2_absolute)
    )
    y_com_link2 = (
        link1_length * math.sin(theta_1)
        + (link2_length / 2 + y_offset_link2) * math.sin(theta_2_absolute)
        - x_offset_link2 * math.sin(math.pi / 2 - theta_2_absolute)
    )

    # Right arm - gripper
    m_gripper = 4.08915  # pounds
    x_gripper_offset = 0.37758314  # inches
    y_gripper_offset = 3.87674  # inches
    x_com_gripper = (
        -body_pivot1_offset
        - link1_length * math.cos(theta_1)
        - link2_length * math.cos(theta_2_absolute)
        - x_gripper_offset
    )
    y_com_gripper = (
        link1_length * math.sin(theta_1)
        + link2_length * math.sin(theta_2_absolute)
        + y_gripper_offset
    )

    # Left arm - link 1
    m_link1_left = m_link1
    x_com_link1_left = body_pivot1_offset + link1_length * math.cos(theta_11)
    y_com_link1_left = link1_length * math.sin(theta_11)

    # Left arm - link 2
    m_link2_left = m_link2
    x_offset_link2_left = (
        x_offset_link2_cubed_coefficient * (theta_12_absolute_degrees**3)
        + x_offset_link2_squared_coefficient * (theta_12_absolute_degrees**2)
        + x_offset_link2_linear_coefficient * theta_12_absolute_degrees
        + x_offset_link2_constant_coefficient
    )
    y_offset_link2_left = (
        y_offset_link2_cubed_coefficient * (theta_12_absolute_degrees**3)
        + y_offset_link2_squared_coefficient * (theta_12_absolute_degrees**2)
        + y_offset_link2_linear_coefficient * theta_12_absolute_degrees
        + y_offset_link2_constant_coefficient
    )
    x_com_link2_left = (
        body_pivot1_offset
        + link1_length * math.cos(theta_11)
        + (link2_length / 2 + y_offset_link2_left) * math.cos(theta_12_absolute)
        + x_offset_link2_left * math.cos(math.pi / 2 - theta_12_absolute)
    )
    y_com_link2_left = (
        link1_length * math.sin(theta_11)
        + (link2_length / 2 + y_offset_link2_left) * math.sin(theta_12_absolute)
        - x_offset_link2_left * math.sin(math.pi / 2 - theta_12_absolute)
    )

    # Left arm - gripper
    m_gripper_left = m_gripper
    x_com_gripper_left = (
        body_pivot1_offset
        + link1_length * math.cos(theta_11)
        + link2_length * math.cos(theta_12_absolute)
        + x_gripper_offset
    )
    y_com_gripper_left = (
        link1_length * math.sin(theta_11)
        + link2_length * math.sin(theta_12_absolute)
        + y_gripper_offset
    )

    m_total = (
        m_body
        + m_tail
        + m_link1
        + m_link2
        + m_gripper
        + m_link1_left
        + m_link2_left
        + m_gripper_left
    )

    x_moment = (
        m_body * x_com_body
        + m_tail * x_com_tail
        + m_link1 * x_com_link1
        + m_link2 * x_com_link2
        + m_gripper * x_com_gripper
        + m_link1_left * x_com_link1_left
        + m_link2_left * x_com_link2_left
        + m_gripper_left * x_com_gripper_left
    )
    y_moment = (
        m_body * y_com_body
        + m_tail * y_com_tail
        + m_link1 * y_com_link1
        + m_link2 * y_com_link2
        + m_gripper * y_com_gripper
        + m_link1_left * y_com_link1_left
        + m_link2_left * y_com_link2_left
        + m_gripper_left * y_com_gripper_left
    )

    x_com_total = x_moment / m_total
    y_com_total = y_moment / m_total
    return x_com_total, y_com_total


def solve_left_arm_target(
    lx,
    ly,
    L1,
    L2,
    elbow_branch="down",
    shoulder_motor_min_deg=None,
    shoulder_motor_max_deg=None,
    elbow_motor_min_deg=None,
    elbow_motor_max_deg=None,
    shoulder_motor_zero_deg=0.0,
    elbow_motor_zero_deg=0.0,
    shoulder_motor_dir=1,
    elbow_motor_dir=1,
):
    """Solve left-arm 2-link IK in the left-shoulder local frame.

    Returns a dict with joint coordinates in the left-shoulder-local frame
    and the COM-style angles theta_11 / theta_12 (in degrees). Motor limits
    are applied using the same convention as the right arm.
    """
    phi, theta = ik_2link(lx, ly, L1, L2, elbow_branch)
    shoulder_motor_deg, elbow_motor_deg = to_motor_angles_deg(
        phi,
        theta,
        shoulder_motor_zero_deg=shoulder_motor_zero_deg,
        elbow_motor_zero_deg=elbow_motor_zero_deg,
        shoulder_motor_dir=shoulder_motor_dir,
        elbow_motor_dir=elbow_motor_dir,
    )
    check_motor_limits(
        shoulder_motor_deg,
        elbow_motor_deg,
        shoulder_motor_min_deg=shoulder_motor_min_deg,
        shoulder_motor_max_deg=shoulder_motor_max_deg,
        elbow_motor_min_deg=elbow_motor_min_deg,
        elbow_motor_max_deg=elbow_motor_max_deg,
    )
    # Reuse right-arm naming: theta_11 is the shoulder absolute angle,
    # theta_12 is the elbow relative angle.
    theta_11_deg = math.degrees(theta)
    theta_12_deg = math.degrees(phi) - 180.0
    x_coords, y_coords = forward_kinematics_2link(phi, theta, L1, L2)
    return {
        "phi": phi,
        "theta": theta,
        "theta_11_deg": theta_11_deg,
        "theta_12_deg": theta_12_deg,
        "theta_12_absolute_deg": theta_11_deg + theta_12_deg,
        "x_coords_local": np.array(x_coords),
        "y_coords_local": np.array(y_coords),
        "hand_local": np.array([x_coords[-1], y_coords[-1]]),
        "shoulder_motor_deg": shoulder_motor_deg,
        "elbow_motor_deg": elbow_motor_deg,
    }


def ik_2link(x_target, y_target, L2, L3, elbow_branch="down"):
    """Solve planar 2-link IK for a target in the arm plane."""
    reach = math.hypot(x_target, y_target)

    if reach > L2 + L3:
        raise ValueError(f"Target unreachable: distance {reach:.3f} > max reach {L2 + L3:.3f}")
    if reach < abs(L2 - L3):
        raise ValueError(f"Target too close: distance {reach:.3f} < min reach {abs(L2 - L3):.3f}")
    if reach < 1e-12:
        raise ValueError("Target is too close to the shoulder to define a stable IK solution")

    cos_phi = (L2 * L2 + L3 * L3 - reach * reach) / (2.0 * L2 * L3)
    cos_phi = max(-1.0, min(1.0, cos_phi))
    phi = math.acos(cos_phi)

    alpha = math.atan2(y_target, x_target)
    beta_arg = (L2 * L2 + reach * reach - L3 * L3) / (2.0 * L2 * reach)
    beta_arg = max(-1.0, min(1.0, beta_arg))
    beta = math.acos(beta_arg)

    if elbow_branch == "down":
        theta = alpha + beta
    else:
        theta = alpha - beta

    return phi, theta


def forward_kinematics_2link(phi, theta, L2, L3):
    """Return shoulder, elbow, and hand positions for the 2-link arm."""
    shoulder = np.array([0.0, 0.0])
    elbow = shoulder + L2 * np.array([math.cos(theta), math.sin(theta)])
    hand_angle = theta + phi - math.pi
    hand = elbow + L3 * np.array([math.cos(hand_angle), math.sin(hand_angle)])

    x_coords = np.array([shoulder[0], elbow[0], hand[0]])
    y_coords = np.array([shoulder[1], elbow[1], hand[1]])
    return x_coords, y_coords


def to_motor_angles_deg(
    phi,
    theta,
    shoulder_motor_zero_deg=0.0,
    elbow_motor_zero_deg=0.0,
    shoulder_motor_dir=1,
    elbow_motor_dir=1,
):
    phi_deg = math.degrees(phi)
    theta_deg = math.degrees(theta)

    shoulder_motor_deg = shoulder_motor_dir * (theta_deg - shoulder_motor_zero_deg)
    elbow_relative_deg = 180.0 - phi_deg
    elbow_motor_deg = elbow_motor_dir * (elbow_relative_deg - elbow_motor_zero_deg)
    return shoulder_motor_deg, elbow_motor_deg


def from_motor_angles_deg(
    shoulder_motor_deg,
    elbow_motor_deg,
    shoulder_motor_zero_deg=0.0,
    elbow_motor_zero_deg=0.0,
    shoulder_motor_dir=1,
    elbow_motor_dir=1,
):
    theta_deg = (shoulder_motor_deg / shoulder_motor_dir) + shoulder_motor_zero_deg
    elbow_relative_deg = (elbow_motor_deg / elbow_motor_dir) + elbow_motor_zero_deg
    phi_deg = 180.0 - elbow_relative_deg
    return math.radians(phi_deg), math.radians(theta_deg)


def check_motor_limits(
    shoulder_motor_deg,
    elbow_motor_deg,
    shoulder_motor_min_deg=None,
    shoulder_motor_max_deg=None,
    elbow_motor_min_deg=None,
    elbow_motor_max_deg=None,
):
    if shoulder_motor_min_deg is not None and shoulder_motor_deg < shoulder_motor_min_deg:
        raise ValueError(
            f"Shoulder motor angle {shoulder_motor_deg:.1f} deg is below limit {shoulder_motor_min_deg:.1f} deg"
        )
    if shoulder_motor_max_deg is not None and shoulder_motor_deg > shoulder_motor_max_deg:
        raise ValueError(
            f"Shoulder motor angle {shoulder_motor_deg:.1f} deg exceeds limit {shoulder_motor_max_deg:.1f} deg"
        )
    if elbow_motor_min_deg is not None and elbow_motor_deg < elbow_motor_min_deg:
        raise ValueError(f"Elbow motor angle {elbow_motor_deg:.1f} deg is below limit {elbow_motor_min_deg:.1f} deg")
    if elbow_motor_max_deg is not None and elbow_motor_deg > elbow_motor_max_deg:
        raise ValueError(f"Elbow motor angle {elbow_motor_deg:.1f} deg exceeds limit {elbow_motor_max_deg:.1f} deg")


def solve_arm_target(
    x_target,
    y_target,
    L2,
    L3,
    elbow_branch="down",
    shoulder_motor_min_deg=None,
    shoulder_motor_max_deg=None,
    elbow_motor_min_deg=None,
    elbow_motor_max_deg=None,
    shoulder_motor_zero_deg=0.0,
    elbow_motor_zero_deg=0.0,
    shoulder_motor_dir=1,
    elbow_motor_dir=1,
):
    phi, theta = ik_2link(x_target, y_target, L2, L3, elbow_branch)
    shoulder_motor_deg, elbow_motor_deg = to_motor_angles_deg(
        phi,
        theta,
        shoulder_motor_zero_deg=shoulder_motor_zero_deg,
        elbow_motor_zero_deg=elbow_motor_zero_deg,
        shoulder_motor_dir=shoulder_motor_dir,
        elbow_motor_dir=elbow_motor_dir,
    )
    check_motor_limits(
        shoulder_motor_deg,
        elbow_motor_deg,
        shoulder_motor_min_deg=shoulder_motor_min_deg,
        shoulder_motor_max_deg=shoulder_motor_max_deg,
        elbow_motor_min_deg=elbow_motor_min_deg,
        elbow_motor_max_deg=elbow_motor_max_deg,
    )
    x_coords, y_coords = forward_kinematics_2link(phi, theta, L2, L3)
    theta_1_deg, theta_2_deg, theta_2_absolute_deg = to_com_right_arm_angles_deg(phi, theta)
    return {
        "x_target": x_target,
        "y_target": y_target,
        "phi": phi,
        "theta": theta,
        "theta_1_deg": theta_1_deg,
        "theta_2_deg": theta_2_deg,
        "theta_2_absolute_deg": theta_2_absolute_deg,
        "x_coords": np.array(x_coords),
        "y_coords": np.array(y_coords),
        "hand_pos": np.array([x_coords[-1], y_coords[-1]]),
        "shoulder_motor_deg": shoulder_motor_deg,
        "elbow_motor_deg": elbow_motor_deg,
    }


def sample_workspace(
    L2,
    L3,
    shoulder_motor_min_deg,
    shoulder_motor_max_deg,
    elbow_motor_min_deg,
    elbow_motor_max_deg,
    shoulder_motor_zero_deg,
    elbow_motor_zero_deg,
    shoulder_motor_dir,
    elbow_motor_dir,
    samples=100,
):
    shoulder_min = -180.0 if shoulder_motor_min_deg is None else shoulder_motor_min_deg
    shoulder_max = 180.0 if shoulder_motor_max_deg is None else shoulder_motor_max_deg
    elbow_min = -180.0 if elbow_motor_min_deg is None else elbow_motor_min_deg
    elbow_max = 180.0 if elbow_motor_max_deg is None else elbow_motor_max_deg

    shoulder_vals = np.linspace(shoulder_min, shoulder_max, samples)
    elbow_vals = np.linspace(elbow_min, elbow_max, samples)
    xs = []
    ys = []

    for shoulder_motor_deg in shoulder_vals:
        for elbow_motor_deg in elbow_vals:
            phi, theta = from_motor_angles_deg(
                shoulder_motor_deg,
                elbow_motor_deg,
                shoulder_motor_zero_deg=shoulder_motor_zero_deg,
                elbow_motor_zero_deg=elbow_motor_zero_deg,
                shoulder_motor_dir=shoulder_motor_dir,
                elbow_motor_dir=elbow_motor_dir,
            )
            x_coords, y_coords = forward_kinematics_2link(phi, theta, L2, L3)
            xs.append(x_coords[-1])
            ys.append(y_coords[-1])

    return np.array(xs), np.array(ys)


class InteractiveArmIK:
    def __init__(self, args):
        self.args = args
        # Right arm workspace (right-arm-local frame)
        self.workspace_x, self.workspace_y = sample_workspace(
            args.L2,
            args.L3,
            args.right_shoulder_min_deg,
            args.right_shoulder_max_deg,
            args.right_elbow_min_deg,
            args.right_elbow_max_deg,
            args.shoulder_motor_zero_deg,
            args.elbow_motor_zero_deg,
            args.shoulder_motor_dir,
            args.elbow_motor_dir,
        )
        # Left arm workspace (left-arm-local frame)
        self.left_workspace_x, self.left_workspace_y = sample_workspace(
            args.L2,
            args.L3,
            args.left_shoulder_min_deg,
            args.left_shoulder_max_deg,
            args.left_elbow_min_deg,
            args.left_elbow_max_deg,
            args.shoulder_motor_zero_deg,
            args.elbow_motor_zero_deg,
            args.shoulder_motor_dir,
            args.elbow_motor_dir,
        )
        self.last_valid = self._home_solution()
        self.last_valid_left = self._home_left_solution()

        self.fig = plt.figure(figsize=(12, 9))
        self.fig.canvas.manager.set_window_title("Interactive Arm IK + COM")
        self.ax_arm = self.fig.add_axes([0.07, 0.34, 0.58, 0.59])
        self.ax_info = self.fig.add_axes([0.70, 0.34, 0.26, 0.59])
        self.ax_info.axis("off")

        self._setup_arm_plot()
        self._setup_controls()
        self._update_solution(None)

    def _home_left_solution(self):
        kwargs = dict(
            elbow_branch=self.args.elbow_branch,
            shoulder_motor_min_deg=self.args.left_shoulder_min_deg,
            shoulder_motor_max_deg=self.args.left_shoulder_max_deg,
            elbow_motor_min_deg=self.args.left_elbow_min_deg,
            elbow_motor_max_deg=self.args.left_elbow_max_deg,
            shoulder_motor_zero_deg=self.args.shoulder_motor_zero_deg,
            elbow_motor_zero_deg=self.args.elbow_motor_zero_deg,
            shoulder_motor_dir=self.args.shoulder_motor_dir,
            elbow_motor_dir=self.args.elbow_motor_dir,
        )
        try:
            return solve_left_arm_target(self.args.lx, self.args.ly, self.args.L2, self.args.L3, **kwargs)
        except ValueError:
            # Fall back to straight-up at full reach (no limits)
            return solve_left_arm_target(
                0.0,
                self.args.L2 + self.args.L3 - 1e-4,
                self.args.L2,
                self.args.L3,
                elbow_branch=self.args.elbow_branch,
            )

    def _home_solution(self):
        try:
            return solve_arm_target(
                self.args.L2 + self.args.L3,
                0.0,
                self.args.L2,
                self.args.L3,
                elbow_branch=self.args.elbow_branch,
                shoulder_motor_min_deg=self.args.right_shoulder_min_deg,
                shoulder_motor_max_deg=self.args.right_shoulder_max_deg,
                elbow_motor_min_deg=self.args.right_elbow_min_deg,
                elbow_motor_max_deg=self.args.right_elbow_max_deg,
                shoulder_motor_zero_deg=self.args.shoulder_motor_zero_deg,
                elbow_motor_zero_deg=self.args.elbow_motor_zero_deg,
                shoulder_motor_dir=self.args.shoulder_motor_dir,
                elbow_motor_dir=self.args.elbow_motor_dir,
            )
        except ValueError:
            shoulder_home = 0.0
            elbow_home = 0.0
            phi, theta = from_motor_angles_deg(
                shoulder_home,
                elbow_home,
                shoulder_motor_zero_deg=self.args.shoulder_motor_zero_deg,
                elbow_motor_zero_deg=self.args.elbow_motor_zero_deg,
                shoulder_motor_dir=self.args.shoulder_motor_dir,
                elbow_motor_dir=self.args.elbow_motor_dir,
            )
            x_coords, y_coords = forward_kinematics_2link(phi, theta, self.args.L2, self.args.L3)
            theta_1_deg, theta_2_deg, theta_2_absolute_deg = to_com_right_arm_angles_deg(phi, theta)
            return {
                "x_target": x_coords[-1],
                "y_target": y_coords[-1],
                "phi": phi,
                "theta": theta,
                "theta_1_deg": theta_1_deg,
                "theta_2_deg": theta_2_deg,
                "theta_2_absolute_deg": theta_2_absolute_deg,
                "x_coords": np.array(x_coords),
                "y_coords": np.array(y_coords),
                "hand_pos": np.array([x_coords[-1], y_coords[-1]]),
                "shoulder_motor_deg": shoulder_home,
                "elbow_motor_deg": elbow_home,
            }

    def _slider_bounds(self):
        # Plot view: body at origin, shoulders at +/- pivot, reach extends beyond.
        reach = LINK1_LENGTH + LINK2_LENGTH
        extent = BODY_PIVOT_OFFSET + reach + 0.05
        full_x_min = -extent
        full_x_max = extent
        full_y_min = min(TAIL_PIVOT_Y - TAIL_LENGTH, -reach) - 0.05
        full_y_max = reach + 0.05

        x_min = self.args.x_min if self.args.x_min is not None else full_x_min
        x_max = self.args.x_max if self.args.x_max is not None else full_x_max
        y_min = self.args.y_min if self.args.y_min is not None else full_y_min
        y_max = self.args.y_max if self.args.y_max is not None else full_y_max
        return x_min, x_max, y_min, y_max

    def _target_slider_bounds(self):
        """Bounds for the target X/Y sliders (arm-local, relative to right shoulder).

        Covers the full geometric reach circle so targets above/below the
        motor-limited workspace are still selectable (IK will flag invalid).
        """
        reach = self.args.L2 + self.args.L3
        margin = 0.02
        return (-reach - margin, reach + margin, -reach - margin, reach + margin)

    def _setup_arm_plot(self):
        x_min, x_max, y_min, y_max = self._slider_bounds()
        self.ax_arm.set_xlim(x_min, x_max)
        self.ax_arm.set_ylim(y_min, y_max)
        self.ax_arm.set_aspect("equal")
        self.ax_arm.grid(True, alpha=0.3)
        self.ax_arm.set_xlabel("X (m)")
        self.ax_arm.set_ylabel("Y (m)")
        self.ax_arm.set_title("Interactive 2-Link Arm IK + COM")

        # Right arm motor-limited workspace: arm-local +x -> plot +x,
        # shifted to right shoulder at +pivot.
        self.ax_arm.scatter(
            self.workspace_x + BODY_PIVOT_OFFSET,
            self.workspace_y,
            s=3,
            color="#cbd5e1",
            alpha=0.25,
            label="Right workspace",
        )
        # Left arm workspace: mirror the left arm's own workspace cloud
        # about the y-axis by negating x, shifted to left shoulder at -pivot.
        self.ax_arm.scatter(
            self.left_workspace_x - BODY_PIVOT_OFFSET,
            self.left_workspace_y,
            s=3,
            color="#bbf7d0",
            alpha=0.25,
            label="Left workspace",
        )
        # Reach circles for both arms
        self.ax_arm.add_patch(
            plt.Circle(
                (BODY_PIVOT_OFFSET, 0),
                self.args.L2 + self.args.L3,
                fill=False,
                linestyle=":",
                color="gray",
                alpha=0.4,
            )
        )
        self.ax_arm.add_patch(
            plt.Circle(
                (-BODY_PIVOT_OFFSET, 0),
                self.args.L2 + self.args.L3,
                fill=False,
                linestyle=":",
                color="gray",
                alpha=0.4,
            )
        )

        # Body
        self.body_line, = self.ax_arm.plot([], [], color="#6b7280", linewidth=6, solid_capstyle="round", label="Body")
        # Right arm
        self.arm_line, = self.ax_arm.plot([], [], color="#0f766e", linewidth=4, label="Right arm")
        self.upper_link, = self.ax_arm.plot([], [], color="#1d4ed8", linewidth=3)
        self.lower_link, = self.ax_arm.plot([], [], color="#dc2626", linewidth=3)
        self.r_gripper_line, = self.ax_arm.plot([], [], color="#7c3aed", linewidth=2, linestyle="--")
        # Left arm
        self.left_upper_link, = self.ax_arm.plot([], [], color="#1d4ed8", linewidth=3, alpha=0.6)
        self.left_lower_link, = self.ax_arm.plot([], [], color="#dc2626", linewidth=3, alpha=0.6)
        self.left_arm_line, = self.ax_arm.plot([], [], color="#16a34a", linewidth=4, alpha=0.7, label="Left arm")
        self.l_gripper_line, = self.ax_arm.plot([], [], color="#7c3aed", linewidth=2, linestyle="--", alpha=0.6)
        # Tail
        self.tail_line, = self.ax_arm.plot([], [], color="#d97706", linewidth=3, label="Tail")
        # Markers
        self.target_marker = self.ax_arm.scatter([], [], c="#f59e0b", marker="x", s=100, label="Right target")
        self.left_target_marker = self.ax_arm.scatter([], [], c="#f97316", marker="x", s=100, label="Left target")
        self.hand_marker = self.ax_arm.scatter([], [], c="#111827", marker="o", s=70, label="Solved hand")
        self.com_marker = self.ax_arm.scatter([], [], c="#0891b2", marker="D", s=70, label="COM")
        self.r_shoulder_marker = self.ax_arm.scatter([], [], c="black", marker="o", s=90)
        self.l_shoulder_marker = self.ax_arm.scatter([], [], c="black", marker="o", s=90)
        self.body_center_marker = self.ax_arm.scatter([], [], c="#6b7280", marker="s", s=50, label="Body center")
        self.status_banner = self.ax_arm.text(
            0.02,
            0.98,
            "",
            transform=self.ax_arm.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#ecfccb", edgecolor="#4d7c0f", alpha=0.9),
        )
        self.ax_arm.legend(loc="lower left", fontsize=9)

        self.info_text = self.ax_info.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            linespacing=1.35,
        )

    def _setup_controls(self):
        x_min, x_max, y_min, y_max = self._target_slider_bounds()

        ax_rx = self.fig.add_axes([0.10, 0.26, 0.78, 0.03], facecolor="#f1f5f9")
        ax_ry = self.fig.add_axes([0.10, 0.21, 0.78, 0.03], facecolor="#f1f5f9")
        ax_lx = self.fig.add_axes([0.10, 0.16, 0.78, 0.03], facecolor="#f1f5f9")
        ax_ly = self.fig.add_axes([0.10, 0.11, 0.78, 0.03], facecolor="#f1f5f9")
        ax_theta_tail = self.fig.add_axes([0.10, 0.06, 0.78, 0.03], facecolor="#f1f5f9")
        ax_reset = self.fig.add_axes([0.82, 0.01, 0.12, 0.04])

        rx_init = max(x_min, min(x_max, self.args.x))
        ry_init = max(y_min, min(y_max, self.args.y))
        lx_init = max(x_min, min(x_max, self.args.lx))
        ly_init = max(y_min, min(y_max, self.args.ly))
        self.x_slider = Slider(ax_rx, "Right Target X (m)", x_min, x_max, valinit=rx_init)
        self.y_slider = Slider(ax_ry, "Right Target Y (m)", y_min, y_max, valinit=ry_init)
        self.lx_slider = Slider(ax_lx, "Left Target X (m)", x_min, x_max, valinit=lx_init)
        self.ly_slider = Slider(ax_ly, "Left Target Y (m)", y_min, y_max, valinit=ly_init)
        self.theta_tail_slider = Slider(
            ax_theta_tail,
            "theta_tail (deg)",
            self.args.theta_tail_min,
            self.args.theta_tail_max,
            valinit=self.args.theta_tail,
        )

        self.reset_button = Button(ax_reset, "Reset", color="#e2e8f0", hovercolor="#cbd5e1")

        self.x_slider.on_changed(self._update_solution)
        self.y_slider.on_changed(self._update_solution)
        self.lx_slider.on_changed(self._update_solution)
        self.ly_slider.on_changed(self._update_solution)
        self.theta_tail_slider.on_changed(self._update_solution)
        self.reset_button.on_clicked(self._reset_sliders)

    def _reset_sliders(self, _event):
        self.x_slider.reset()
        self.y_slider.reset()
        self.lx_slider.reset()
        self.ly_slider.reset()
        self.theta_tail_slider.reset()

    def _set_arm_from_solution(self, solution, color="#0f766e"):
        # Plot frame: body center at origin.
        # Right shoulder at (+BODY_PIVOT_OFFSET, 0), right-arm-local +x
        # maps directly to plot +x (outward to the right).
        rs_x = BODY_PIVOT_OFFSET
        x_local = solution["x_coords"]
        y_local = solution["y_coords"]
        x_plot = x_local + rs_x
        y_plot = y_local
        self.arm_line.set_data(x_plot, y_plot)
        self.arm_line.set_color(color)
        self.upper_link.set_data([x_plot[0], x_plot[1]], [y_plot[0], y_plot[1]])
        self.lower_link.set_data([x_plot[1], x_plot[2]], [y_plot[1], y_plot[2]])
        self.hand_marker.set_offsets([[x_plot[-1], y_plot[-1]]])
        self.r_shoulder_marker.set_offsets([[rs_x, 0.0]])

        # Right gripper: parallelogram linkage keeps it vertical at all times
        hand_x = x_plot[-1]
        hand_y = y_plot[-1]
        self.r_gripper_line.set_data(
            [hand_x, hand_x],
            [hand_y, hand_y + GRIPPER_DRAW_LENGTH],
        )

    def _draw_body_and_left_arm_and_tail(self, left_solution, theta_tail_deg, left_color="#16a34a"):
        # Plot frame: body center at origin.
        # Right shoulder at (+pivot, 0), left shoulder at (-pivot, 0).
        body_cx = 0.0
        body_cy = 0.0
        right_shoulder = (BODY_PIVOT_OFFSET, 0.0)
        left_shoulder = (-BODY_PIVOT_OFFSET, 0.0)

        # Body line between the two shoulders
        self.body_line.set_data(
            [right_shoulder[0], left_shoulder[0]],
            [right_shoulder[1], left_shoulder[1]],
        )
        self.body_center_marker.set_offsets([[body_cx, body_cy]])
        self.l_shoulder_marker.set_offsets([[left_shoulder[0], left_shoulder[1]]])

        # Left arm: left-local +x points outward (to the robot's left),
        # which in plot frame is -x. Negate local x and shift to left shoulder.
        lx_local = left_solution["x_coords_local"]
        ly_local = left_solution["y_coords_local"]
        lx_plot = -lx_local + left_shoulder[0]
        ly_plot = ly_local + left_shoulder[1]

        self.left_arm_line.set_data(lx_plot, ly_plot)
        self.left_arm_line.set_color(left_color)
        self.left_upper_link.set_data([lx_plot[0], lx_plot[1]], [ly_plot[0], ly_plot[1]])
        self.left_lower_link.set_data([lx_plot[1], lx_plot[2]], [ly_plot[1], ly_plot[2]])

        # Left gripper: parallelogram linkage keeps it vertical at all times
        l_hand_x = lx_plot[-1]
        l_hand_y = ly_plot[-1]
        self.l_gripper_line.set_data(
            [l_hand_x, l_hand_x],
            [l_hand_y, l_hand_y + GRIPPER_DRAW_LENGTH],
        )

        # Tail: pivot at (body_cx, TAIL_PIVOT_Y). GUI angle 0 = vertical-up,
        # so shift into the internal (cos, -sin) convention.
        tail_pivot_x = body_cx
        tail_pivot_y = TAIL_PIVOT_Y
        tail_angle = math.radians(theta_tail_deg + TAIL_GUI_OFFSET_DEG + TAIL_ANGLE_STOP_DEG)
        tail_end_x = tail_pivot_x + TAIL_LENGTH * math.cos(tail_angle)
        tail_end_y = tail_pivot_y - TAIL_LENGTH * math.sin(tail_angle)
        self.tail_line.set_data([tail_pivot_x, tail_end_x], [tail_pivot_y, tail_end_y])

    def _compute_com_state(self, solution, left_solution):
        theta_1_deg = solution["theta_1_deg"]
        theta_2_deg = solution["theta_2_deg"]
        theta_2_absolute_deg = solution["theta_2_absolute_deg"]

        theta_11_deg = left_solution["theta_11_deg"]
        theta_12_deg = left_solution["theta_12_deg"]
        theta_12_absolute_deg = left_solution["theta_12_absolute_deg"]
        theta_tail_deg = self.theta_tail_slider.val
        # Translate GUI tail angle (0 = vertical-up) to the internal convention
        # expected by the unmodified COM_Prediction function.
        theta_tail_internal_deg = theta_tail_deg + TAIL_GUI_OFFSET_DEG

        x_com_in, y_com_in = COM_Prediction(
            theta_1_deg,
            theta_2_deg,
            theta_11_deg,
            theta_12_deg,
            theta_tail_internal_deg,
        )

        return {
            "theta_1_deg": theta_1_deg,
            "theta_2_deg": theta_2_deg,
            "theta_2_absolute_deg": theta_2_absolute_deg,
            "theta_11_deg": theta_11_deg,
            "theta_12_deg": theta_12_deg,
            "theta_12_absolute_deg": theta_12_absolute_deg,
            "theta_tail_deg": theta_tail_deg,
            "x_com_in": x_com_in,
            "y_com_in": y_com_in,
            "x_com_m": x_com_in * INCH_TO_M,
            "y_com_m": y_com_in * INCH_TO_M,
        }

    def _format_info(self, solution, target_x, target_y, lx_target, ly_target, status, com_state, error_text=None):
        lines = [
            f"status {status}",
            f"R target  ({target_x:6.3f}, {target_y:6.3f}) m",
            f"L target  ({lx_target:6.3f}, {ly_target:6.3f}) m",
            "",
            f"theta_1       {com_state['theta_1_deg']:8.2f} deg",
            f"theta_2       {com_state['theta_2_deg']:8.2f} deg",
            f"theta_2_abs   {com_state['theta_2_absolute_deg']:8.2f} deg",
            "",
            f"theta_11      {com_state['theta_11_deg']:8.2f} deg",
            f"theta_12      {com_state['theta_12_deg']:8.2f} deg",
            f"theta_12_abs  {com_state['theta_12_absolute_deg']:8.2f} deg",
            f"theta_tail    {com_state['theta_tail_deg']:8.2f} deg",
            "",
            f"shoulder_motor {solution['shoulder_motor_deg']:7.2f} deg",
            f"elbow_motor    {solution['elbow_motor_deg']:7.2f} deg",
            "",
            f"hand x {solution['hand_pos'][0]:8.3f} m",
            f"hand y {solution['hand_pos'][1]:8.3f} m",
            "",
            f"COM x {com_state['x_com_m']:8.3f} m ({com_state['x_com_in']:7.2f} in)",
            f"COM y {com_state['y_com_m']:8.3f} m ({com_state['y_com_in']:7.2f} in)",
            "",
            f"R shldr lim {self.args.right_shoulder_min_deg:+6.1f} to {self.args.right_shoulder_max_deg:+6.1f}",
            f"R elbow lim {self.args.right_elbow_min_deg:+6.1f} to {self.args.right_elbow_max_deg:+6.1f}",
            f"L shldr lim {self.args.left_shoulder_min_deg:+6.1f} to {self.args.left_shoulder_max_deg:+6.1f}",
            f"L elbow lim {self.args.left_elbow_min_deg:+6.1f} to {self.args.left_elbow_max_deg:+6.1f}",
        ]

        if error_text:
            lines.extend(["", error_text])
        return "\n".join(lines)

    def _update_solution(self, _value):
        target_x = self.x_slider.val
        target_y = self.y_slider.val
        lx_target = self.lx_slider.val
        ly_target = self.ly_slider.val
        # Plot-frame target markers
        # Right target: right-arm-local +x outward -> plot +x, shifted by +pivot
        self.target_marker.set_offsets([[target_x + BODY_PIVOT_OFFSET, target_y]])
        # Left target: left-arm-local +x outward -> plot -x, shifted by -pivot
        self.left_target_marker.set_offsets([[-lx_target - BODY_PIVOT_OFFSET, ly_target]])

        error_text = None
        status = "valid"

        # Right arm
        try:
            solution = solve_arm_target(
                target_x,
                target_y,
                self.args.L2,
                self.args.L3,
                elbow_branch=self.args.elbow_branch,
                shoulder_motor_min_deg=self.args.right_shoulder_min_deg,
                shoulder_motor_max_deg=self.args.right_shoulder_max_deg,
                elbow_motor_min_deg=self.args.right_elbow_min_deg,
                elbow_motor_max_deg=self.args.right_elbow_max_deg,
                shoulder_motor_zero_deg=self.args.shoulder_motor_zero_deg,
                elbow_motor_zero_deg=self.args.elbow_motor_zero_deg,
                shoulder_motor_dir=self.args.shoulder_motor_dir,
                elbow_motor_dir=self.args.elbow_motor_dir,
            )
            self.last_valid = solution
            display_solution = solution
            self._set_arm_from_solution(display_solution, color="#0f766e")
            right_status = "valid"
        except ValueError as exc:
            right_status = "invalid"
            error_text = f"right: {exc}"
            display_solution = self.last_valid
            self._set_arm_from_solution(display_solution, color="#9ca3af")

        # Left arm
        try:
            left_solution = solve_left_arm_target(
                lx_target,
                ly_target,
                self.args.L2,
                self.args.L3,
                elbow_branch=self.args.elbow_branch,
                shoulder_motor_min_deg=self.args.left_shoulder_min_deg,
                shoulder_motor_max_deg=self.args.left_shoulder_max_deg,
                elbow_motor_min_deg=self.args.left_elbow_min_deg,
                elbow_motor_max_deg=self.args.left_elbow_max_deg,
                shoulder_motor_zero_deg=self.args.shoulder_motor_zero_deg,
                elbow_motor_zero_deg=self.args.elbow_motor_zero_deg,
                shoulder_motor_dir=self.args.shoulder_motor_dir,
                elbow_motor_dir=self.args.elbow_motor_dir,
            )
            self.last_valid_left = left_solution
            display_left = left_solution
            left_color = "#16a34a"
            left_status = "valid"
        except ValueError as exc:
            left_status = "invalid"
            left_err = f"left: {exc}"
            error_text = left_err if error_text is None else f"{error_text}\n{left_err}"
            display_left = self.last_valid_left
            left_color = "#9ca3af"

        if right_status == "valid" and left_status == "valid":
            status = "valid"
            self.status_banner.set_text("IK valid")
            self.status_banner.set_bbox(dict(boxstyle="round", facecolor="#ecfccb", edgecolor="#4d7c0f", alpha=0.9))
        else:
            status = "invalid"
            self.status_banner.set_text(f"IK invalid (R:{right_status} L:{left_status})")
            self.status_banner.set_bbox(dict(boxstyle="round", facecolor="#fee2e2", edgecolor="#b91c1c", alpha=0.9))

        com_state = self._compute_com_state(display_solution, display_left)
        # Draw the rest of the robot (body, left arm, tail)
        self._draw_body_and_left_arm_and_tail(
            display_left,
            com_state["theta_tail_deg"],
            left_color=left_color,
        )
        # COM_Prediction returns coordinates in the body frame (body center at 0).
        # Our plot frame is mirrored in x relative to the COM body frame
        # (right shoulder at +pivot in plot, but at -pivot in COM body frame),
        # so negate x to display.
        com_plot_x = -com_state["x_com_m"]
        com_plot_y = com_state["y_com_m"]
        self.com_marker.set_offsets([[com_plot_x, com_plot_y]])
        self.info_text.set_text(
            self._format_info(
                display_solution, target_x, target_y, lx_target, ly_target, status, com_state, error_text
            )
        )
        self.fig.canvas.draw_idle()

    def show(self):
        print("\nInteractive Arm IK + COM Sliders")
        print("=================================")
        print("Drag Target X and Target Y to solve right-arm IK.")
        print("Drag theta_11, theta_12, and theta_tail to update full-body COM.")
        print("The COM marker and COM text update live using COM_Prediction naming.")
        print("If the IK target is invalid, the arm stays on the last valid pose.")
        print("\nClose the window to exit.\n")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive 2-link arm IK with COM prediction")
    # Default: right hand pointing straight up from the shoulder (+y)
    parser.add_argument("--x", type=float, default=0.0, help="Initial X target position (m)")
    parser.add_argument(
        "--y",
        type=float,
        default=(LINK1_LENGTH + LINK2_LENGTH),
        help="Initial Y target position (m) — straight-up by default",
    )
    parser.add_argument("--x-min", type=float, default=None, help="Optional slider minimum for X")
    parser.add_argument("--x-max", type=float, default=None, help="Optional slider maximum for X")
    parser.add_argument("--y-min", type=float, default=None, help="Optional slider minimum for Y")
    parser.add_argument("--y-max", type=float, default=None, help="Optional slider maximum for Y")
    parser.add_argument("--L2", type=float, default=LINK1_LENGTH, help="Link 1 / upper arm length (m)")
    parser.add_argument("--L3", type=float, default=LINK2_LENGTH, help="Link 2 / forearm length (m)")
    parser.add_argument(
        "--elbow-branch",
        choices=["down", "up"],
        default="down",
        help="IK elbow branch",
    )

    # Left arm target in the LEFT shoulder's local frame
    # (origin at left shoulder, +x outward, +y up). Default: straight-up.
    parser.add_argument("--lx", type=float, default=0.0, help="Initial left-arm target X (m, left-shoulder local)")
    parser.add_argument(
        "--ly",
        type=float,
        default=(LINK1_LENGTH + LINK2_LENGTH),
        help="Initial left-arm target Y (m) — straight-up by default",
    )
    parser.add_argument("--theta-tail", type=float, default=0.0, help="Initial theta_tail angle (deg)")
    parser.add_argument("--theta-tail-min", type=float, default=-60.0, help="theta_tail slider minimum (deg)")
    parser.add_argument("--theta-tail-max", type=float, default=60.0, help="theta_tail slider maximum (deg)")

    # Per-arm motor-frame limits (applied AFTER the zero offset).
    # With shoulder-motor-zero-deg = 90, motor = theta_deg - 90, so these
    # directly describe each joint's range about the vertical-up pose.
    # Defaults come from positions_log.json (left arm, motors 16/18) and
    # are mirrored for the right arm.
    parser.add_argument("--right-shoulder-min-deg", type=float, default=-70.9)
    parser.add_argument("--right-shoulder-max-deg", type=float, default=46.0)
    parser.add_argument("--right-elbow-min-deg", type=float, default=-90.4)
    parser.add_argument("--right-elbow-max-deg", type=float, default=140.0)
    parser.add_argument("--left-shoulder-min-deg", type=float, default=-46.0)
    parser.add_argument("--left-shoulder-max-deg", type=float, default=70.9)
    parser.add_argument("--left-elbow-min-deg", type=float, default=-140.0)
    parser.add_argument("--left-elbow-max-deg", type=float, default=90.4)
    # Zero = arms vertical (+y). With dir=1, motor = theta_deg - 90,
    # so theta=90 deg (straight up) corresponds to motor angle 0.
    parser.add_argument("--shoulder-motor-zero-deg", type=float, default=90.0)
    parser.add_argument("--elbow-motor-zero-deg", type=float, default=0.0)
    parser.add_argument("--shoulder-motor-dir", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--elbow-motor-dir", type=int, choices=[-1, 1], default=1)

    args = parser.parse_args()

    try:
        viewer = InteractiveArmIK(args)
        viewer.show()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()