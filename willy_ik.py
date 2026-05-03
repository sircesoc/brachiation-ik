"""
willy_ik.py
============
Inverse and forward kinematics for the Willy double-parallelogram arm.

ACTUATION CHAIN (from Fusion model Willy_v4.5):
  * Motor 1 sits on the base and drives Comp139 (the upper arm).
  * Motor 2 is mounted ON Comp139 at the elbow, and drives Comp146 (the
    forearm) directly. So motor 2 sets the elbow angle RELATIVE to the
    upper arm.
  * Two passive parallelograms keep the wrist plate (Comp143) parallel
    to the base in every configuration:
      - Proximal parallelogram: Comp139 || Comp148 with Comp151 as the
        floating coupler.
      - Distal parallelogram: Comp146 || Comp147 with Comp143 as the
        floating coupler.
  * The parallelograms only fix the wrist orientation -- they do not
    affect end-effector position. So the IK reduces to a standard
    planar 2R serial-arm problem.

KINEMATIC MODEL:
    alpha1 = theta1 + k1   (absolute shoulder angle)
    alpha2 = theta2 + k2   (relative elbow angle)
    u = a*cos(alpha1) + L*cos(alpha1 + alpha2)
    v = a*sin(alpha1) + L*sin(alpha1 + alpha2)

WORKING PLANE:
  y-z plane of the Fusion model. Origin at motor 1's axis
  (world (13.940, 13.644, -2.019) cm). In-plane coords:
      u = world_y - 13.644
      v = world_z + 2.019

INVERSE KINEMATICS (textbook 2R):
    cos(alpha2) = (u^2 + v^2 - a^2 - L^2) / (2*a*L)
    alpha2 = +/- acos(cos(alpha2))           # +/- = elbow up/down
    alpha1 = atan2(v, u) - atan2(L*sin(alpha2), a + L*cos(alpha2))

WORKSPACE:
  Annulus from |a - L| = 27.01 cm to a + L = 46.56 cm, centered at M1.
"""

import math


# Calibrated geometry (from Fusion model home pose)
A     = 9.778      # upper arm length, |M1 -> elbow| in cm
L     = 36.786     # forearm + wrist length, |elbow -> end-effector| in cm
K1    = 27.963     # deg, alpha1 = theta1 + K1
K2    = -31.965    # deg, alpha2 = theta2 + K2 (relative elbow angle)

# World-to-plane origin (Fusion world cm)
WORLD_ORIGIN_Y = 13.644
WORLD_ORIGIN_Z = -2.019

REACH_MIN = abs(A - L)   # ~27.008 cm
REACH_MAX = A + L        # ~46.564 cm

_D2R = math.pi / 180.0
_R2D = 180.0 / math.pi


def fk(theta1_deg, theta2_deg):
    """Forward kinematics. Standard 2R: motor 2 angle is RELATIVE to motor 1."""
    a1  = (theta1_deg + K1) * _D2R
    a2  = (theta2_deg + K2) * _D2R
    a12 = a1 + a2
    u = A * math.cos(a1) + L * math.cos(a12)
    v = A * math.sin(a1) + L * math.sin(a12)
    return (u, v)


def ik(u_cm, v_cm, elbow='up'):
    """Inverse kinematics for the 2R arm.

    Parameters
    ----------
    u_cm, v_cm : float
        Target position in cm in the working plane (origin at M1).
    elbow : {'up', 'down'}, default 'up'
        Branch selection. Pick the one that matches your assembly's home pose.

    Returns
    -------
    (theta1_deg, theta2_deg) : tuple of float

    Raises
    ------
    ValueError
        If target is outside the reachable annulus.
    """
    r2 = u_cm * u_cm + v_cm * v_cm
    r  = math.sqrt(r2)
    if r > REACH_MAX - 1e-9 or r < REACH_MIN + 1e-9:
        raise ValueError(
            f"Target ({u_cm:.3f}, {v_cm:.3f}) unreachable: "
            f"r={r:.3f} cm, reach=[{REACH_MIN:.3f}, {REACH_MAX:.3f}] cm"
        )

    cos_alpha2 = (r2 - A * A - L * L) / (2.0 * A * L)
    cos_alpha2 = max(-1.0, min(1.0, cos_alpha2))   # clamp numerical noise
    alpha2 = -math.acos(cos_alpha2) if elbow == 'up' else math.acos(cos_alpha2)

    alpha1 = math.atan2(v_cm, u_cm) - math.atan2(
        L * math.sin(alpha2),
        A + L * math.cos(alpha2)
    )

    return (alpha1 * _R2D - K1, alpha2 * _R2D - K2)


def jacobian(theta1_deg, theta2_deg):
    """2x2 Jacobian matrix d(u, v) / d(theta1, theta2) in cm/rad.

    For a 2R arm:
        J = [ -a*s1 - L*s12   -L*s12 ]
            [  a*c1 + L*c12    L*c12 ]
    """
    a1  = (theta1_deg + K1) * _D2R
    a12 = a1 + (theta2_deg + K2) * _D2R
    s1, c1 = math.sin(a1), math.cos(a1)
    s12, c12 = math.sin(a12), math.cos(a12)
    return [
        [-A * s1 - L * s12, -L * s12],
        [ A * c1 + L * c12,  L * c12]
    ]


def inverse_jacobian(theta1_deg, theta2_deg):
    """Inverse Jacobian, in rad/cm. Singular at workspace boundaries."""
    J = jacobian(theta1_deg, theta2_deg)
    a, b = J[0]
    c, d = J[1]
    det = a * d - b * c
    if abs(det) < 1e-9:
        raise ValueError("Jacobian singular (workspace boundary or folded pose)")
    return [
        [ d / det, -b / det],
        [-c / det,  a / det]
    ]


def world_to_plane(y_world, z_world):
    return (y_world - WORLD_ORIGIN_Y, z_world - WORLD_ORIGIN_Z)


def plane_to_world(u_cm, v_cm):
    return (u_cm + WORLD_ORIGIN_Y, v_cm + WORLD_ORIGIN_Z)


# ============================================================
# Self-test
# ============================================================
if __name__ == '__main__':
    print("Willy double-parallelogram IK -- self test")
    print(f"  upper arm a = {A:.3f} cm,  forearm L = {L:.3f} cm")
    print(f"  K1 = {K1:+.3f} deg,  K2 = {K2:+.3f} deg")
    print(f"  reach: {REACH_MIN:.3f} <= r <= {REACH_MAX:.3f} cm")
    print()

    print("Round-trip FK -> IK -> motor angles  (elbow='up'):")
    print(f"  {'(t1,t2) in':>20s}  {'(u,v) cm':>20s}  {'(t1,t2) back':>22s}  err")
    test_inputs = [(0, 0), (5, 5), (10, -5), (-10, 5), (15, -15),
                   (20, 30), (-30, 40)]
    for t1, t2 in test_inputs:
        u, v = fk(t1, t2)
        try:
            t1b, t2b = ik(u, v, elbow='up')
            err = math.hypot(t1 - t1b, t2 - t2b)
            tag = 'OK' if err < 1e-9 else f'err={err:.2e}'
        except ValueError:
            t1b, t2b, tag = float('nan'), float('nan'), 'UNREACH'
        print(f"  ({t1:+6.1f},{t2:+6.1f})        ({u:7.3f},{v:7.3f})    "
              f"({t1b:+6.2f},{t2b:+6.2f})  {tag}")

    print()
    u0, v0 = fk(0, 0)
    print(f"Home pose FK(0,0) = ({u0:.4f}, {v0:.4f}) cm   "
          f"(model expects approx (45.33, 2.02))")

    # FK at the (41.5, 41.5) pose we observed in the model
    u1, v1 = fk(41.5, 41.5)
    print(f"FK(41.5, 41.5)    = ({u1:.4f}, {v1:.4f}) cm   "
          f"(model showed approx (43.13, 2.02))")