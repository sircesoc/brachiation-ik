# Willy IK — context

Mechanism: double-parallelogram arm.
- Motor 1 (on base) drives Comp139 — sets absolute shoulder angle
- Motor 2 (mounted on Comp139 at the elbow) drives Comp146 directly — sets
  elbow angle relative to the upper arm
- Two passive parallelograms (Comp139∥Comp148 with Comp151 floating coupler,
  Comp146∥Comp147 with Comp143 floating coupler) keep the wrist plate
  parallel to the base in every pose
- Net kinematics reduce to a textbook 2R serial arm

Calibrated geometry (cm, deg):
- a (upper arm)   = 9.778
- L (forearm+wrist) = 36.786
- K1 = +27.963
- K2 = -31.965
- Comp151 corner offsets from elbow: distal (+7.62, 0), purple (+7.62, -3.493)
- Comp143 length = 7.62 cm (J4 to J7)
- Bilateral arm spacing: 11.7055 in = 29.74 cm between M1 pivots, mirrored about v axis

Working plane: y-z plane of the Fusion model. Plane origin at motor 1 axis
(world (13.940, 13.644, -2.019) cm).

Reach annulus: 27.01 ≤ r ≤ 46.56 cm from each M1.

Files:
- willy_ik.py — fk(), ik(), jacobian(), inverse_jacobian()
- willy_ik_visualizer.html — interactive forward/inverse demo, both arms

Verified: home pose FK(0,0) = (45.33, 2.02) cm exactly matches J6 in the
Fusion model. Round-trip IK→FK is exact at machine precision on the
elbow-up branch (which matches the assembly's natural pose).

Open follow-ups:
- Independent left/right control (currently both arms share the same motors)
- Velocity Jacobian / manipulability ellipsoid visualization
- Lateral base rotation for full 3D positioning
- Trajectory planning (straight-line end-effector paths, motor-time plots)
- IK target offset to the gripper midpoint of Comp143 (currently targets J6
  on Comp146; gripper sits at J4-J7 midpoint, +3.81 cm in u from J6)
- Re-verify against live Fusion model once the joint API recovers
  (it threw InternalValidationError throughout this session)