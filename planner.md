# Brachiation planner — code summary

This file documents the U-shape "swing to next bar" planner and the closed-loop
control stack that runs on top of it. It is not a tutorial; it indexes the code
so you can find what you need quickly.

All paths are relative to the repo root. Filename:line links are clickable in
the IDE.


## Files

| File | Role |
|---|---|
| [brachiation_planner.py](brachiation_planner.py) | Geometry, planner, viz, closed-loop controller, IMU stabilizer, motor execution |
| [qp_replanner.py](qp_replanner.py) | OSQP-based MPC replanner (sub-millisecond) |
| [willy_ik.py](willy_ik.py) | Willy double-parallelogram FK / IK |
| [InvserseKinematic_COM_Combined.py](InvserseKinematic_COM_Combined.py) | Mass model used by `COM_Prediction()` |
| [COM Prediction.py](COM%20Prediction.py) | Original mass-model COM function (unmodified) |
| [imu_reader.py](imu_reader.py) | WIT-Motion serial driver |
| [motor_reader.py](motor_reader.py) | Robstride motor read/command driver |


## Plan-to-next-bar — `plan_trajectory_minimal`

The function is at [brachiation_planner.py:1218](brachiation_planner.py#L1218) and
solves a single bar-to-bar swing offline as an SLSQP problem.

### Decision variables
- 5 DOF per waypoint × N waypoints. State is
  `[t1_grip, t2_grip, t1_free, t2_free, tail]` in radians, where t1/t2 are
  Willy motor angles (the geometry angle is `motor + K1`, `motor + K2`).

### Cost
- **Smoothness** in joint space: `Σ ||q_{k+1} − q_k||²`.
- **Soft tracking** of a Cartesian target U-curve for the free-hand tip.
  See target generation at
  [brachiation_planner.py:1361](brachiation_planner.py#L1361):
  - x: smoothstep from `bar1_x` to `bar2_x`.
  - y: `bar_y − bar_clearance − (dip_depth − clearance) · sin(π · t^k)` with
    `dip_skew_k > 1` shifting the dip peak late in the swing.

### Constraints (all hard, ineq or eq)
| Name | Where | What |
|---|---|---|
| Joint bounds | per-WP box bounds | per-arm shoulder/elbow + tail limits |
| WP0 pin | bounds collapsed to a point when `pin_wp0=True` | locks the start state |
| COM band | [brachiation_planner.py:1416](brachiation_planner.py#L1416) | per-WP band on `COM_x − bar1_x`; widens after `t > com_relax_t` |
| Final tip = bar2 | [brachiation_planner.py:1456](brachiation_planner.py#L1456) | hard equality on free-hand x and y at last WP |
| Monotone progress | [brachiation_planner.py:1471](brachiation_planner.py#L1471) | `free_hand_x[k+1] ≥ free_hand_x[k]` |
| Bar clearance | [brachiation_planner.py:1480](brachiation_planner.py#L1480) | `free_hand_y ≤ bar_y − bar_clearance_m` (except catch) |
| Collisions | [brachiation_planner.py:1494](brachiation_planner.py#L1494) | parallel-link, body-bar, arm-body, inter-arm — see below |

### Two-stage solve
At [brachiation_planner.py:1248](brachiation_planner.py#L1248) — when
`enable_collision=True`, the planner first runs itself with collisions off,
WP0 pinned to a known non-colliding pose, to produce a clean warm start. The
second stage then warm-starts from that with the full collision constraint
set. This is what makes the constrained problem actually converge.

### Default tunables (CLI flags below in parentheses)
- `dip_depth_m=0.10` (`--dip-depth`) — peak dip depth.
- `dip_skew_k=4.0` (`--dip-skew`) — pushes peak to ~84% of swing.
- `com_band_m=0.03` (`--com-band`) — direct phase tolerance (≈3.4° static tilt).
- `com_band_late_m=0.05` (`--com-band-late`) — dip/catch tolerance (≈5.7°).
- `com_relax_t=0.70` (`--com-relax-t`) — when band starts widening.
- `bar_clearance_m=0.02` (`--bar-clearance`) — keeps disc under the bars.
- `track_weight=20.0` (`--track-weight`) — soft target weight.
- `tail_lim=(-60, 60)` (`--tail-min`, `--tail-max`) — tail range in deg.
- `enable_collision=True` (`--no-collision` to disable).


## Geometry helpers

| Symbol | Meaning |
|---|---|
| `A_M`, `L_M` | upper-arm and forearm lengths in metres |
| `K1`, `K2` | Willy motor offsets so `alpha = motor + K` |
| `BODY_PIVOT_OFFSET` | half of bilateral shoulder spacing (≈ 0.149 m) |
| `BAR_Y` | default bar height in metres |
| `LINK_HALF_WIDTH_M` = 0.625" | half-thickness of parallelogram links |
| `DISC_RADIUS_M` = 2" | wrist-plate disc radius (4" diameter) |
| `BAR_RADIUS_M` = 0.5" | bar radius |

Key kinematic functions:
- [`shoulder_from_bar`](brachiation_planner.py#L548) — back-solves grip-arm
  shoulder world position from a bar position and motor angles.
- [`body_and_free_shoulder`](brachiation_planner.py#L564) — body centre and
  free-shoulder world positions from grip shoulder.
- [`free_hand_world`](brachiation_planner.py#L580) — free-arm gripper-tip
  world position.
- [`compute_com_world_x_and_grad`](brachiation_planner.py#L696) — body-frame
  COM_x in world units plus its analytic 5-vector gradient. Used by the COM
  constraint and the QP replanner.
- [`_arm_landmarks`](brachiation_planner.py#L1066) — returns world-frame
  positions of M1, M2, elbow, J_distal, J_purple, J4, J7, disc for both arms.
  This is the workhorse for the collision constraints.


## Collision system

The planner uses light-weight 2D distance constraints rather than full mesh
collision. Threshold values in [brachiation_planner.py:949-955](brachiation_planner.py#L949-L955).

### What is enforced
1. **Same-arm parallel link pairs.** Comp139 ∥ Comp148 (M1→elbow vs M2→J_purple)
   and Comp146 ∥ Comp147 (elbow→J4 vs J_distal→J7) for both arms. Threshold:
   centre-line distance ≥ 1.25" (sum of half-widths).
2. **Body-line vs bars.** Segment between the two shoulders must clear bar1
   and bar2 (skipped at catch).
3. **Arm vs body box.** Each arm's forearm and wrist plate vs the four edges
   of the body rectangle (4" tall) below the shoulders.
4. **Inter-arm pairs.** Free upper-arm and forearm vs grip upper-arm and
   forearm (4 link-link pairs). The bilateral spacing makes 3D collision
   impossible, but the 2D check keeps the visualisation clean and provides
   safety margin.
5. **Inter-arm disc vs other-arm links.** Disc rendered as a horizontal line
   of length `2 × DISC_RADIUS_M` (perpendicular to the green ee line, i.e.
   edge-on in the swing view) at `ee_base`. Distance ≥ link half-width.
6. **Disc vs disc.** Distance ≥ 5 mm.

### What is intentionally NOT enforced
- Free-disc vs grip-disc collision at the catch waypoint (they end up close
  by construction).
- Anything to do with bar2 collision at the catch waypoint (the gripper IS
  on the bar there).

### Distance primitives
- [`_point_seg_dist_sq`](brachiation_planner.py#L890) — squared min distance
  from a point to a segment.
- [`_seg_seg_dist_sq`](brachiation_planner.py#L908) — squared min distance
  between two 2D segments (handles intersection → 0).


## Visualisation overlay

`draw_robot_at_waypoint` ([brachiation_planner.py:1620](brachiation_planner.py#L1620))
overlays `arm_collision_polygons` output as filled polygons before the line
drawing. Each link is a 1.25"-wide rectangle, the Comp151 triangle is a
filled triangle, and the disc is a horizontal 4"-long thin rectangle at the
wrist-plate midpoint. Toggle with `show_widths=True/False`.


## Closed-loop control stack

Three layers, each running at a different rate.

```
   IMU + motors (200–500 Hz)
        │
        ▼
  ┌─────────────────────┐    Δtail
  │  InnerStabilizer    │ ─────────────┐
  │  PD on tilt → tail  │              │
  └─────────────────────┘              ▼
        ▲                       motor command (q)
        │ q_ref (10–50 Hz)
  ┌─────────────────────┐
  │  ReplanningCtrl     │ ◄── measured tilt + joint state
  │  QPReplanner (MPC)  │
  └─────────────────────┘
        ▲
        │ trajectory (once per swing)
  ┌─────────────────────┐
  │  plan_trajectory_   │
  │  minimal (SLSQP)    │
  └─────────────────────┘
```

### Layer 3 — offline plan
`plan_trajectory_minimal` produces the smooth U reference. Tilt-naive
(assumes φ = 0). One call per swing.

### Layer 2 — `ReplanningController` + `QPReplanner` MPC
- [`ReplanningController`](brachiation_planner.py#L2030) — wraps the SLSQP
  trajectory, monitors IMU/motor drift, triggers replans.
- [`QPReplanner.replan(x_ref, current_state_rad, current_tilt_rad)`](qp_replanner.py#L242)
  — pre-allocated OSQP problem. Re-evaluates the linearised COM and goal
  constraints at every tick. Sub-millisecond solve.
- **Tilt awareness**:
  - **COM constraint** stays in **body frame** ([brachiation_planner.py:2336](brachiation_planner.py#L2336)). The deliberate choice: commanding a config whose body-frame COM is at bar1 means the body wants to hang vertically, so any disturbance tilt is a restoring-torque condition.
  - **Goal constraint** uses **world frame** ([brachiation_planner.py:2359](brachiation_planner.py#L2359)). The body-frame tip is rotated by the measured `current_tilt_rad` around bar1, so the joint config the QP picks has a *rendered* tip on bar2 even when the body is tilted.
  - `current_tilt_rad` is passed explicitly via the `replan()` API and through
    a small `_call_eval` shim ([qp_replanner.py:39](qp_replanner.py#L39))
    that supports both old (`fn(state)`) and new (`fn(state, tilt)`) eval_fn
    signatures.

### Layer 1 — `InnerStabilizer`
[brachiation_planner.py:2025](brachiation_planner.py#L2025). PD on body tilt
with the tail as actuator:

```
Δtail = tail_sign · (kp · (φ_meas − φ_ref) + kd · φ̇_meas)
```

Defaults `kp=30 deg/rad`, `kd=5 deg/(rad/s)`, `tail_sign=−1`. Output clamped
to `±max_correction_deg` and to motor `tail_limits_deg`. `apply()` returns
the planner's `q_ref` dict with `tail_gui_deg` adjusted; other joints pass
through unchanged.


## SLSQP math — `plan_trajectory_minimal`

The offline planner is a single nonlinear program solved with
`scipy.optimize.minimize(method="SLSQP")`. Finite-difference Jacobians for
the cost; analytic Jacobian only for the (currently disabled) hard COM
equality variant.

### Decision variables

```
x ∈ ℝ^{N·5}        flat state vector, reshaped to (N, 5) for indexing
                   each row x[k] = [t1g, t2g, t1f, t2f, tail] in radians
```

### Forward kinematics

```
shoulder_grip(x[k])  = shoulder_from_bar(bar1, t1g, t2g)
shoulder_free(x[k])  = body_and_free_shoulder(shoulder_grip)[1]
tip(x[k])            = free_hand_world(shoulder_free, t1f, t2f)        ∈ ℝ²
COM_x(x[k])          = compute_com_world_x_and_grad(x[k])[0]            ∈ ℝ
```

All in body frame (no tilt rotation in the offline plan).

### Cartesian U-target

For each `k = 0 .. N-1` with `t_k = k / (N-1)`:

```
s_k     = t_k² · (3 − 2 t_k)                          smoothstep ease-in-out
bump_k  = max(0, sin(π · t_k^p))                      p = dip_skew_k
                                                      p > 1 shifts peak late
target_x_k = bar1_x + (bar2_x − bar1_x) · s_k
target_y_k = bar_y − bar_clearance
              − (dip_depth − bar_clearance) · bump_k
target_{N-1} = (bar2_x, bar_y)                        exact catch
```

Built once at [brachiation_planner.py:1361](brachiation_planner.py#L1361).

### Cost

```
J(x) = w_smooth · Σ_{k=1..N-1}  ‖x[k] − x[k−1]‖²
     + w_track  · Σ_{k=0..N-1}  ‖tip(x[k]) − target_k‖²
```

Implementation at [brachiation_planner.py:1380](brachiation_planner.py#L1380).
SLSQP uses finite-difference for ∇J.

### Constraints

All inequality constraints are written as `g(x) ≥ 0`. Equalities as
`h(x) = 0`. SLSQP supports both.

**(1) Joint box bounds** — per-WP per-joint, applied via `bounds=` argument.

**(2) WP 0 pin** — when `pin_wp0=True` and `start_state_deg` is supplied, the
first 5 box bounds are collapsed to a degenerate interval `[v, v]`. SLSQP
handles this far better than redundant equality constraints alongside box
bounds. See [brachiation_planner.py:1286-1294](brachiation_planner.py#L1286-L1294).

**(3) Final tip equality** — `h(x) = 0`:
```
tip_x(x[N-1]) − bar2_x = 0
tip_y(x[N-1]) − bar_y  = 0
```

**(4) Monotone progress** — for each `k = 0 .. N-2`:
```
tip_x(x[k+1]) − tip_x(x[k]) ≥ 0
```

**(5) Bar clearance** — for each `k = 0 .. N-2`:
```
(bar_y − bar_clearance_m) − tip_y(x[k]) ≥ 0
```

**(6) COM band** — per-WP, with the band widening late in the swing:
```
band(t_k) = com_band_m  if  t_k ≤ com_relax_t
          = com_band_m + (t_k − com_relax_t) · (com_band_late_m − com_band_m)
                          / (1 − com_relax_t)        otherwise
```

Two inequalities per WP:
```
COM_x(x[k]) − (bar1_x − band(t_k)) ≥ 0     lower
(bar1_x + band(t_k)) − COM_x(x[k]) ≥ 0     upper
```

If both `com_band_m` and `com_band_late_m` are zero, the planner switches to
a hard equality `COM_x(x[k]) − bar1_x = 0` with the analytic 5-vector
gradient supplied by `compute_com_world_x_and_grad`.

**(7) Same-arm parallel link distance** — for each arm and each parallel
pair (Comp139∥Comp148, Comp146∥Comp147):
```
seg_seg_dist²(arm.shoulder→arm.elbow,
              arm.M2→arm.J_purple)        ≥ (2·LINK_HALF_WIDTH)²
seg_seg_dist²(arm.elbow→arm.J4,
              arm.J_distal→arm.J7)        ≥ (2·LINK_HALF_WIDTH)²
```

**(8) Body line vs bars** — `body_line` is the segment between the two
shoulders (`grip_shoulder → free_shoulder`):
```
point_seg_dist²(bar1, body_line) ≥ (LINK_HALF_WIDTH + BAR_RADIUS)²
point_seg_dist²(bar2, body_line) ≥ (LINK_HALF_WIDTH + BAR_RADIUS)²    (skip catch)
```

**(9) Arm vs body box** — body box is a rectangle of width
`|grip_shoulder.x − free_shoulder.x|` and height 4" hanging below the
shoulder line. For each arm, for `(forearm, wrist_plate)` × each of the 4
box edges:
```
seg_seg_dist²(arm_segment, box_edge) ≥ LINK_HALF_WIDTH²
```

**(10) Inter-arm link-link** — for each pair in
`{(free_upper, grip_upper), (free_upper, grip_forearm),
   (free_forearm, grip_upper), (free_forearm, grip_forearm)}`:
```
seg_seg_dist²(free_seg, grip_seg) ≥ (2·LINK_HALF_WIDTH)²
```

**(11) Inter-arm disc vs other-arm link** — disc is rendered as a horizontal
segment of length `2·DISC_RADIUS_M` at `ee_base` (perpendicular to the
gripper-tip line). For each disc side and each of the other arm's three
segments (upper, forearm, wrist plate):
```
seg_seg_dist²(disc_horizontal_segment, other_arm_segment) ≥ LINK_HALF_WIDTH²
```

**(12) Inter-arm disc-disc** — except at the catch:
```
seg_seg_dist²(free_disc, grip_disc) ≥ (5 mm)²
```

Distance primitives at [brachiation_planner.py:890,908](brachiation_planner.py#L890).
All squared-distance forms: `d² ≥ d_min²` is smoother than `d ≥ d_min`
near the active boundary, so SLSQP behaves better.

### Two-stage solve

When `enable_collision=True`, the function calls itself recursively with
`enable_collision=False` and `pin_wp0=True` to a hand-picked non-colliding
WP 0 pose `(70°, 65°, −6°, 136°, 0°)`. The result becomes the initial guess
`x0` for the second stage, which then runs the full collision constraint
set. WP 0 stays pinned across both stages so the optimiser cannot drift the
start back into a colliding configuration. See
[brachiation_planner.py:1248-1283](brachiation_planner.py#L1248-L1283).

Stage 1 typically converges in 30–60 iterations. Stage 2 with collisions
enabled may finish at a `Positive directional derivative for linesearch`
warning — that's SLSQP saying the local minimum is on a constraint boundary
and no descent direction satisfies all constraints. The returned solution
is still feasible (verified empirically) and the catch lands on bar2
exactly.

### Solver call

```
scipy.optimize.minimize(
    cost, x0, method="SLSQP",
    bounds=bounds,
    constraints=constraints,        # list of {type: 'eq'/'ineq', fun, jac?}
    options={"maxiter": 500, "ftol": 1e-9, "disp": True},
)
```

Each constraint is a closure capturing its waypoint index and the
geometric data it needs. Cost finite-difference takes the bulk of the
solve time; analytic Jacobians could speed it up 10–20× if needed.


## MPC math — `QPReplanner`

The MPC is a quadratic program built once and re-solved each tick by updating
the linearised COM and goal rows in place. OSQP solves it to sub-millisecond.

### Decision variables

```
z = [ x ; s_lo ; s_hi ]
        x      ∈ ℝ^{nx}        joint state stacked, nx = 5·N
        s_lo   ∈ ℝ^{N-1}≥0     COM lower-band slack (skip WP 0)
        s_hi   ∈ ℝ^{N-1}≥0     COM upper-band slack
```

`x[k]` is the 5-vector for waypoint `k`: `[t1g, t2g, t1f, t2f, tail]` in
radians.

### Cost

OSQP form: minimise `½ zᵀ P z + qᵀ z`. Total cost:

```
J(z) = w_track   · ‖x − x_ref‖²
     + w_smooth  · ‖D x‖²              (D = first-difference operator)
     + w_slack   · (‖s_lo‖² + ‖s_hi‖²)
```

After expanding, the constant `‖x_ref‖²` term is dropped:

```
P = block_diag( 2·w_track·I_nx + 2·w_smooth·DᵀD ,   2·w_slack·I_2(N-1) )
q = [ −2·w_track · x_ref ; 0 ]
```

Setup at [qp_replanner.py:107](qp_replanner.py#L107). Re-solving at a new
`x_ref` only requires updating `q` — `P` stays.

### Constraints `l ≤ A z ≤ u`

The constraint matrix `A` is allocated once with all gradient slots set to
zero placeholders. Each replan rewrites just the COM and goal Jacobian
entries via OSQP's `update(Ax=...)` — the sparsity stays fixed.

**(1) Box bounds + WP0 pin** — rows `0 .. nx-1`:
```
l[j] = lo_j ,  u[j] = hi_j     (per-joint per-WP)
l[0..4] = u[0..4] = current_state_rad   (pins WP 0 to measured state)
```

**(2) Slack non-negativity** — rows `nx .. nx + 2(N-1) - 1`:
```
0 ≤ s_lo[k] ≤ +∞
0 ≤ s_hi[k] ≤ +∞
```

**(3) COM band, linearised at `x_ref`** — for each `k = 1 .. N-1`:

The COM eval_fn returns `(COM_x_ref_k, ∇COM_x_k)` from
`compute_com_world_x_and_grad`. Linearise:

```
COM_x(x[k]) ≈ COM_x_ref_k + ∇COM_x_k · (x[k] − x_ref[k])
            = ∇COM_x_k · x[k] + offset_k
where  offset_k = COM_x_ref_k − ∇COM_x_k · x_ref[k]
```

The hard COM band would be `lo ≤ ∇COM_x_k · x[k] + offset_k ≤ hi`. With
slacks added to soften it:

```
∇COM_x_k · x[k] + s_lo[k]  ≥  lo − offset_k          (lower row)
∇COM_x_k · x[k] − s_hi[k]  ≤  hi − offset_k          (upper row)
```

OSQP-form: each row encodes a single inequality on `[x; s_lo; s_hi]`. The
slack columns have constant ±1 coefficients (set up once); only the COM
gradient `∇COM_x_k` and the bounds `(lo − offset_k, hi − offset_k)` are
rewritten each replan. Setup of column structure at
[qp_replanner.py:135-167](qp_replanner.py#L135-L167); rewrite at
[qp_replanner.py:289-303](qp_replanner.py#L289-L303).

The COM grad and offset are evaluated at `x_ref[k]`, not at the current
solution `x[k]`. That's intentional — solving a sequential-QP where each
solve uses the previous `x_ref` as the linearisation point.

**(4) Goal equality, world frame, linearised at `x_ref[N-1]`**:

The goal eval_fn rotates the body-frame tip by the supplied IMU tilt around
bar1 to get world-frame tip:

```
tip_world(x, φ) = bar1 + R(−φ) · ( body_tip(x) − bar1 )
```

Linearise at `x_ref[N-1]`:

```
tip_world(x[N-1], φ) ≈ tip_world_ref + J · (x[N-1] − x_ref[N-1])
where J = ∂tip_world / ∂x  (2×5, finite-difference over 5 columns at fixed φ)
```

Set `tip_world(x[N-1], φ) = bar2`:

```
J · x[N-1] = bar2 − tip_world_ref + J · x_ref[N-1]   ←  rhs
```

Two equality rows (one per Cartesian component) on the last WP's columns.
Update at [qp_replanner.py:316-335](qp_replanner.py#L316-L335).

### Why COM stays in body frame on purpose

Two competing choices for the COM constraint at intermediate WPs:

| Choice | Constraint | Behaviour under disturbance tilt φ |
|---|---|---|
| Body-frame *(used)* | `body_COM_x(x[k]) = bar1_x` | commanded body wants to hang vertical; gravity restores tilt to 0 |
| World-frame | `world_COM_x(x[k], φ) = bar1_x` | commanded body **leans further** in the direction of φ; destabilising |

Body-frame is the stabilising choice. The goal constraint can stay in world
frame because the catch is a one-shot event where we *do* want the rendered
tip to land on bar2 regardless of body tilt at that instant.

### Solve loop

For each replan tick:

```
1. Pin WP 0 box rows to measured state.
2. For k = 1..N-1:
     evaluate (COM_x_ref_k, grad_k) at x_ref[k]
     write (l, u) entries with offset_k
     write grad_k into the COM rows of A
3. Evaluate (tip_world_ref, J) at x_ref[N-1] using current_tilt_rad
   write rhs into goal rows of (l, u)
   write J into goal rows of A
4. Update q = -2·w_track · x_ref
5. self.prob.update(q=q, l=l, u=u, Ax=new_Ax)
6. self.prob.solve()
7. Return x_out = z[:nx].reshape(N, 5)
```

Reported timing on the dev box: setup ~20 ms once per problem size, each
solve ~0.3–0.8 ms.


## How tilt flows from sensor to plan

```
imu.read()
   → imu_reading.tilt_rad
       → ReplanningController.step(reading)
           → self.current_imu_tilt_rad = reading.tilt_rad
               → _replan_from_current(...)
                   → _qp_replan(...)
                       → replanner.replan(x_ref, state_rad,
                                          current_tilt_rad=self.current_imu_tilt_rad)
                           → goal eval_fn(state, tilt) rotates body-frame tip
                             by R(−tilt) around bar1 before forming the
                             equality on bar2.
   → also goes through stabilizer.apply(q_ref, reading)
       → tail PD correction
```


## CLI quick reference

The default `python brachiation_planner.py --minimal` plans bar1→bar2 with all
the current defaults and pops up the matplotlib viz.

| Flag | Default | Effect |
|---|---|---|
| `--n-waypoints` | 12 | trajectory resolution |
| `--bar-spacing` | 12 | inches between bars |
| `--grip` | `left` | which hand grips bar1 |
| `--minimal` | off | use `plan_trajectory_minimal` |
| `--no-collision` | off | disable collision constraints |
| `--dip-depth` | 0.10 | m below bar at peak dip |
| `--dip-skew` | 4.0 | >1 pushes dip peak late |
| `--com-band` | 0.03 | direct-phase COM tolerance (m) |
| `--com-band-late` | 0.05 | dip/catch COM tolerance (m) |
| `--com-relax-t` | 0.70 | when band widening starts (0..1) |
| `--bar-clearance` | 0.02 | min y-distance below bar (m) |
| `--track-weight` | 20.0 | tracking cost on the U-target |
| `--tail-min` / `--tail-max` | −60 / 60 | tail range (deg) |

Other flags drive closed-loop and hardware modes (`--closed-loop`, `--imu`,
`--motors`, `--mpc-viz`, `--qp-mpc`, `--execute`, …) — see
[brachiation_planner.py:main()](brachiation_planner.py#L3380) for the full list.


## Known constraints / sharp edges

- The offline plan is **tilt-naive**. The QP MPC corrects for tilt at the
  goal; the inner stabilizer corrects for tilt at the body. There is no
  iterative tilt-aware self-consistency in the offline plan (could be added
  as an outer loop around the SLSQP solve if needed).
- The two-stage solve relies on a hand-tuned non-colliding warm start at
  WP0 (`(70, 65, -6, 136, 0)` deg). Different bar geometries may need a
  different warm start; see the search routine notes in commit history if
  this needs re-tuning.
- The COM constraint in the QP MPC is body-frame by design. If you ever need
  it in world frame, the `eval_fn` signature already supports it — just
  multiply the body-frame Jacobian by the rotation matrix.
- The `monotone_x` ineq prevents free-hand backtracking in the SLSQP solve.
  Disable it via `monotone_x=False` if you ever need motions that go back
  before forward.

## Saved viz outputs

`viz/` contains snapshots from various tuning runs:
- `u_trajectory_widths.png` — early no-collision run with widths overlay.
- `u_trajectory_collisions.png` — first version of the collision constraints.
- `u_trajectory_lean_collisions.png` — relaxed COM band, collisions on, but
  forearm-vs-forearm still binding.
- `u_trajectory_pinned.png` — current default: WP0 pinned to non-colliding
  pose, all collisions enforced, COM lean ±3–5 cm.
