"""Sub-millisecond MPC replanner via OSQP, with optional soft COM constraint.

Wraps a pre-allocated OSQP problem that tracks a reference trajectory
`x_ref` (from a slower full SLSQP plan) with smoothness, joint bounds,
a hard pin on WP0, and an optional **soft** linearized COM-x band
applied at WP 1..N-1.

State variables: stacked planner state in radians, length n_waypoints*5.

If `com_constraint` is supplied, the QP is augmented with two
non-negative slack variables per waypoint — `s_lo[k]` and `s_hi[k]` —
that absorb violations of the COM-x band:

    com_lo - s_lo[k] ≤ COM_x(x[k]) ≤ com_hi + s_hi[k]
    s_lo[k], s_hi[k] ≥ 0

The slacks are penalized quadratically in the cost via `slack_weight`.
Set `slack_weight` high (~1e3+) for near-hard behavior; low (~1) to let
the COM band freely deform when in tension with tracking.

Each replan() call:
  • Re-evaluates COM_x and ∂COM_x/∂state at every reference waypoint
    (finite-difference, 6×N compute_com_world calls).
  • Updates A's COM rows (via OSQP's update(Ax=...) — sparsity is fixed).
  • Updates l/u for the COM band and pins WP0 to the measured state.
  • Solves the QP. Returns only the joint state portion of the solution.
"""

from __future__ import annotations

import math
import time

import numpy as np
import osqp
import scipy.sparse as sp


STATE_DIM = 5


def _call_eval(fn, state_rad, tilt_rad):
    """Call an eval_fn that may take 1 or 2 positional args.

    Older eval_fns accept just `state_rad`; newer ones take
    `(state_rad, tilt_rad)` so they can rotate body-frame kinematics
    into world frame around bar1. This shim accepts both.
    """
    try:
        return fn(state_rad, tilt_rad)
    except TypeError:
        return fn(state_rad)


def _smoothness_dt_d(n_waypoints):
    """D such that D x stacks the consecutive differences x[k+1]-x[k]."""
    nx = n_waypoints * STATE_DIM
    n_diffs = (n_waypoints - 1) * STATE_DIM
    rows, cols, vals = [], [], []
    row = 0
    for k in range(n_waypoints - 1):
        for j in range(STATE_DIM):
            i_curr = (k + 1) * STATE_DIM + j
            i_prev = k * STATE_DIM + j
            rows.append(row); cols.append(i_curr); vals.append(1.0)
            rows.append(row); cols.append(i_prev); vals.append(-1.0)
            row += 1
    return sp.csc_matrix((vals, (rows, cols)), shape=(n_diffs, nx))


class QPReplanner:
    """Pre-allocated OSQP MPC replanner.

    Variable layout when `com_constraint` is set:
        z = [ x ; s_lo ; s_hi ]
            x       — joint state, length nx = n_waypoints*5
            s_lo    — lower-bound slack, length n_com = n_waypoints-1
            s_hi    — upper-bound slack, length n_com

    Constraint rows (in order):
        rows  0 .. nx-1                     : box bounds + WP0 pin on x
        rows  nx .. nx+n_slack-1            : s_lo, s_hi ≥ 0
        rows  nx+n_slack .. +n_com-1        : COM lower:  a·x + s_lo ≥ l_k
        rows  nx+n_slack+n_com .. +n_com-1  : COM upper:  a·x - s_hi ≤ u_k
    """

    def __init__(self, n_waypoints, per_wp_bounds_rad,
                 w_track=10.0, w_smooth=2.0,
                 com_constraint=None, slack_weight=1000.0,
                 goal_constraint=None,
                 eps_abs=1e-4, eps_rel=1e-4, max_iter=400):
        self.n_waypoints = int(n_waypoints)
        self.per_wp_bounds_rad = list(per_wp_bounds_rad)
        self.w_track = float(w_track)
        self.w_smooth = float(w_smooth)
        self.com_constraint = com_constraint
        self.slack_weight = float(slack_weight)
        self.goal_constraint = goal_constraint

        nx = self.n_waypoints * STATE_DIM
        n_com = (self.n_waypoints - 1) if com_constraint is not None else 0
        n_slack = 2 * n_com
        n_goal = 2 if goal_constraint is not None else 0  # tip_x, tip_y
        n_vars = nx + n_slack
        n_rows = nx + n_slack + 2 * n_com + n_goal

        self._nx = nx
        self._n_com = n_com
        self._n_slack = n_slack
        self._n_goal = n_goal
        self._n_vars = n_vars
        # Convenient row offsets:
        self._row_slack_start = nx
        self._row_com_lo_start = nx + n_slack
        self._row_com_hi_start = nx + n_slack + n_com
        self._row_goal_start   = nx + n_slack + 2 * n_com

        # ── Cost: 0.5 z^T P z + q^T z ────────────────────────────────
        # Per-||·||² weight needs the 2× because OSQP uses 0.5 z^T P z.
        D = _smoothness_dt_d(self.n_waypoints)
        P_xx = 2.0 * (self.w_track * sp.eye(nx, format='csc')
                      + self.w_smooth * (D.T @ D))
        if n_slack == 0:
            self.P = P_xx.tocsc()
        else:
            P_ss = 2.0 * self.slack_weight * sp.eye(n_slack, format='csc')
            self.P = sp.block_diag([P_xx, P_ss], format='csc')

        # ── A and Ax layout ──────────────────────────────────────────
        # We build CSC by columns to know the exact data-array layout so
        # OSQP's update(Ax=...) can rewrite the COM gradient values in
        # place every replan. Within each column we list rows in
        # ascending order (CSC requirement).
        if com_constraint is None and goal_constraint is None:
            self.A = sp.eye(nx, format='csc')
            self._com_grad_pos_in_Ax = None
            self._goal_grad_pos_in_Ax = None
        else:
            last_wp = self.n_waypoints - 1
            indptr = np.zeros(n_vars + 1, dtype=np.intc)
            indices_list = []
            data_list = []
            com_grad_pos_in_Ax = []
            goal_grad_pos_in_Ax = []
            # Joint-state columns 0..nx-1. For each column we list rows in
            # ascending order: box identity → COM rows (if WP>=1) → goal rows
            # (if last WP).
            for j in range(nx):
                wp_k = j // STATE_DIM
                # Box-bounds identity (always present).
                indices_list.append(j)
                data_list.append(1.0)
                if wp_k >= 1 and com_constraint is not None:
                    com_idx = wp_k - 1   # row index within COM blocks
                    # COM lower row (a·x + s_lo ≥ l)
                    indices_list.append(self._row_com_lo_start + com_idx)
                    data_list.append(0.0)              # gradient placeholder
                    com_grad_pos_in_Ax.append(len(data_list) - 1)
                    # COM upper row (a·x - s_hi ≤ u)
                    indices_list.append(self._row_com_hi_start + com_idx)
                    data_list.append(0.0)              # gradient placeholder
                    com_grad_pos_in_Ax.append(len(data_list) - 1)
                if wp_k == last_wp and goal_constraint is not None:
                    # Goal tip_x equality at last WP.
                    indices_list.append(self._row_goal_start + 0)
                    data_list.append(0.0)              # J row 0 placeholder
                    goal_grad_pos_in_Ax.append(len(data_list) - 1)
                    # Goal tip_y equality at last WP.
                    indices_list.append(self._row_goal_start + 1)
                    data_list.append(0.0)              # J row 1 placeholder
                    goal_grad_pos_in_Ax.append(len(data_list) - 1)
                indptr[j + 1] = len(indices_list)
            # s_lo columns: nx .. nx+n_com-1
            for i in range(n_com):
                # s_lo[i] ≥ 0 row: identity
                indices_list.append(self._row_slack_start + i)
                data_list.append(1.0)
                # COM-lower row: +s_lo coefficient
                indices_list.append(self._row_com_lo_start + i)
                data_list.append(1.0)
                indptr[nx + i + 1] = len(indices_list)
            # s_hi columns: nx+n_com .. nx+2*n_com-1
            for i in range(n_com):
                # s_hi[i] ≥ 0 row: identity (offset by n_com inside slack block)
                indices_list.append(self._row_slack_start + n_com + i)
                data_list.append(1.0)
                # COM-upper row: -s_hi coefficient
                indices_list.append(self._row_com_hi_start + i)
                data_list.append(-1.0)
                indptr[nx + n_com + i + 1] = len(indices_list)

            indices = np.array(indices_list, dtype=np.intc)
            data    = np.array(data_list,    dtype=float)
            self.A = sp.csc_matrix((data, indices, indptr),
                                   shape=(n_rows, n_vars))
            self._com_grad_pos_in_Ax = (np.array(com_grad_pos_in_Ax, dtype=int)
                                        if com_grad_pos_in_Ax else None)
            self._goal_grad_pos_in_Ax = (np.array(goal_grad_pos_in_Ax, dtype=int)
                                         if goal_grad_pos_in_Ax else None)
            # Sanity checks:
            if com_constraint is not None:
                assert len(self._com_grad_pos_in_Ax) == 2 * STATE_DIM * n_com
            if goal_constraint is not None:
                assert len(self._goal_grad_pos_in_Ax) == 2 * STATE_DIM

        # ── l, u templates ───────────────────────────────────────────
        self._l_template = np.empty(n_rows)
        self._u_template = np.empty(n_rows)
        # Box bounds (rows 0..nx-1)
        for k in range(self.n_waypoints):
            for j in range(STATE_DIM):
                lo, hi = self.per_wp_bounds_rad[j]
                self._l_template[k * STATE_DIM + j] = lo
                self._u_template[k * STATE_DIM + j] = hi
        # s_lo, s_hi ≥ 0 (rows nx..nx+n_slack-1)
        if n_slack > 0:
            self._l_template[nx : nx + n_slack] = 0.0
            self._u_template[nx : nx + n_slack] = np.inf
        # COM lower / upper rows: placeholder (rewritten per replan)
        if n_com > 0:
            self._l_template[self._row_com_lo_start :
                             self._row_com_lo_start + n_com] = -np.inf
            self._u_template[self._row_com_lo_start :
                             self._row_com_lo_start + n_com] = np.inf
            self._l_template[self._row_com_hi_start :
                             self._row_com_hi_start + n_com] = -np.inf
            self._u_template[self._row_com_hi_start :
                             self._row_com_hi_start + n_com] = np.inf
        # Goal tip equality rows: placeholder (rewritten per replan)
        if n_goal > 0:
            self._l_template[self._row_goal_start :
                             self._row_goal_start + n_goal] = -np.inf
            self._u_template[self._row_goal_start :
                             self._row_goal_start + n_goal] = np.inf

        # ── OSQP setup ───────────────────────────────────────────────
        self.prob = osqp.OSQP()
        self.prob.setup(
            self.P,
            np.zeros(n_vars),
            self.A,
            self._l_template.copy(),
            self._u_template.copy(),
            verbose=False,
            warm_starting=True,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
        )
        self._setup_done = True
        self._last_solve_ms = 0.0
        self._last_com_eval_ms = 0.0
        self._last_max_slack = 0.0

    def replan(self, x_ref, current_state_rad, current_tilt_rad=0.0):
        """Solve one QP. Returns ((n_waypoints, STATE_DIM), status_str).

        current_tilt_rad: latest IMU tilt (body angle from vertical, +CW).
            Passed through to the COM and goal eval_fn callbacks as the
            second positional argument so they can rotate body-frame
            kinematics into world frame around bar1. Eval_fns may ignore
            it (the COM eval_fn intentionally stays in body frame; the
            goal eval_fn rotates by it).
        """
        x_ref = np.asarray(x_ref, dtype=float)
        if x_ref.shape != (self.n_waypoints, STATE_DIM):
            raise ValueError(
                f"x_ref shape {x_ref.shape} != ({self.n_waypoints}, {STATE_DIM})")
        current = np.asarray(current_state_rad, dtype=float).flatten()
        if current.shape[0] != STATE_DIM:
            raise ValueError(
                f"current_state_rad must be length {STATE_DIM}; got {current.shape}")

        nx = self._nx
        n_vars = self._n_vars

        # q = [-2*w_track*x_ref, 0_for_slacks]  (slack penalty is purely quadratic)
        q = np.zeros(n_vars)
        q[:nx] = (-2.0 * self.w_track) * x_ref.flatten()

        # Pin WP0 (box bounds row 0..STATE_DIM-1).
        l = self._l_template.copy()
        u = self._u_template.copy()
        for j in range(STATE_DIM):
            lo = self._l_template[j]; hi = self._u_template[j]
            v  = float(np.clip(current[j], lo, hi))
            l[j] = v; u[j] = v

        update_kwargs = {"q": q, "l": l, "u": u}

        # If we have any nonlinear linearised constraints, we'll update Ax.
        need_Ax_update = (self.com_constraint is not None
                          or self.goal_constraint is not None)
        new_Ax = self.A.data.copy() if need_Ax_update else None

        if self.com_constraint is not None:
            eval_fn = self.com_constraint["eval_fn"]
            com_lo  = self.com_constraint["lo"]
            com_hi  = self.com_constraint["hi"]

            t_com = time.monotonic()
            grad_pos = self._com_grad_pos_in_Ax
            for kk in range(self._n_com):
                wp = kk + 1   # waypoint index (skip WP 0)
                com_x_ref_k, grad_k = _call_eval(eval_fn, x_ref[wp],
                                                  current_tilt_rad)
                grad_dot = float(np.dot(grad_k, x_ref[wp]))
                # Linearised band:
                #   grad·x ∈ [com_lo - com_x_ref_k + grad·x_ref,
                #             com_hi - com_x_ref_k + grad·x_ref]
                offset = com_x_ref_k - grad_dot
                # COM lower row: a·x + s_lo ≥ l   →   l ≤ a·x + s_lo ≤ +inf
                l[self._row_com_lo_start + kk] = com_lo - offset
                u[self._row_com_lo_start + kk] = np.inf
                # COM upper row: a·x - s_hi ≤ u   →   -inf ≤ a·x - s_hi ≤ u
                l[self._row_com_hi_start + kk] = -np.inf
                u[self._row_com_hi_start + kk] = com_hi - offset
                # Each WP's gradient is at 2*STATE_DIM positions in Ax
                # (one for the lower row, one for the upper row, per state j).
                base = kk * 2 * STATE_DIM
                for j in range(STATE_DIM):
                    pos_lo = grad_pos[base + 2 * j]      # COM-lower row entry
                    pos_hi = grad_pos[base + 2 * j + 1]  # COM-upper row entry
                    new_Ax[pos_lo] = float(grad_k[j])
                    new_Ax[pos_hi] = float(grad_k[j])
            self._last_com_eval_ms = (time.monotonic() - t_com) * 1000.0

        if self.goal_constraint is not None:
            target_xy = np.asarray(self.goal_constraint["target_xy"], dtype=float)
            eval_fn = self.goal_constraint["eval_fn"]
            x_last = x_ref[-1]
            tip_xy_ref, J = _call_eval(eval_fn, x_last, current_tilt_rad)
            tip_xy_ref = np.asarray(tip_xy_ref, dtype=float).flatten()
            J = np.asarray(J, dtype=float)
            # Linearise: tip(x_last) ≈ tip_ref + J · (x_last - x_ref_last)
            #   tip(x_last) = target  ⇒  J · x_last = target - tip_ref + J · x_ref_last
            J_dot_xref = J @ x_last
            rhs = target_xy - tip_xy_ref + J_dot_xref
            # Equality: l = u = rhs
            l[self._row_goal_start + 0] = rhs[0]
            u[self._row_goal_start + 0] = rhs[0]
            l[self._row_goal_start + 1] = rhs[1]
            u[self._row_goal_start + 1] = rhs[1]
            # Update the 2*STATE_DIM Jacobian entries in Ax.
            goal_pos = self._goal_grad_pos_in_Ax
            for j in range(STATE_DIM):
                # Order in goal_pos: per column j of last WP, two entries
                # (row tip_x then row tip_y). They were appended in that
                # order during construction.
                pos_x = goal_pos[2 * j + 0]
                pos_y = goal_pos[2 * j + 1]
                new_Ax[pos_x] = float(J[0, j])
                new_Ax[pos_y] = float(J[1, j])

        if need_Ax_update:
            update_kwargs["Ax"] = new_Ax

        self.prob.update(**update_kwargs)
        t0 = time.monotonic()
        res = self.prob.solve()
        self._last_solve_ms = (time.monotonic() - t0) * 1000.0

        status = res.info.status
        if status not in ("solved", "solved inaccurate"):
            return None, status

        z = res.x
        x_out = z[:nx].reshape(self.n_waypoints, STATE_DIM)
        if self._n_slack > 0:
            self._last_max_slack = float(np.max(z[nx:]))
        else:
            self._last_max_slack = 0.0
        return x_out, status

    @property
    def last_solve_ms(self):
        return self._last_solve_ms

    @property
    def last_com_eval_ms(self):
        return self._last_com_eval_ms

    @property
    def last_max_slack_m(self):
        """Max slack across all WPs after last solve, in metres of COM_x violation."""
        return self._last_max_slack
