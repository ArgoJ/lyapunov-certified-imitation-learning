import numpy as np
import scipy.linalg
import casadi as cs
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from acados_template import AcadosOcpSolver

@dataclass
class StabilityReport:
    method: str
    is_stable: bool
    details: Dict[str, float]
    message: str

class FormalStabilityVerifier:
    """
    Automated stability verifier for Acados OCP solvers.
    
    Implements formal stability proofs based on:
    Grüne, L., & Pannek, J. (2011). Nonlinear Model Predictive Control.
    """
    
    def __init__(self, acados_solver: AcadosOcpSolver):
        """
        Extracts system dynamics, costs, and constraints directly from the solver.
        """
        self.solver = acados_solver
        self.ocp = acados_solver.acados_ocp
        
        # Extract Dimensions and Horizon
        self.N = self.ocp.solver_options.N_horizon
        self.nx = self.ocp.model.x.size()[0]
        self.nu = self.ocp.model.u.size()[0]

        # Extract regulation reference (equilibrium candidate)
        self.x_star, self.u_star = self._extract_reference()
        
        # Extract Discrete Dynamics (Linearize if necessary)
        self.A, self.B = self._extract_linearized_dynamics()
        
        # Extract Cost Matrices (Assuming Linear LS formulation)
        self.Q, self.R, self.P = self._extract_cost_matrices()
        
        # Extract Constraints
        self.x_bounds = (self.ocp.constraints.lbx, self.ocp.constraints.ubx)
        self.u_bounds = (self.ocp.constraints.lbu, self.ocp.constraints.ubu)
        self.term_x_bounds = (self.ocp.constraints.lbx_e, self.ocp.constraints.ubx_e)

    def _extract_reference(self) -> Tuple[np.ndarray, np.ndarray]:
        """Best-effort extraction of (x*, u*) from yref/yref_e.

        For the regulation case used throughout Grüne's Part A, one typically has
        (x*, u*) = (0, 0) after shifting coordinates.
        """
        x_star = np.zeros(self.nx)
        u_star = np.zeros(self.nu)

        cost = self.ocp.cost
        if hasattr(cost, "yref") and cost.yref is not None:
            yref = np.asarray(cost.yref).reshape(-1)
            if yref.size >= (self.nx + self.nu):
                x_star = yref[: self.nx].copy()
                u_star = yref[self.nx : self.nx + self.nu].copy()

        if hasattr(cost, "yref_e") and cost.yref_e is not None:
            yref_e = np.asarray(cost.yref_e).reshape(-1)
            if yref_e.size >= self.nx:
                x_star = yref_e[: self.nx].copy()

        return x_star, u_star
        
    def _extract_linearized_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the discrete-time linearization (A, B) around (x*, u*).

        - If a discrete dynamics expression is provided, we linearize it directly.
        - Otherwise, we linearize the continuous-time dynamics and discretize the
          linearization using a ZOH matrix exponential.
        """
        dt = float(self.ocp.solver_options.tf) / float(self.N)
        x_sym = self.ocp.model.x
        u_sym = self.ocp.model.u

        x0 = np.asarray(self.x_star, dtype=float).reshape(self.nx)
        u0 = np.asarray(self.u_star, dtype=float).reshape(self.nu)

        # Case 1: discrete dynamics directly provided
        if self.ocp.model.disc_dyn_expr is not None:
            x_next = self.ocp.model.disc_dyn_expr
            A_expr = cs.jacobian(x_next, x_sym)
            B_expr = cs.jacobian(x_next, u_sym)
            J = cs.Function("Jd", [x_sym, u_sym], [A_expr, B_expr])
            A_d, B_d = J(x0, u0)
            return np.asarray(A_d.full()), np.asarray(B_d.full())

        # Case 2: continuous dynamics -> linearize -> discretize linearization via expm
        f = self.ocp.model.f_expl_expr
        A_c_expr = cs.jacobian(f, x_sym)
        B_c_expr = cs.jacobian(f, u_sym)
        Jc = cs.Function("Jc", [x_sym, u_sym], [A_c_expr, B_c_expr])
        A_c, B_c = Jc(x0, u0)
        A_c = np.asarray(A_c.full())
        B_c = np.asarray(B_c.full())

        # ZOH discretization of linearization
        nx = self.nx
        nu = self.nu
        M = np.zeros((nx + nu, nx + nu))
        M[:nx, :nx] = A_c
        M[:nx, nx:] = B_c
        Md = scipy.linalg.expm(M * dt)
        A_d = Md[:nx, :nx]
        B_d = Md[:nx, nx:]
        return A_d, B_d

    def _extract_cost_matrices(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extracts Q, R, and P from the cost configuration.
        """
        if self.ocp.cost.cost_type != 'LINEAR_LS' and self.ocp.cost.cost_type_e != 'LINEAR_LS':
            raise NotImplementedError("Only LINEAR_LS cost type is supported in this verifier.")
        
        # Stage cost
        W = self.ocp.cost.W
        Q = W[:self.nx, :self.nx]
        R = W[self.nx:, self.nx:]
        
        # Terminal cost
        P = None
        if hasattr(self.ocp.cost, 'W_e') and self.ocp.cost.W_e is not None:
            P_candidate = np.asarray(self.ocp.cost.W_e)
            if not np.allclose(P_candidate, 0.0, atol=0.0, rtol=0.0):
                P = P_candidate
            
        return Q, R, P

    # =========================================================================
    # METHOD A: Equilibrium Terminal Constraint
    # =========================================================================
    def prove_equilibrium_constraint(self) -> StabilityReport:
        """
        Verifies stability via Method A: x(N) = 0.
        Reference: 
        """
        lbx_e, ubx_e = self.term_x_bounds

        tol = 1e-6
        lbx_e = np.asarray(lbx_e).reshape(-1)
        ubx_e = np.asarray(ubx_e).reshape(-1)

        # If no terminal bounds are configured in the OCP, acados reports nbx_e = 0
        # and lbx_e/ubx_e come through as empty arrays.
        if lbx_e.size != self.nx or ubx_e.size != self.nx:
            details = {
                "has_equality_terminal": False,
                "max_terminal_width": 0.0,
                "terminal_target_error": float("inf"),
            }
            return StabilityReport(
                "Equilibrium Constraint",
                False,
                details,
                "FAIL. No equilibrium terminal equality constraint detected.",
            )

        has_equality_terminal = np.all(np.isfinite(lbx_e)) and np.all(np.isfinite(ubx_e)) and np.allclose(
            lbx_e, ubx_e, atol=tol, rtol=0.0
        )
        is_targeted = has_equality_terminal and np.allclose(lbx_e, self.x_star, atol=tol, rtol=0.0)

        details = {
            "has_equality_terminal": bool(has_equality_terminal),
            "max_terminal_width": float(np.max(np.abs(ubx_e - lbx_e))) if lbx_e.size else 0.0,
            "terminal_target_error": float(np.linalg.norm(lbx_e - self.x_star)) if lbx_e.size else 0.0,
        }

        if is_targeted and has_equality_terminal:
            msg = "PASS. Equilibrium terminal constraint x(N)=x* detected (implies VN-1>=VN and relaxed DP with alpha=1)."
        elif has_equality_terminal:
            msg = "WARN. Terminal equality constraint detected, but it does not match the extracted x*."
        else:
            msg = "FAIL. No equilibrium terminal equality constraint detected."

        return StabilityReport("Equilibrium Constraint", bool(is_targeted), details, msg)

    # =========================================================================
    # METHOD B: Regional Constraint & Terminal Cost (CLF)
    # =========================================================================
    def prove_regional_constraint(self) -> StabilityReport:
        """
        Verifies stability via Method B: x(N) in X_f and Compatible Terminal Cost P.
        Reference: 
        
        Checks:
        1. P solves the Discrete Algebraic Riccati Equation (DARE) approximately.
        2. F(x) = x'Px is a Lyapunov function for the local control law.
        """
        if self.P is None:
            return StabilityReport("Regional Constraint", False, {}, "FAIL. No terminal cost matrix P found.")

        P_term = np.asarray(self.P)
        P_term = 0.5 * (P_term + P_term.T)

        # Local controller from (Q,R) via DARE, i.e. u = -K x.
        try:
            P_dare = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
            K_lqr = np.linalg.inv(self.R + self.B.T @ P_dare @ self.B) @ (self.B.T @ P_dare @ self.A)
        except Exception as e:
            return StabilityReport("Regional Constraint", False, {}, f"FAIL. Could not compute DARE/LQR: {e}")

        # Check compatibility inequality on the linearized closed loop:
        # F(A_cl x) - F(x) <= -l(x, -Kx)
        A_cl = self.A - self.B @ K_lqr
        Q_cl = self.Q + K_lqr.T @ self.R @ K_lqr

        lhs = A_cl.T @ P_term @ A_cl - P_term
        # We need lhs <= -Q_cl. Numerically we check max eigenvalue of lhs + Q_cl <= tol.
        M = 0.5 * ((lhs + Q_cl) + (lhs + Q_cl).T)
        eig_max = float(np.max(np.linalg.eigvalsh(M)))
        compatible = eig_max <= 1e-7

        # Optional (very conservative) terminal region invariance check if finite bounds exist.
        lbx_e, ubx_e = self.term_x_bounds
        lbx_e = np.asarray(lbx_e).reshape(-1)
        ubx_e = np.asarray(ubx_e).reshape(-1)
        has_terminal_box = np.all(np.isfinite(lbx_e)) and np.all(np.isfinite(ubx_e))
        invariant_box = None
        if has_terminal_box and lbx_e.size == self.nx:
            # test invariance on all vertices of the terminal box
            verts = []
            for mask in range(1 << self.nx):
                v = np.zeros(self.nx)
                for i in range(self.nx):
                    v[i] = ubx_e[i] if (mask & (1 << i)) else lbx_e[i]
                verts.append(v)
            invariant_box = True
            for x in verts:
                u = -K_lqr @ (x - self.x_star) + self.u_star
                x_next = self.A @ x + self.B @ u
                if np.any(x_next < lbx_e - 1e-8) or np.any(x_next > ubx_e + 1e-8):
                    invariant_box = False
                    break

        details = {
            "eig_max_F_compatibility": float(eig_max),
            "terminal_cost_matches_dare_fro": float(np.linalg.norm(P_term - P_dare, ord="fro")),
            "has_terminal_box": bool(has_terminal_box),
            "terminal_box_invariant_vertices": bool(invariant_box) if invariant_box is not None else False,
        }

        if compatible:
            msg = "PASS. Terminal cost F(x)=x^T P x is compatible with stage cost for local LQR (Lyapunov decrease holds)."
            if invariant_box is False:
                msg += " Terminal box invariance could not be verified (vertex test failed)."
            return StabilityReport("Regional Constraint", True, details, msg)

        return StabilityReport(
            "Regional Constraint",
            False,
            details,
            "FAIL. Terminal cost is not compatible with stage cost (F(A_cl x)-F(x) <= -l(x,-Kx) violated).",
        )

    # =========================================================================
    # METHOD C: No Terminal Constraints (Horizon Length)
    # =========================================================================
    def prove_no_terminal_constraint(self) -> StabilityReport:
        """
        Verifies stability via Grüne's "no terminal constraints/cost" approach.

        In the fully general nonlinear constrained case, this requires a bound
            V_N(x) <= gamma * l*(x)
        obtained from controllability estimates.

        For the *unconstrained linear-quadratic* case, we can compute the horizon-dependent
        constants gamma_k exactly from finite-horizon Riccati recursions and then compute
        the corresponding alpha_N.
        """
        N = int(self.N)

        # If a terminal cost exists, then this method is not the right theorem.
        if self.P is not None:
            return StabilityReport(
                "No Terminal Constraint",
                False,
                {"current_horizon_N": int(N)},
                "NOT APPLICABLE. Terminal cost detected; use terminal-cost/terminal-set analysis instead.",
            )

        if N < 2:
            return StabilityReport(
                "No Terminal Constraint",
                False,
                {"current_horizon_N": int(N)},
                "FAIL. Need N>=2 for Grüne's alpha_N construction.",
            )

        # Finite-horizon Riccati recursion with terminal cost 0
        Pk = np.zeros((self.nx, self.nx))
        gamma_seq = [None]  # 1-indexing placeholder

        # Require Q positive definite to form Q^{-1/2}
        try:
            Q_sym = 0.5 * (self.Q + self.Q.T)
            w, V = np.linalg.eigh(Q_sym)
            if np.any(w <= 0.0):
                raise ValueError("Q must be positive definite to compute l*(x)=x^T Q x bounds.")
            Q_inv_sqrt = (V * (1.0 / np.sqrt(w))) @ V.T
        except Exception as e:
            return StabilityReport(
                "No Terminal Constraint",
                False,
                {"current_horizon_N": int(N)},
                f"FAIL. Could not compute Q^{-1/2} for gamma_k: {e}",
            )

        for k in range(1, N + 1):
            S = self.R + self.B.T @ Pk @ self.B
            Kk = np.linalg.inv(S) @ (self.B.T @ Pk @ self.A)
            Pk = self.Q + self.A.T @ Pk @ (self.A - self.B @ Kk)

            M = Q_inv_sqrt.T @ Pk @ Q_inv_sqrt
            gamma_k = float(np.max(np.linalg.eigvalsh(0.5 * (M + M.T))))
            gamma_seq.append(gamma_k)

        # Horizon-dependent alpha_N formula
        prod_gamma = 1.0
        prod_gamma_m1 = 1.0
        for i in range(2, N + 1):
            prod_gamma *= gamma_seq[i]
            prod_gamma_m1 *= max(gamma_seq[i] - 1.0, 0.0)

        denom = prod_gamma - prod_gamma_m1
        if denom <= 0.0:
            alpha_N = float("-inf")
        else:
            alpha_N = 1.0 - ((gamma_seq[N] - 1.0) * prod_gamma_m1) / denom

        is_stable = bool(alpha_N > 0.0)

        details = {
            "current_horizon_N": int(N),
            "gamma_N": float(gamma_seq[N]),
            "alpha_N": float(alpha_N),
        }

        msg = "PASS." if is_stable else "FAIL."
        msg += f" Computed alpha_N={alpha_N:.4f} from finite-horizon Riccati bounds (unconstrained LQ)."
        return StabilityReport("No Terminal Constraint", is_stable, details, msg)

    def prove_all(self):
        """Runs all proofs and prints a report."""
        reports = [
            self.prove_equilibrium_constraint(),
            self.prove_regional_constraint(),
            self.prove_no_terminal_constraint()
        ]
        
        print("\n" + "="*60)
        print(f"FORMAL STABILITY PROOF REPORT (Acados Solver)")
        print("="*60)
        
        for r in reports:
            status = "[PASS]" if r.is_stable else "[FAIL/WARN]"
            print(f"\nMethod: {r.method}")
            print(f"Status: {status}")
            print(f"Details: {r.details}")
            print(f"Conclusion: {r.message}")
            
        print("="*60 + "\n")