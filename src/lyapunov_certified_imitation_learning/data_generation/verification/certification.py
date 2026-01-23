from __future__ import annotations

import numpy as np

from typing import Optional, Tuple

try:
    from acados_template import AcadosOcpSolver
except Exception:
    raise ImportError(
        "StabilityVerifier requires `acados_template` (acados Python interface). "
        "Install it and ensure it is importable."
    )

from .reports import TerminalIngredientsReport, GruneNoTerminalCertificateReport
from .gruene import grune_required_horizon_and_alpha
from .reports import StabilityReport
from ..extractor import MPCConfigExtractor, LinearSystemExtractor
from ...utils.linalg import as_mat, as_vec, sym



def _finite_horizon_riccati_no_terminal(
    *,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    N: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute P0 and time-varying gains K_k for terminal weight 0.

    Uses the discrete-time finite-horizon Riccati recursion with terminal weight
    P_N = 0.

    Returns
    -------
    P0 : np.ndarray
        Quadratic value function matrix for V_N(x) = x^T P0 x.
    K_seq : list[np.ndarray]
        List of gains K_k (k=0..N-1), where u_k = -K_k x_k.
    """
    if N < 1:
        raise ValueError("N must be >= 1")

    nx = A.shape[0]
    nu = B.shape[1]

    P_next = np.zeros((nx, nx), dtype=float)
    K_seq: list[np.ndarray] = []
    for _ in range(N):
        # Backward recursion; append and reverse later
        S = sym(R + B.T @ P_next @ B)
        K = np.linalg.solve(S, B.T @ P_next @ A)
        P = sym(Q + A.T @ P_next @ A - A.T @ P_next @ B @ K)
        K_seq.append(K)
        P_next = P
    K_seq.reverse()
    P0 = P_next
    return P0, K_seq


def certify_linear_mpc_grune_no_terminal(
    *,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    N: int,
    x_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    u_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    x_star: Optional[np.ndarray] = None,
    u_star: Optional[np.ndarray] = None,
    atol: float = 1e-10,
) -> GruneNoTerminalCertificateReport:
    """Certify stability for no-terminal linear/quadratic MPC via Grüne's theorem.

    This is intended for OCPs with **no terminal cost** and **no terminal bounds**.
    It computes the exact unconstrained finite-horizon value function

    $$V_N(x)=x^\top P_0 x$$

    via Riccati recursion with terminal weight $P_N=0$, then computes

    $$\gamma = \lambda_{\max}(Q^{-1/2} P_0 Q^{-1/2})$$

    and the sufficient horizon length and decrease factor from Grüne's formulas.

    If state/input bounds are provided, the function additionally certifies a
    (possibly local) ellipsoidal region $\{x: x^\top P_0 x \le \rho\}$ on which
    the unconstrained optimal sequence is feasible (constraints inactive).

    Notes
    -----
    This certificate assumes stage cost of the form $x^\top Q x + u^\top R u$
    in shifted coordinates around $(x^*,u^*)$.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays")

    nx = A.shape[0]
    if A.shape[1] != nx:
        raise ValueError(f"A must be square, got {A.shape}.")
    if B.shape[0] != nx:
        raise ValueError(f"B must have shape ({nx}, nu), got {B.shape}.")
    nu = B.shape[1]

    Q = sym(as_mat(Q, (nx, nx), "Q"))
    R = sym(as_mat(R, (nu, nu), "R"))
    if x_star is None:
        x_star = np.zeros((nx,), dtype=float)
    else:
        x_star = as_vec(x_star, nx, "x_star")
    if u_star is None:
        u_star = np.zeros((nu,), dtype=float)
    else:
        u_star = as_vec(u_star, nu, "u_star")

    rep = GruneNoTerminalCertificateReport(
        applicability=False,
        is_stable=False,
        message="Not evaluated.",
        details={},
    )

    # Basic definiteness checks
    eig_Q = np.linalg.eigvalsh(Q) if Q.size else np.array([0.0])
    eig_R = np.linalg.eigvalsh(R) if R.size else np.array([0.0])
    rep.min_eig_Q = float(np.min(eig_Q))
    rep.min_eig_R = float(np.min(eig_R))
    if rep.min_eig_Q <= 1e-12:
        rep.applicability = False
        rep.message = f"Not applicable: Q must be PD for Grüne gamma (min_eig_Q={rep.min_eig_Q:.3e})."
        return rep
    if rep.min_eig_R <= 1e-12:
        rep.applicability = False
        rep.message = f"Not applicable: R must be PD (min_eig_R={rep.min_eig_R:.3e})."
        return rep

    # Symmetric bounds around equilibrium (shifted coordinates)
    x_hw = None
    u_hw = None
    if x_bounds is not None:
        x_sym, x_hw = _is_symmetric_bounds(x_bounds[0], x_bounds[1], x_star, atol)
        if not x_sym:
            rep.message = "Not applicable: x_bounds must be symmetric around x_star."
            return rep
    if u_bounds is not None:
        u_sym, u_hw = _is_symmetric_bounds(u_bounds[0], u_bounds[1], u_star, atol)
        if not u_sym:
            rep.message = "Not applicable: u_bounds must be symmetric around u_star."
            return rep

    # Compute P0 and K-sequence for terminal weight 0
    try:
        P0, K_seq = _finite_horizon_riccati_no_terminal(A=A, B=B, Q=Q, R=R, N=int(N))
    except np.linalg.LinAlgError:
        rep.message = "FAIL: Riccati recursion failed (singular solve)."
        rep.applicability = True
        return rep

    rep.min_eig_P0 = float(np.min(np.linalg.eigvalsh(P0)))
    if rep.min_eig_P0 <= 1e-12:
        rep.message = f"FAIL: P0 not PD (min_eig_P0={rep.min_eig_P0:.3e})."
        rep.applicability = True
        return rep

    # Grüne's theorem uses a *horizon-independent* cost controllability constant gamma.
    # For linear/quadratic regulation, we can obtain such a bound from the infinite-horizon
    # DARE solution P_inf (LQR): V_N(x) <= V_inf(x) = x^T P_inf x for all N.
    try:
        try:
            from scipy.linalg import solve_discrete_are
        except Exception as e:  # pragma: no cover
            raise ImportError("scipy is required to compute the DARE-based Grüne gamma.") from e

        P_inf = solve_discrete_are(A, B, Q, R)
        P_inf = sym(np.asarray(P_inf, dtype=float))

        L = np.linalg.cholesky(Q)
        tmp = np.linalg.solve(L, P_inf)
        M = np.linalg.solve(L, tmp.T).T
        M = sym(M)
        rep.gamma = float(np.max(np.linalg.eigvalsh(M)))
    except np.linalg.LinAlgError:
        rep.message = "FAIL: could not compute gamma (Cholesky/solve failed)."
        rep.applicability = True
        return rep

    N_required, alpha_N = grune_required_horizon_and_alpha(gamma=rep.gamma, N=int(N))
    rep.required_horizon = float(N_required)
    rep.alpha_N = float(alpha_N)

    # Certify a region where constraints are inactive for the unconstrained optimal sequence.
    # Use ellipsoid induced by P0: {x: x^T P0 x <= rho}.
    try:
        P0_inv = sym(np.linalg.inv(P0))
    except np.linalg.LinAlgError:
        rep.message = "FAIL: P0 is singular; cannot bound constraints on ellipsoid."
        rep.applicability = True
        return rep

    rho_candidates: list[float] = []

    # Precompute linear maps x_k = M_k x0 and u_k = U_k x0.
    M_k = np.eye(nx)
    for k in range(int(N)):
        Kk = np.asarray(K_seq[k], dtype=float)
        U_k = -Kk @ M_k

        if x_hw is not None:
            for i in range(nx):
                a = M_k[i : i + 1, :]
                denom = float((a @ P0_inv @ a.T).item())
                if denom > 0.0:
                    rho_candidates.append(float((x_hw[i] ** 2) / denom))

        if u_hw is not None:
            for j in range(nu):
                a = U_k[j : j + 1, :]
                denom = float((a @ P0_inv @ a.T).item())
                if denom > 0.0:
                    rho_candidates.append(float((u_hw[j] ** 2) / denom))

        Acl_k = A - B @ Kk
        M_k = Acl_k @ M_k

    if rho_candidates:
        rep.rho_max = float(np.min(rho_candidates))
    else:
        rep.rho_max = float("inf")

    if (not np.isfinite(rep.rho_max)) and rho_candidates:
        rep.message = "FAIL: non-finite rho_max from constraint sizing."
        rep.applicability = True
        return rep
    if rho_candidates and rep.rho_max <= 0.0:
        rep.message = "FAIL: could not find rho>0 ensuring constraints inactivity on an ellipsoid."
        rep.applicability = True
        return rep

    # Compute conservative margins at rho_max (if finite constraints exist)
    rep.state_margin = float("inf") if x_hw is not None else float("nan")
    rep.input_margin = float("inf") if u_hw is not None else float("nan")

    if rho_candidates and np.isfinite(rep.rho_max):
        min_state_slack = float("inf")
        min_input_slack = float("inf")

        M_k = np.eye(nx)
        for k in range(int(N)):
            Kk = np.asarray(K_seq[k], dtype=float)
            U_k = -Kk @ M_k

            if x_hw is not None:
                for i in range(nx):
                    a = M_k[i : i + 1, :]
                    denom = float((a @ P0_inv @ a.T).item())
                    peak = float(np.sqrt(max(0.0, rep.rho_max * denom)))
                    min_state_slack = min(min_state_slack, float(x_hw[i] - peak))

            if u_hw is not None:
                for j in range(nu):
                    a = U_k[j : j + 1, :]
                    denom = float((a @ P0_inv @ a.T).item())
                    peak = float(np.sqrt(max(0.0, rep.rho_max * denom)))
                    min_input_slack = min(min_input_slack, float(u_hw[j] - peak))

            Acl_k = A - B @ Kk
            M_k = Acl_k @ M_k

        if x_hw is not None:
            rep.state_margin = float(min_state_slack)
        if u_hw is not None:
            rep.input_margin = float(min_input_slack)

    rep.applicability = True

    stable = bool((int(N) >= int(N_required)) and np.isfinite(rep.alpha_N) and (rep.alpha_N > 0.0))
    if rho_candidates and np.isfinite(rep.rho_max):
        if x_hw is not None:
            stable = stable and (rep.state_margin >= -atol)
        if u_hw is not None:
            stable = stable and (rep.input_margin >= -atol)

    rep.is_stable = stable
    if rep.is_stable:
        rep.message = (
            "PASS: certified by Grüne no-terminal conditions "
            f"(gamma={rep.gamma:.3e}, alpha_N={rep.alpha_N:.3e}, N_required={int(N_required)})."
        )
    else:
        rep.message = (
            "FAIL: Grüne no-terminal conditions not met "
            f"(gamma={rep.gamma:.3e}, alpha_N={rep.alpha_N:.3e}, N_required={int(N_required)})."
        )

    rep.details = {
        "P0": P0,
        "P_inf": P_inf,
        "K0": np.asarray(K_seq[0]) if K_seq else None,
        "K_sequence_length": len(K_seq),
    }

    return rep


def _is_symmetric_bounds(
    lb: np.ndarray, ub: np.ndarray, center: np.ndarray, atol: float
) -> tuple[bool, np.ndarray]:
    """Return (is_symmetric, halfwidth) for bounds around a given center."""
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    center = np.asarray(center, dtype=float).reshape(-1)
    if lb.shape != ub.shape or lb.shape != center.shape:
        return False, np.array([])

    hw1 = ub - center
    hw2 = center - lb
    if np.any(hw1 < -atol) or np.any(hw2 < -atol):
        return False, np.array([])

    if not np.allclose(hw1, hw2, atol=atol, rtol=0.0):
        return False, np.array([])

    return True, np.maximum(hw1, 0.0)


def certify_linear_mpc_terminal_ingredients(
    *,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    P: Optional[np.ndarray],
    x_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    u_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    terminal_x_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    x_star: Optional[np.ndarray] = None,
    u_star: Optional[np.ndarray] = None,
    atol: float = 1e-10,
) -> TerminalIngredientsReport:
    """Certify linear MPC stability assumptions via terminal ingredients.

    The certificate targets the classical terminal-ingredients assumptions for
    linear/quadratic MPC around an equilibrium $(x^*,u^*)$ (in shifted coordinates
    $\tilde x=x-x^*$, $\tilde u=u-u^*$):

    - Dynamics: $\tilde x^+ = A\tilde x + B\tilde u$
    - Stage cost: $\ell(\tilde x,\tilde u)=\tilde x^\top Q\tilde x+\tilde u^\top R\tilde u$
    - Terminal cost: $V_f(\tilde x)=\tilde x^\top P\tilde x$

    Terminal controller and DARE
    ----------------------------
    This function uses the standard LQR gain for the **terminal law**

    $$\tilde u = -K\tilde x,\qquad K=(R+B^\top P B)^{-1}B^\top P A.$$

    It checks the discrete algebraic Riccati equation (DARE) residual

    $$P = A^\top P A - A^\top P B (R+B^\top P B)^{-1} B^\top P A + Q.$$

    Ellipsoidal terminal set sizing
    -------------------------------
    Instead of attempting to certify invariance of an axis-aligned terminal box
    directly (often very conservative), we certify the existence of $\rho>0$ such
    that the **ellipsoidal terminal set**

    $$\mathcal X_f(\rho)=\{\tilde x: \tilde x^\top P\tilde x\le \rho\}$$

    is contained in the OCP's terminal/state boxes and respects input bounds under
    $\tilde u=-K\tilde x$.

    The worst-case bound used for each coordinate is

    $$\max_{\tilde x^\top P\tilde x \le \rho} |a^\top\tilde x| = \sqrt{\rho\,a^\top P^{-1} a}.$$

    Applicability
    -------------
    Only axis-aligned **symmetric** box bounds are supported (around $(x^*,u^*)$).
    If any required bound is missing or non-symmetric, the report is marked as
    `applicability=False`.

    Parameters
    ----------
    A, B : np.ndarray
        Discrete-time dynamics matrices.
    Q, R : np.ndarray
        Quadratic stage cost weights.
    P : np.ndarray | None
        Terminal cost weight. If None, attempts to compute it via `scipy.linalg.solve_discrete_are`.
    x_bounds, u_bounds, terminal_x_bounds : tuple[np.ndarray, np.ndarray] | None
        Lower/upper bounds for state, input, and terminal state.
    x_star, u_star : np.ndarray | None
        Equilibrium (center of symmetry for bounds). Defaults to zero.
    atol : float
        Numerical tolerance for symmetry and inequality checks.

    Returns
    -------
    TerminalIngredientsReport
        `is_stable=True` means **certified** (within the stated assumptions).
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays")

    nx = A.shape[0]
    if A.shape[1] != nx:
        raise ValueError(f"A must be square, got {A.shape}.")
    if B.shape[0] != nx:
        raise ValueError(f"B must have shape ({nx}, nu), got {B.shape}.")
    nu = B.shape[1]

    Q = sym(as_mat(Q, (nx, nx), "Q"))
    R = sym(as_mat(R, (nu, nu), "R"))

    if x_star is None:
        x_star = np.zeros((nx,), dtype=float)
    else:
        x_star = as_vec(x_star, nx, "x_star")

    if u_star is None:
        u_star = np.zeros((nu,), dtype=float)
    else:
        u_star = as_vec(u_star, nu, "u_star")

    report = TerminalIngredientsReport(
        applicability=False,
        is_stable=False,
        message="Not evaluated.",
        details={},
    )

    # Applicability requires terminal bounds and (if provided) symmetric x/u bounds.
    if terminal_x_bounds is None:
        report.message = "Not applicable: missing terminal_x_bounds."
        return report

    term_sym, h = _is_symmetric_bounds(terminal_x_bounds[0], terminal_x_bounds[1], x_star, atol)
    if not term_sym:
        report.message = "Not applicable: terminal_x_bounds must be symmetric around x_star."
        return report

    if x_bounds is not None:
        x_sym, _ = _is_symmetric_bounds(x_bounds[0], x_bounds[1], x_star, atol)
        if not x_sym:
            report.message = "Not applicable: x_bounds must be symmetric around x_star."
            return report

    if u_bounds is not None:
        u_sym, u_max = _is_symmetric_bounds(u_bounds[0], u_bounds[1], u_star, atol)
        if not u_sym:
            report.message = "Not applicable: u_bounds must be symmetric around u_star."
            return report
    else:
        u_max = None

    # Compute P if not provided.
    if P is None:
        try:
            from scipy.linalg import solve_discrete_are
        except Exception as e:  # pragma: no cover
            raise ImportError("P is None and scipy is required to solve the DARE.") from e
        P = solve_discrete_are(A, B, Q, R)

    P = sym(as_mat(P, (nx, nx), "P"))

    # Basic definiteness checks
    eig_Q = np.linalg.eigvalsh(Q) if Q.size else np.array([0.0])
    eig_R = np.linalg.eigvalsh(R) if R.size else np.array([0.0])
    eig_P = np.linalg.eigvalsh(P) if P.size else np.array([0.0])

    report.min_eig_Q = float(np.min(eig_Q))
    report.min_eig_R = float(np.min(eig_R))
    report.min_eig_P = float(np.min(eig_P))

    if report.min_eig_Q < -1e-8:
        report.message = f"FAIL: Q not PSD (min_eig_Q={report.min_eig_Q:.3e})."
        report.applicability = True
        return report

    if report.min_eig_R <= 1e-12:
        report.message = f"FAIL: R not PD (min_eig_R={report.min_eig_R:.3e})."
        report.applicability = True
        return report

    if report.min_eig_P <= 1e-12:
        report.message = f"FAIL: P not PD (min_eig_P={report.min_eig_P:.3e})."
        report.applicability = True
        return report

    # LQR gain and DARE residual check
    S = R + B.T @ P @ B
    S = sym(S)
    try:
        K = np.linalg.solve(S, B.T @ P @ A)
    except np.linalg.LinAlgError as e:
        report.message = "FAIL: (R+B^T P B) is singular."
        report.applicability = True
        return report

    # DARE residual: P = A^T P A - A^T P B S^{-1} B^T P A + Q
    M = np.linalg.solve(S, B.T @ P @ A)
    dare_rhs = A.T @ P @ A - A.T @ P @ B @ M + Q
    dare_resid = sym(P - dare_rhs)
    report.dare_residual_max_abs_eig = float(np.max(np.abs(np.linalg.eigvalsh(dare_resid))))
    if report.dare_residual_max_abs_eig > 1e-8:
        report.message = f"FAIL: DARE residual too large (max_abs_eig={report.dare_residual_max_abs_eig:.3e})."
        report.applicability = True
        report.K = K
        return report

    # --- Terminal set sizing: ellipsoid level set of V_f(x)=x^T P x
    # We certify the existence of rho>0 such that
    #   X_f := {x: x^T P x <= rho}
    # is contained in the (box) terminal bounds and respects state/input constraints.
    # Positive invariance follows from the DARE-based decrease, hence no separate
    # invariance check is needed.

    try:
        P_inv = np.linalg.inv(P)
    except np.linalg.LinAlgError:
        report.message = "FAIL: P is singular; cannot size ellipsoidal terminal set."
        report.applicability = True
        report.K = K
        return report

    P_inv = sym(P_inv)

    # Terminal box inclusion: max |x_i| over ellipsoid is sqrt(rho * (P^{-1})_{ii}).
    diag_Pinv = np.diag(P_inv)
    if np.any(diag_Pinv <= 0.0):
        report.message = "FAIL: P^{-1} has non-positive diagonal entries (numerical issue)."
        report.applicability = True
        report.K = K
        return report

    rho_candidates: list[float] = []

    rho_term = float(np.min((h ** 2) / diag_Pinv))
    rho_candidates.append(rho_term)

    # State constraints (optional): ensure ellipsoid is within global state box.
    if x_bounds is not None:
        x_sym, x_hw = _is_symmetric_bounds(x_bounds[0], x_bounds[1], x_star, atol)
        if not x_sym:
            report.message = "Not applicable: x_bounds must be symmetric around x_star."
            return report
        rho_state = float(np.min((x_hw ** 2) / diag_Pinv))
        rho_candidates.append(rho_state)

    # Input constraints (optional): u = u* - K (x-x*), so tilde u = -K tilde x.
    if u_max is not None:
        for k in range(nu):
            a = np.asarray(K[k, :], dtype=float).reshape(1, -1)
            denom = float((a @ P_inv @ a.T).item())
            if denom <= 0.0:
                # If denom == 0, this row is (numerically) zero -> no restriction.
                continue
            rho_u_k = float((u_max[k] ** 2) / denom)
            rho_candidates.append(rho_u_k)

    rho_max = float(np.min(rho_candidates)) if rho_candidates else 0.0
    if not np.isfinite(rho_max) or rho_max <= 0.0:
        report.message = "FAIL: could not find rho>0 for an admissible terminal ellipsoid."
        report.applicability = True
        report.K = K
        return report

    # Convert rho into conservative slack margins.
    x_peak = np.sqrt(rho_max * diag_Pinv)
    report.invariance_margin = float(np.min(h - x_peak))  # terminal inclusion margin

    if u_max is not None:
        u_peak = np.zeros((nu,), dtype=float)
        for k in range(nu):
            a = np.asarray(K[k, :], dtype=float).reshape(1, -1)
            denom = float((a @ P_inv @ a.T).item())
            u_peak[k] = float(np.sqrt(max(0.0, rho_max * denom)))
        report.input_margin = float(np.min(u_max - u_peak))
    else:
        report.input_margin = float("nan")

    if x_bounds is not None:
        report.state_margin = float(np.min(x_hw - x_peak))
    else:
        report.state_margin = float("nan")

    report.applicability = True
    report.K = K

    stable = True
    stable = stable and (report.invariance_margin >= -atol)
    if u_max is not None:
        stable = stable and (report.input_margin >= -atol)
    if x_bounds is not None:
        stable = stable and (report.state_margin >= -atol)

    report.is_stable = bool(stable)
    if report.is_stable:
        report.message = (
            "PASS: terminal ingredients satisfied via ellipsoidal terminal set "
            f"(terminal_margin={report.invariance_margin:.3e}, input_margin={report.input_margin:.3e})."
        )
    else:
        report.message = (
            "FAIL: terminal ingredients not satisfied via ellipsoidal terminal set "
            f"(terminal_margin={report.invariance_margin:.3e}, input_margin={report.input_margin:.3e})."
        )

    return report


class StabilityCertifier:
    def __init__(self, solver: AcadosOcpSolver):
        self.cfg = MPCConfigExtractor.get_cfg(solver)
        self.sys = LinearSystemExtractor.get_system(solver)
        
    
    @classmethod
    def certify(
        cls,
        solver: AcadosOcpSolver,
        gd_atol: float = 1e-10,
    ) -> StabilityReport:
        """Real MPC stability certification via theorem-backed conditions.

        This method attempts to certify asymptotic stability by checking:

        1) Terminal-ingredients conditions (regional certificate), or
        2) Grüne no-terminal conditions (for OCPs without terminal ingredients),
           together with a certified region on which constraints are inactive.

        The returned `is_stable` means **certified**.
        """
        cer = cls(solver)
        # 1) Theorem-backed certificate (does not depend on dataset sampling).
        if cer.sys.gd is not None and np.max(np.abs(np.asarray(cer.sys.gd).reshape(-1))) > float(gd_atol):
            terminal_rep = TerminalIngredientsReport(
                applicability=False,
                is_stable=False,
                message=f"Not applicable: linearization offset gd not near zero (||gd||_inf>{gd_atol:.1e}).",
            )
        elif cer.cfg.Qf is None:
            terminal_rep = TerminalIngredientsReport(
                applicability=False,
                is_stable=False,
                message="Not applicable: OCP has no terminal cost matrix P.",
            )
        else:
            terminal_rep = certify_linear_mpc_terminal_ingredients(
                A=cer.sys.A,
                B=cer.sys.B,
                Q=cer.cfg.Q,
                R=cer.cfg.R,
                P=cer.cfg.Qf,
                x_bounds=cer.cfg.state_bounds,
                u_bounds=cer.cfg.input_bounds,
                terminal_x_bounds=cer.cfg.terminal_state_bounds,
                x_star=cer.cfg.x_ref,
                u_star=cer.cfg.u_ref,
            )

        # Optional: Grüne no-terminal certificate for problems without terminal ingredients.
        grune_cert_rep = GruneNoTerminalCertificateReport(
            applicability=False,
            is_stable=False,
            message="Not evaluated.",
        )
        if (cer.cfg.Qf is None) and (cer.cfg.terminal_state_bounds is None):
            try:
                grune_cert_rep = certify_linear_mpc_grune_no_terminal(
                    A=cer.sys.A,
                    B=cer.sys.B,
                    Q=cer.cfg.Q,
                    R=cer.cfg.R,
                    N=int(cer.cfg.N),
                    x_bounds=cer.cfg.state_bounds,
                    u_bounds=cer.cfg.input_bounds,
                    x_star=cer.cfg.x_ref,
                    u_star=cer.cfg.u_ref,
                )
            except Exception as e:
                grune_cert_rep = GruneNoTerminalCertificateReport(
                    applicability=False,
                    is_stable=False,
                    message=f"Not applicable: Grüne no-terminal certificate failed with error: {type(e).__name__}: {e}",
                )

        certified_terminal = bool(terminal_rep.applicability and terminal_rep.is_stable)
        certified_grune = bool(grune_cert_rep.applicability and grune_cert_rep.is_stable)

        if certified_terminal:
            msg = f"PASS. Certified by terminal ingredients. {terminal_rep.message}"
        elif certified_grune:
            msg = f"PASS. Certified by Grüne no-terminal conditions. {grune_cert_rep.message}"
        else:
            if terminal_rep.applicability:
                msg = f"FAIL. Not certified by terminal ingredients: {terminal_rep.message}"
            elif grune_cert_rep.applicability:
                msg = f"FAIL. Not certified by Grüne no-terminal conditions: {grune_cert_rep.message}"
            else:
                msg = (
                    "FAIL. No applicable certification method. "
                    f"terminal='{terminal_rep.message}', grune_no_terminal='{grune_cert_rep.message}'"
                )

        return StabilityReport(
            method="Certification",
            is_stable=bool(certified_terminal or certified_grune),
            details={
                "terminal_ingredients_report": terminal_rep,
                "grune_no_terminal_report": grune_cert_rep,
            },
            message=msg,
        )
    
