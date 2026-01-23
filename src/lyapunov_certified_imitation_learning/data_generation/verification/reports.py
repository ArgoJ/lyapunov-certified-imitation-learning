from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class StabilityReport:
    method: str = ""
    is_stable: bool = False
    applicability: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

@dataclass
class LyapunovDecreaseReport(StabilityReport):
    method: str = "Lyapunov Decrease"
    min_alpha: Optional[float] = float("nan")
    max_violation: Optional[float] = float("nan")
    applicability: bool = True # Always applicable

@dataclass
class GrüneHorizonReport(StabilityReport):
    method: str = "Grüne Horizon Condition"
    gamma_estimate: float = float("nan")
    alpha_N_estimate: float = float("nan")
    required_horizon: float = float("nan")

@dataclass
class AlphaViolationStats:
    min_alpha: Optional[float] = None
    max_violation: Optional[float] = None
    min_residual: Optional[float] = None
    n_used: int = 0


@dataclass
class TerminalIngredientsReport(StabilityReport):
    """Report for a linear MPC terminal-ingredients certificate.

    This report is produced by
    `lyapunov_certified_imitation_learning.lyapunov_verification.mpc_certification.certify_linear_mpc_terminal_ingredients`.

    The certificate checks (when applicable):

    - Quadratic weights (approx.) definiteness: $Q\succeq 0$, $R\succ 0$, $P\succ 0$.
    - DARE consistency (via a residual eigenvalue bound).
    - Existence of a terminal ellipsoid $\{x: x^\top P x \le \rho\}$ contained in the
        terminal/state boxes and respecting the input box under the LQR terminal law
        $u = u^* - K(x-x^*)$.

    The scalar margins are conservative “slacks” computed from closed-form maxima
    over the ellipsoid, e.g. $\max |x_i| = \sqrt{\rho\,(P^{-1})_{ii}}$.
    """
    method: str = "Terminal Ingredients"
    applicability: bool = False

    dare_residual_max_abs_eig: float = float("nan")
    min_eig_Q: float = float("nan")
    min_eig_R: float = float("nan")
    min_eig_P: float = float("nan")

    invariance_margin: float = float("nan")
    input_margin: float = float("nan")
    state_margin: float = float("nan")

    K: Optional[Any] = None


@dataclass
class GruneNoTerminalCertificateReport(StabilityReport):
        """Report for a Grüne-style no-terminal MPC certificate (linear/quadratic).

        This report is produced by
        `lyapunov_certified_imitation_learning.lyapunov_verification.mpc_certification.certify_linear_mpc_grune_no_terminal`.

        It targets the common “no terminal cost / no terminal constraints” MPC setup
        for *linear* dynamics with *quadratic* stage cost.

        The certificate is theorem-backed **under its applicability conditions**:

        - Linear time-invariant dynamics: $x^+=Ax+Bu$.
        - Quadratic stage cost: $\ell(x,u)=x^\top Q x + u^\top R u$ with $Q\succ 0$, $R\succ 0$.
        - Terminal weight is zero (no terminal ingredients).
        - If state/input bounds exist, the report certifies an ellipsoidal region
            on which the unconstrained finite-horizon optimal sequence is feasible,
            hence the constrained MPC coincides with the unconstrained one there.

        The Grüne constant is computed from a horizon-independent upper bound on the
        value function using the infinite-horizon DARE solution $P_\infty$:

        $$\gamma = \lambda_{\max}\bigl(Q^{-1/2} P_\infty Q^{-1/2}\bigr).$$
        """

        method: str = "Grüne No-Terminal Certificate"
        applicability: bool = False

        gamma: float = float("nan")
        alpha_N: float = float("nan")
        required_horizon: float = float("nan")

        rho_max: float = float("nan")
        state_margin: float = float("nan")
        input_margin: float = float("nan")

        min_eig_Q: float = float("nan")
        min_eig_R: float = float("nan")
        min_eig_P0: float = float("nan")