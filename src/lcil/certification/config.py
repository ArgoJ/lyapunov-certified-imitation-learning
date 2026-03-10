from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..lyapunov_learning.config import LyapunovTrainingConfig


@dataclass(frozen=True)
class LyapunovCertificationConfig:
    """Configuration for Lyapunov certification only.

    Parameters
    ----------
    state_dim : int
        Dimension of the system state.
    state_bounds : Sequence[float]
        Per-dimension absolute bounds defining the certification region.
    kappa : float
        Exponential decay factor in the Lyapunov decrease condition.
    invariance_weight : float
        Weight of the set-invariance penalty term.
    rho_min : float
        Minimum admissible sublevel value.
    cert_step : float
        Grid step size for certification region decomposition.
    cert_origin_exclusion : float | Sequence[float] | None
        Radius around the origin to skip during certification. If a sequence is
        provided, it is interpreted per state dimension.
    cert_rho_scaling : float
        Multiplicative factor used in rho scaling before bisection.
    cert_bisection_tol : float
        Tolerance used in rho bisection.
    cert_max_scale_steps : int
        Maximum rho scaling attempts during certification.
    cert_max_bisection_steps : int
        Maximum bisection iterations for certification.
    cert_method : str
        AutoLiRPA bound method (e.g., "crown", "alpha-crown").
    condition_tolerance : float
        Numerical tolerance for condition satisfaction.
    """

    state_dim: int
    state_bounds: Sequence[float]
    kappa: float = 0.05
    invariance_weight: float = 1.0
    rho_min: float = 1e-6
    cert_step: float = 1.0
    origin_exclusion: float | Sequence[float] | None = None
    rho_scaling: float = 1.2
    bisection_tol: float = 1e-3
    max_scale_steps: int = 20
    max_bisection_steps: int = 40
    cert_method: str = "alpha-crown"
    condition_tolerance: float = 1e-6
    suppress_native_output: bool = True

    @staticmethod
    def from_training_config(
        config: LyapunovTrainingConfig,
        state_bounds: Sequence[float] | None = None,
        cert_step: float = 1.0,
        cert_origin_exclusion: float | Sequence[float] | None = None,
        cert_rho_scaling: float = 1.2,
        cert_bisection_tol: float = 1e-3,
        cert_max_scale_steps: int = 20,
        cert_max_bisection_steps: int = 40,
        cert_method: str = "alpha-crown",
        cert_suppress_native_output: bool = True,
    ) -> "LyapunovCertificationConfig":
        """Build a certification config from a training config.

        Explicit function arguments override values derived from ``config``.
        """
        config_values = {
            "state_dim": config.state_dim,
            "state_bounds": config.state_bounds if state_bounds is None else state_bounds,
            "kappa": config.kappa,
            "invariance_weight": config.invariance_weight,
            "rho_min": config.rho_min,
            "cert_step": cert_step,
            "origin_exclusion": cert_origin_exclusion,
            "rho_scaling": cert_rho_scaling,
            "bisection_tol": cert_bisection_tol,
            "max_scale_steps": cert_max_scale_steps,
            "max_bisection_steps": cert_max_bisection_steps,
            "cert_method": cert_method,
            "condition_tolerance": config.condition_tolerance,
            "suppress_native_output": cert_suppress_native_output,
        }
        return LyapunovCertificationConfig(**config_values)
