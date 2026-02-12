from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..training.train_config import LyapunovTrainingConfig


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
    cert_origin_exclusion : float | None
        Radius around the origin to skip during certification.
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
    cert_origin_exclusion: float | None = None
    cert_rho_scaling: float = 1.2
    cert_bisection_tol: float = 1e-3
    cert_max_scale_steps: int = 20
    cert_max_bisection_steps: int = 40
    cert_method: str = "alpha-crown"
    condition_tolerance: float = 1e-6

    @staticmethod
    def from_training_config(
        config: LyapunovTrainingConfig,
        cert_step: float = 1.0,
        cert_origin_exclusion: float | None = None,
        cert_rho_scaling: float = 1.2,
        cert_bisection_tol: float = 1e-3,
        cert_max_scale_steps: int = 20,
        cert_max_bisection_steps: int = 40,
        cert_method: str = "alpha-crown",
    ) -> "LyapunovCertificationConfig":
        """Build a certification config from a training config."""
        return LyapunovCertificationConfig(
            state_dim=config.state_dim,
            state_bounds=config.state_bounds,
            kappa=config.kappa,
            invariance_weight=config.invariance_weight,
            rho_min=config.rho_min,
            cert_step=cert_step,
            cert_origin_exclusion=cert_origin_exclusion,
            cert_rho_scaling=cert_rho_scaling,
            cert_bisection_tol=cert_bisection_tol,
            cert_max_scale_steps=cert_max_scale_steps,
            cert_max_bisection_steps=cert_max_bisection_steps,
            cert_method=cert_method,
            condition_tolerance=config.condition_tolerance,
        )
