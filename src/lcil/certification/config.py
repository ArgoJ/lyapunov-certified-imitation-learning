from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numpy.typing import NDArray

from ..lyapunov_learning.config import LyapunovTrainingConfig


@dataclass(frozen=True)
class LyapunovCertificationConfig:
    """Configuration for Lyapunov certification only.

    Parameters
    ----------
    state_dim : int
        Dimension of the system state.
    cert_bounds : NDArray
        Per-dimension bounds defining the certification region with shape (2, state_dim): [lb, ub].
    kappa : float
        Exponential decay factor in the Lyapunov decrease condition.
    invariance_weight : float
        Weight of the set-invariance penalty term.
    rho_min : float
        Minimum admissible sublevel value.
    cert_bins_per_dim : int | Sequence[int]
        Number of initial certification bins per state dimension. If a scalar is
        provided, the same number of bins is used in every dimension.
    cert_center_refinement_factor : float | Sequence[float]
        Optional per-dimension geometric refinement factor for bins near the
        origin. ``1.0`` keeps the grid uniform, smaller values create finer
        bins toward the center while keeping the same total bin count.
    origin_exclusion : float | Sequence[float] | None
        Radius around the origin to skip during certification. If a sequence is
        provided, it is interpreted per state dimension.
    rho_scaling : float
        Multiplicative factor used in rho scaling before bisection.
    bisection_tol : float
        Tolerance used in rho bisection.
    max_scale_steps : int
        Maximum rho scaling attempts during certification.
    max_bisection_steps : int
        Maximum bisection iterations for certification.
    cert_method : str
        AutoLiRPA bound method (e.g., "crown", "alpha-crown").
    condition_tolerance : float
        Numerical tolerance for condition satisfaction.
    """

    state_dim: int
    cert_bounds: NDArray
    kappa: float = 0.05
    invariance_weight: float = 1.0
    rho_min: float = 1e-6
    cert_bins_per_dim: int | Sequence[int] = 4
    cert_center_refinement_factor: float | Sequence[float] = 1.0
    origin_exclusion: float | Sequence[float] | None = None
    rho_scaling: float = 1.2
    bisection_tol: float = 1e-3
    max_scale_steps: int = 20
    max_bisection_steps: int = 40
    cert_method: str = "alpha-crown"
    condition_tolerance: float = 1e-6
    suppress_native_output: bool = True

    def __post_init__(self) -> None:
        raw_bins = self.cert_bins_per_dim
        if isinstance(raw_bins, int):
            bins_per_dim = (int(raw_bins),) * self.state_dim
        elif isinstance(raw_bins, Sequence):
            bins_per_dim = tuple(int(bins) for bins in raw_bins)
            if len(bins_per_dim) != self.state_dim:
                raise ValueError("cert_bins_per_dim must be scalar or match state_dim.")
        else:
            raise ValueError("cert_bins_per_dim must be scalar or match state_dim.")
        if any(bins <= 0 for bins in bins_per_dim):
            raise ValueError("cert_bins_per_dim must contain only positive integers.")

        raw_refinement = self.cert_center_refinement_factor
        if isinstance(raw_refinement, (int, float)):
            refinement_factors = (float(raw_refinement),) * self.state_dim
        elif isinstance(raw_refinement, Sequence):
            refinement_factors = tuple(float(factor) for factor in raw_refinement)
            if len(refinement_factors) != self.state_dim:
                raise ValueError(
                    "cert_center_refinement_factor must be scalar or match state_dim."
                )
        else:
            raise ValueError(
                "cert_center_refinement_factor must be scalar or match state_dim."
            )
        if any((factor <= 0.0 or factor > 1.0) for factor in refinement_factors):
            raise ValueError(
                f"cert_center_refinement_factor must contain values in (0, 1]. Got: {str(refinement_factors)}"
            )

        # object.__setattr__, because of frozen=True
        object.__setattr__(self, "cert_bins_per_dim", bins_per_dim)
        object.__setattr__(self, "cert_center_refinement_factor", refinement_factors)
        object.__setattr__(self, "cert_method", self.cert_method.strip().lower())
        object.__setattr__(self, "rho_scaling", max(self.rho_scaling, 1.01))

    @staticmethod
    def from_training_config(
        config: LyapunovTrainingConfig,
        cert_bounds: NDArray | None = None,
        cert_bins_per_dim: int | Sequence[int] = 4,
        cert_center_refinement_factor: float | Sequence[float] = 1.0,
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
            "cert_bounds": config.state_bounds if cert_bounds is None else cert_bounds,
            "kappa": config.kappa,
            "invariance_weight": config.invariance_weight,
            "rho_min": config.rho_min,
            "cert_bins_per_dim": cert_bins_per_dim,
            "cert_center_refinement_factor": cert_center_refinement_factor,
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
