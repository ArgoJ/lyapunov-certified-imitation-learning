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
        Numerical tolerance for condition satisfaction. Positive values can be used 
        to relax the condition for improved numerical stability.
    suppress_native_output : bool
        Whether to suppress native solver output (e.g., from AutoLiRPA or ABCrown) 
        during certification.
    use_ibp_filter : bool
        Whether to use an IBP filter to pre-filter candidate regions before 
        running the main certification method. This can speed up certification by 
        quickly discarding regions that are clearly not certified, at the cost of 
        additional false negatives. Using conservative IBP filtering.
    batch_size : int
        Batch size to use during certification. Larger batch sizes can improve 
        GPU utilization and speed up certification, but require more memory.
    max_recursion_depth : int
        Maximum recursion depth for the certification process. 
        This limits how many times the certification will recursively subdivide 
        regions if they fail certification, to prevent infinite recursion in edge cases.
    """

    state_dim: int
    cert_bounds: NDArray
    kappa: float = 0.05
    invariance_weight: float = 1.0
    rho_min: float = 1e-6
    bins_per_dim: int | Sequence[int] = 4
    center_refinement_factor: float | Sequence[float] = 1.0
    origin_exclusion: float | Sequence[float] | None = None
    rho_scaling: float = 1.2
    bisection_tol: float = 1e-3
    max_scale_steps: int = 20
    max_bisection_steps: int = 40
    use_ibp_filter: bool = True
    cert_method: str = "alpha-crown"
    condition_tolerance: float = 1e-6
    suppress_native_output: bool = True
    batch_size: int = 512
    max_recursion_depth: int = 10

    def __post_init__(self) -> None:
        raw_bins = self.bins_per_dim
        if isinstance(raw_bins, int):
            bins_per_dim = (int(raw_bins),) * self.state_dim
        elif isinstance(raw_bins, Sequence):
            bins_per_dim = tuple(int(bins) for bins in raw_bins)
            if len(bins_per_dim) != self.state_dim:
                raise ValueError("bins_per_dim must be scalar or match state_dim.")
        else:
            raise ValueError("bins_per_dim must be scalar or match state_dim.")
        if any(bins <= 0 for bins in bins_per_dim):
            raise ValueError("bins_per_dim must contain only positive integers.")

        raw_refinement = self.center_refinement_factor
        if isinstance(raw_refinement, (int, float)):
            refinement_factors = (float(raw_refinement),) * self.state_dim
        elif isinstance(raw_refinement, Sequence):
            refinement_factors = tuple(float(factor) for factor in raw_refinement)
            if len(refinement_factors) != self.state_dim:
                raise ValueError(
                    "center_refinement_factor must be scalar or match state_dim."
                )
        else:
            raise ValueError(
                "center_refinement_factor must be scalar or match state_dim."
            )
        if any((factor <= 0.0 or factor > 1.0) for factor in refinement_factors):
            raise ValueError(
                f"center_refinement_factor must contain values in (0, 1]. Got: {str(refinement_factors)}"
            )

        # object.__setattr__, because of frozen=True
        object.__setattr__(self, "bins_per_dim", bins_per_dim)
        object.__setattr__(self, "center_refinement_factor", refinement_factors)
        object.__setattr__(self, "cert_method", self.cert_method.strip().lower())
        object.__setattr__(self, "rho_scaling", max(self.rho_scaling, 1.01))

    @staticmethod
    def from_training_config(
        config: LyapunovTrainingConfig,
        cert_bounds: NDArray | None = None,
        bins_per_dim: int | Sequence[int] = 4,
        center_refinement_factor: float | Sequence[float] = 1.0,
        origin_exclusion: float | Sequence[float] | None = None,
        rho_scaling: float = 1.2,
        bisection_tol: float = 1e-3,
        max_scale_steps: int = 20,
        max_bisection_steps: int = 40,
        cert_method: str = "alpha-crown",
        suppress_native_output: bool = True,
        use_ibp_filter: bool = True,
        batch_size: int = 512,
        max_recursion_depth: int = 10,
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
            "bins_per_dim": bins_per_dim,
            "center_refinement_factor": center_refinement_factor,
            "origin_exclusion": origin_exclusion,
            "rho_scaling": rho_scaling,
            "bisection_tol": bisection_tol,
            "max_scale_steps": max_scale_steps,
            "max_bisection_steps": max_bisection_steps,
            "cert_method": cert_method,
            "use_ibp_filter": use_ibp_filter,
            "condition_tolerance": config.condition_tolerance,
            "suppress_native_output": suppress_native_output,
            "batch_size": batch_size,
            "max_recursion_depth": max_recursion_depth,
        }
        return LyapunovCertificationConfig(**config_values)
