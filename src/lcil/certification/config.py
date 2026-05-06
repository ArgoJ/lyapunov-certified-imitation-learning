from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numpy.typing import NDArray

from ..lyapunov_learning.config import LyapunovTrainingConfig
from ..utils.base_config import ArgumentParserConfig, JsonDataclass, config_field


@dataclass(frozen=True)
class LyapunovCertificationConfig(JsonDataclass, ArgumentParserConfig):
    """Configuration for Lyapunov certification only.

    Parameters
    ----------
    state_dim : int
        Dimension of the system state.
    cert_bounds : NDArray
        Per-dimension bounds defining the certification region with shape (2, state_dim): [lb, ub].
    kappa : float
        Exponential decay factor in the Lyapunov decrease condition.
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
    sublevel_tolerance : float
        Conservative slack used in the rho-sublevel gate of the single-output
        certification graph. Positive values keep boundary-touching states
        inside the checked set.
    condition_tolerance : float
        Numerical tolerance on the fused verifier output. Positive values allow
        a small residual certification slack for numerical stability.
    condition_margin : float
        Optional additive safety margin on the decrease term during
        certification. Defaults to ``0.0`` to keep certification unchanged.
    suppress_native_output : bool
        Whether to suppress native solver output (e.g., from AutoLiRPA or ABCrown) 
        during certification.
    batch_size : int
        Batch size to use during certification. Larger batch sizes can improve 
        GPU utilization and speed up certification, but require more memory.
    abcrown_timeout : float | None
        Optional per-region ABCrown branch-and-bound timeout in seconds. Lower
        values speed up rho search by turning hard regions into ``unknown``
        sooner so they can be split instead of blocking one batch for minutes.
    abcrown_max_domains : int | None
        Optional cap on the number of branch-and-bound domains processed per
        region. This provides a second runtime guard in addition to timeouts.
    max_recursion_depth : int
        Maximum recursion depth for the certification process. 
        This limits how many times the certification will recursively subdivide 
        regions if they fail certification, to prevent infinite recursion in edge cases.
    skip_boundary_core_cert : bool
        Whether to skip the boundary-region core-certification prepass and send
        boundary regions directly into complete certification.
    """

    state_dim: int = config_field(cli=False)
    cert_bounds: NDArray = config_field(cli=False)
    kappa: float = config_field(default=0.05, help="Exponential decay factor in the Lyapunov decrease condition.")
    rho_min: float = config_field(default=1e-6, help="Minimum admissible rho value.")
    bins_per_dim: int | Sequence[int] = config_field(default=4, help="Initial certification bins per state dimension.")
    center_refinement_factor: float | Sequence[float] = config_field(default=1.0, help="Optional geometric refinement factor for bins near the origin.")
    origin_exclusion: float | Sequence[float] = config_field(default=0.0, help="Radius around the origin to skip during certification.")
    rho_scaling: float = config_field(default=1.2, help="Multiplicative factor used in rho scaling before bisection.")
    bisection_tol: float = config_field(default=1e-3, help="Tolerance used in rho bisection.")
    max_scale_steps: int = config_field(default=20, help="Maximum rho scaling attempts during certification.")
    max_bisection_steps: int = config_field(default=40, help="Maximum bisection iterations during certification.")
    cert_method: str = config_field(default="alpha-crown", help="AutoLiRPA certification backend method.")
    sublevel_tolerance: float = config_field(default=1e-6, help="Slack used in the rho-sublevel gate of the verification graph.")
    condition_tolerance: float = config_field(default=1e-6, help="Numerical tolerance on the fused verifier output.")
    condition_margin: float = config_field(default=0.0, help="Optional additive safety margin on the decrease term during certification.")
    suppress_native_output: bool = config_field(default=True, help="Whether to suppress native solver output during certification.")
    batch_size: int = config_field(default=512, help="Batch size used during certification.")
    abcrown_timeout: float | None = config_field(default=None, help="Optional per-region ABCrown branch-and-bound timeout in seconds.")
    abcrown_max_domains: int | None = config_field(default=None, help="Optional cap on ABCrown branch-and-bound domains per region.")
    max_recursion_depth: int = config_field(default=10, help="Maximum recursion depth for certification region splitting.")
    skip_boundary_core_cert: bool = config_field(default=False, help="Skip the boundary-region core-certification prepass and route boundary regions directly to complete certification.")
    NP_ARRAY_FIELDS = ("cert_bounds",)

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
        if self.sublevel_tolerance < 0.0:
            raise ValueError("sublevel_tolerance must be non-negative.")
        if self.condition_tolerance < 0.0:
            raise ValueError("condition_tolerance must be non-negative.")
        if self.condition_margin < 0.0:
            raise ValueError("condition_margin must be non-negative.")
        if self.abcrown_timeout is not None and self.abcrown_timeout <= 0.0:
            raise ValueError("abcrown_timeout must be positive when provided.")
        if self.abcrown_max_domains is not None and self.abcrown_max_domains <= 0:
            raise ValueError("abcrown_max_domains must be positive when provided.")

        raw_origin_exclusion = self.origin_exclusion
        normalized_origin_exclusion = raw_origin_exclusion
        if isinstance(raw_origin_exclusion, Sequence) and not isinstance(raw_origin_exclusion, (str, bytes)):
            origin_values = tuple(float(value) for value in raw_origin_exclusion)
            if any(value < 0.0 for value in origin_values):
                raise ValueError("origin_exclusion must be non-negative.")
            if len(origin_values) == 1:
                normalized_origin_exclusion = origin_values[0]
            elif len(origin_values) == self.state_dim:
                normalized_origin_exclusion = origin_values
            else:
                raise ValueError("origin_exclusion must be scalar or match state_dim.")
        elif not isinstance(raw_origin_exclusion, (int, float)):
            raise ValueError("origin_exclusion must be scalar or match state_dim.")

        # object.__setattr__, because of frozen=True
        object.__setattr__(self, "bins_per_dim", bins_per_dim)
        object.__setattr__(self, "center_refinement_factor", refinement_factors)
        object.__setattr__(self, "origin_exclusion", normalized_origin_exclusion)
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
        sublevel_tolerance: float | None = None,
        condition_margin: float = 0.0,
        suppress_native_output: bool = True,
        batch_size: int = 512,
        abcrown_timeout: float | None = None,
        abcrown_max_domains: int | None = None,
        max_recursion_depth: int = 10,
        skip_boundary_core_cert: bool = False,
    ) -> "LyapunovCertificationConfig":
        """Build a certification config from a training config.

        Explicit function arguments override values derived from ``config``.
        """
        config_values = {
            "state_dim": config.state_dim,
            "cert_bounds": config.state_bounds if cert_bounds is None else cert_bounds,
            "kappa": config.kappa,
            "rho_min": config.rho_min,
            "bins_per_dim": bins_per_dim,
            "center_refinement_factor": center_refinement_factor,
            "origin_exclusion": origin_exclusion,
            "rho_scaling": rho_scaling,
            "bisection_tol": bisection_tol,
            "max_scale_steps": max_scale_steps,
            "max_bisection_steps": max_bisection_steps,
            "cert_method": cert_method,
            "sublevel_tolerance": (
                config.condition_tolerance
                if sublevel_tolerance is None
                else sublevel_tolerance
            ),
            "condition_tolerance": config.condition_tolerance,
            "condition_margin": condition_margin,
            "suppress_native_output": suppress_native_output,
            "batch_size": batch_size,
            "abcrown_timeout": abcrown_timeout,
            "abcrown_max_domains": abcrown_max_domains,
            "max_recursion_depth": max_recursion_depth,
            "skip_boundary_core_cert": skip_boundary_core_cert,
        }
        return LyapunovCertificationConfig(**config_values)
