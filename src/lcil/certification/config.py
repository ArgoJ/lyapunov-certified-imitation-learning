from __future__ import annotations

import logging

from collections.abc import Sequence
from dataclasses import dataclass
from numpy.typing import NDArray

from ..lyapunov_learning.config import LyapunovTrainingConfig
from ..utils.base_config import (
    ArgumentParserConfig,
    JsonDataclass,
    config_field,
    fraction_validator,
    optional_validator,
    positive_validator,
    non_negative_validator,
    growth_rate_validator,
    run_field_validators,
    sequence_validator,
    normalize_scalar_or_sequence,
)
from ..utils.constants import *


__logger__ = logging.getLogger(__name__)

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
        Origin exclusion per dimension of bounds. If a sequence is
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

    state_dim: int = config_field(cli=False, validators=(positive_validator,))
    cert_bounds: NDArray = config_field(cli=False)

    # Regions
    bins_per_dim: int | Sequence[int] = config_field(
        default=1,
        help="Initial certification bins per state dimension.",
        validators=(sequence_validator(positive_validator),),
    )
    center_refinement_factor: float | Sequence[float] = config_field(
        default=1.0,
        help="Optional geometric refinement factor for bins near the origin.",
        validators=(
            sequence_validator(positive_validator),
            sequence_validator(fraction_validator),
        ),
    )
    origin_exclusion: float | Sequence[float] = config_field(
        default=0.0,
        help="Origin exclusion per dimension of bounds.",
        validators=(sequence_validator(non_negative_validator),),
    )

    # Certification parameters
    kappa: float = config_field(
        default=0.05,
        help="Exponential decay factor in the Lyapunov decrease condition.",
        display_alias="\u03BA",
        validators=(positive_validator,)
    )
    rho_min: float = config_field(
        default=1e-6,
        help="Minimum admissible rho value.",
        display_alias="\u03C1_min",
        validators=(positive_validator,)
    )
    cert_method: str = config_field(
        default="alpha-crown",
        help="AutoLiRPA certification backend method.",
        display_alias="method",
    )
    sublevel_tolerance: float = config_field(
        default=1e-6,
        help="Slack used in the rho-sublevel gate of the verification graph.",
        display_alias="sublevel_tol",
        validators=(non_negative_validator,)
    )
    condition_tolerance: float = config_field(
        default=1e-6,
        help="Numerical tolerance on the fused verifier output.",
        display_alias="cond_tol",
        validators=(non_negative_validator,)
    )
    condition_margin: float = config_field(
        default=0.0,
        help="Optional additive safety margin on the decrease term during certification.",
        display_alias="margin",
        validators=(non_negative_validator,)
    )

    # Search and Bisection parameters
    rho_scaling: float = config_field(
        default=1.2,
        help="Multiplicative factor used in rho scaling before bisection.",
        display_alias="\u03C1_scale",
        validators=(growth_rate_validator,)
    )
    bisection_tol: float = config_field(
        default=1e-3,
        help="Tolerance used in rho bisection.",
        display_alias="bisect_tol",
        validators=(positive_validator,)
    )
    max_scale_steps: int = config_field(
        default=20,
        help="Maximum rho scaling attempts during certification.",
        display_alias="scale_steps",
        validators=(positive_validator,)
    )
    max_bisection_steps: int = config_field(
        default=40,
        help="Maximum bisection iterations during certification.",
        display_alias="bisect_steps",
        validators=(positive_validator,)
    )
    max_recursion_depth: int = config_field(
        default=10,
        help="Maximum recursion depth for certification region splitting.",
        display_alias="split_depth",
        validators=(non_negative_validator,)
    )
    skip_boundary_core_cert: bool = config_field(
        default=False,
        help="Skip the boundary-region core-certification prepass and route boundary regions directly to complete certification.",
        display_alias="skip_core_cert"
    )


    # AB-Crown specific
    abcrown_timeout: float | None = config_field(
        default=None,
        help="Optional per-region ABCrown branch-and-bound timeout in seconds.",
        validators=(optional_validator(positive_validator),)
    )
    abcrown_max_domains: int | None = config_field(
        default=None,
        help="Optional cap on ABCrown branch-and-bound domains per region.",
        display_alias="abcrown_domains",
        validators=(optional_validator(positive_validator),),
    )
    batch_size: int = config_field(
        default=512,
        help="Batch size used during certification.",
        validators=(positive_validator,)
    )

    # Others
    suppress_native_output: bool = config_field(
        default=True,
        help="Whether to suppress native solver output during certification."
    )

    NP_ARRAY_FIELDS = ("cert_bounds",)
    DEFAULT_FILE_NAME = CERTIFICATION_CONFIG_FILENAME

    def __post_init__(self) -> None:
        run_field_validators(self)
        bins_per_dim = normalize_scalar_or_sequence(
            self.bins_per_dim,
            state_dim=self.state_dim,
            name="bins_per_dim",
            caster=int,
        )
        refinement_factors = normalize_scalar_or_sequence(
            self.center_refinement_factor,
            state_dim=self.state_dim,
            name="center_refinement_factor",
            caster=float,
        )
        normalized_origin_exclusion = normalize_scalar_or_sequence(
            self.origin_exclusion,
            state_dim=self.state_dim,
            name="origin_exclusion",
            caster=float,
        )

        # object.__setattr__, because of frozen=True
        object.__setattr__(self, "bins_per_dim", bins_per_dim)
        object.__setattr__(self, "center_refinement_factor", refinement_factors)
        object.__setattr__(self, "origin_exclusion", normalized_origin_exclusion)
        object.__setattr__(self, "cert_method", self.cert_method.strip().lower())

        if all(e == 0.0 for e in self.origin_exclusion):
            __logger__.warning("You may want to set a positive origin_exclusion to avoid numerical issues near the origin during certification.")

    @staticmethod
    def from_training_config(
        config: LyapunovTrainingConfig,
        cert_bounds: NDArray | None = None,
        bins_per_dim: int | Sequence[int] = 4,
        center_refinement_factor: float | Sequence[float] = 1.0,
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
            "origin_exclusion": config.origin_exclusion,
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
