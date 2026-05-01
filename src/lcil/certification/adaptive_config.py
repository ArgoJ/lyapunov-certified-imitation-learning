from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..lyapunov_learning.config import LyapunovTrainingConfig
from ..utils.base_config import ArgumentParserConfig, JsonDataclass, config_field


@dataclass(frozen=True)
class AdaptiveCertificationConfig(JsonDataclass, ArgumentParserConfig):
    """Standalone configuration for adaptive LiRPA screening plus per-box ABCrown checks."""

    state_dim: int = config_field(cli=False)
    cert_bounds: NDArray = config_field(cli=False)
    kappa: float = config_field(
        default=0.05,
        help="Exponential decay factor in the Lyapunov decrease condition.",
    )
    bins_per_dim: int | Sequence[int] = config_field(
        default=4,
        help="Initial adaptive-region bins per state dimension.",
    )
    center_refinement_factor: float | Sequence[float] = config_field(
        default=1.0,
        help="Optional geometric refinement factor for bins near the origin.",
    )
    origin_exclusion: float | Sequence[float] | None = config_field(
        default=0.0,
        help="Radius around the origin to skip during root region generation.",
    )
    lirpa_bound_method: str = config_field(
        default="crown",
        help="AutoLiRPA method used to bound scalar V(x) on adaptive regions.",
    )
    sublevel_tolerance: float = config_field(
        default=1e-6,
        help="Slack added to rho when classifying regions against V(x) <= rho.",
    )
    condition_tolerance: float = config_field(
        default=1e-6,
        help="Numerical tolerance used in the ABCrown safety predicate.",
    )
    condition_margin: float = config_field(
        default=0.0,
        help="Optional additive safety margin on the decrease condition.",
    )
    suppress_native_output: bool = config_field(
        default=True,
        help="Whether to suppress native AutoLiRPA and ABCrown output.",
    )
    batch_size: int = config_field(
        default=512,
        help="Batch size used for LiRPA bounds and ABCrown internal solving.",
    )

    NP_ARRAY_FIELDS = ("cert_bounds",)

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")

        bounds = np.asarray(self.cert_bounds, dtype=np.float32)
        if bounds.shape != (2, self.state_dim):
            raise ValueError(
                f"cert_bounds must have shape (2, {self.state_dim}); got {bounds.shape}."
            )
        if not np.all(bounds[1] > bounds[0]):
            raise ValueError("Each upper certification bound must exceed the lower bound.")

        bins_per_dim = self._normalize_bins(self.bins_per_dim)
        center_refinement_factor = self._normalize_refinement(self.center_refinement_factor)
        origin_exclusion = self._normalize_origin_exclusion(self.origin_exclusion)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.sublevel_tolerance < 0.0:
            raise ValueError("sublevel_tolerance must be non-negative.")
        if self.condition_tolerance < 0.0:
            raise ValueError("condition_tolerance must be non-negative.")
        if self.condition_margin < 0.0:
            raise ValueError("condition_margin must be non-negative.")

        object.__setattr__(self, "cert_bounds", bounds)
        object.__setattr__(self, "bins_per_dim", bins_per_dim)
        object.__setattr__(self, "center_refinement_factor", center_refinement_factor)
        object.__setattr__(self, "origin_exclusion", origin_exclusion)
        object.__setattr__(self, "lirpa_bound_method", self.lirpa_bound_method.strip().lower())

    def _normalize_bins(self, bins_per_dim: int | Sequence[int]) -> tuple[int, ...]:
        if isinstance(bins_per_dim, int):
            bins = (int(bins_per_dim),) * self.state_dim
        elif isinstance(bins_per_dim, Sequence):
            bins = tuple(int(value) for value in bins_per_dim)
            if len(bins) != self.state_dim:
                raise ValueError("bins_per_dim must be scalar or match state_dim.")
        else:
            raise ValueError("bins_per_dim must be scalar or match state_dim.")

        if any(value <= 0 for value in bins):
            raise ValueError("bins_per_dim must contain only positive integers.")
        return bins

    def _normalize_refinement(
        self,
        center_refinement_factor: float | Sequence[float],
    ) -> tuple[float, ...]:
        if isinstance(center_refinement_factor, (int, float)):
            refinement = (float(center_refinement_factor),) * self.state_dim
        elif isinstance(center_refinement_factor, Sequence):
            refinement = tuple(float(value) for value in center_refinement_factor)
            if len(refinement) != self.state_dim:
                raise ValueError(
                    "center_refinement_factor must be scalar or match state_dim."
                )
        else:
            raise ValueError(
                "center_refinement_factor must be scalar or match state_dim."
            )

        if any((value <= 0.0 or value > 1.0) for value in refinement):
            raise ValueError("center_refinement_factor must contain values in (0, 1].")
        return refinement

    def _normalize_origin_exclusion(
        self,
        origin_exclusion: float | Sequence[float] | None,
    ) -> float | tuple[float, ...] | None:
        if origin_exclusion is None:
            return None
        if isinstance(origin_exclusion, (int, float)):
            scalar = float(origin_exclusion)
            if scalar < 0.0:
                raise ValueError("origin_exclusion must be non-negative.")
            return scalar
        if isinstance(origin_exclusion, Sequence) and not isinstance(origin_exclusion, (str, bytes)):
            values = tuple(float(value) for value in origin_exclusion)
            if any(value < 0.0 for value in values):
                raise ValueError("origin_exclusion must be non-negative.")
            if len(values) != self.state_dim:
                raise ValueError("origin_exclusion must be scalar or match state_dim.")
            return values
        raise ValueError("origin_exclusion must be scalar or match state_dim.")

    @staticmethod
    def from_training_config(
        config: LyapunovTrainingConfig,
        cert_bounds: NDArray | None = None,
        bins_per_dim: int | Sequence[int] = 4,
        center_refinement_factor: float | Sequence[float] = 1.0,
        origin_exclusion: float | Sequence[float] | None = None,
        lirpa_bound_method: str = "crown",
        sublevel_tolerance: float | None = None,
        condition_margin: float = 0.0,
        suppress_native_output: bool = True,
        batch_size: int = 512,
    ) -> "AdaptiveCertificationConfig":
        return AdaptiveCertificationConfig(
            state_dim=config.state_dim,
            cert_bounds=config.state_bounds if cert_bounds is None else cert_bounds,
            kappa=config.kappa,
            bins_per_dim=bins_per_dim,
            center_refinement_factor=center_refinement_factor,
            origin_exclusion=origin_exclusion,
            lirpa_bound_method=lirpa_bound_method,
            sublevel_tolerance=(
                config.condition_tolerance
                if sublevel_tolerance is None
                else sublevel_tolerance
            ),
            condition_tolerance=config.condition_tolerance,
            condition_margin=condition_margin,
            suppress_native_output=suppress_native_output,
            batch_size=batch_size,
        )