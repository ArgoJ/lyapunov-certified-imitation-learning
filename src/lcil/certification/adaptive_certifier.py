from __future__ import annotations

import math
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch as th
import torch.nn as nn

from .abcrown_region_certifier import CompleteABCrownCertifier
from .config import LyapunovCertificationConfig
from .lirpa_lyapunov_bounds import LiRPALyapunovRegionBounds, LyapunovRegionBounds
from .region_builder import RegionBuilder

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveCertificationResult:
    """Cached region classification and ABCrown results for one rho value."""

    rho: float
    sublevel_threshold: float
    region_bounds: LyapunovRegionBounds
    inside_mask: th.Tensor
    boundary_mask: th.Tensor
    outside_mask: th.Tensor
    inside_regions: th.Tensor
    boundary_regions: th.Tensor
    outside_regions: th.Tensor
    certified_inside_regions: th.Tensor
    failed_inside_regions: th.Tensor
    certified_boundary_regions: th.Tensor
    failed_boundary_regions: th.Tensor

    @property
    def global_success(self) -> bool:
        return (
            len(self.failed_inside_regions) == 0
            and len(self.failed_boundary_regions) == 0
            and (len(self.certified_inside_regions) + len(self.certified_boundary_regions) > 0)
        )

    @property
    def partial_success(self) -> bool:
        return len(self.certified_inside_regions) + len(self.certified_boundary_regions) > 0


@dataclass(frozen=True)
class AdaptiveVolumeStats:
    """Volume summary for one adaptive certification evaluation."""

    certified_volume: float
    certified_boundary_volume: float
    failed_inside_volume: float
    failed_boundary_volume: float
    boundary_volume: float
    outside_volume: float
    relevant_volume: float
    unresolved_volume: float
    unresolved_ratio: float


@dataclass(frozen=True)
class AdaptiveParetoPoint:
    """One sampled point on the adaptive rho/volume curve."""

    rho: float
    result: AdaptiveCertificationResult
    volumes: AdaptiveVolumeStats
    refinement_rounds: int
    feasible: bool


@dataclass(frozen=True)
class AdaptiveCertifyResult:
    """Selection result for variant-2 certification under unresolved-volume tolerance."""

    unresolved_tolerance: float
    pareto_points: tuple[AdaptiveParetoPoint, ...]
    best_point: AdaptiveParetoPoint | None

    @property
    def best_rho(self) -> float:
        if self.best_point is None:
            return 0.0
        return float(self.best_point.rho)


class AdaptiveCertifier:
    """Cache LiRPA root-region bounds and certify inside-sublevel boxes with ABCrown."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovCertificationConfig,
        device: th.device = th.device("cpu"),
    ) -> None:
        self.config = config
        self.device = device

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()

        self.regions: th.Tensor | None = None
        self.region_bounds: LyapunovRegionBounds | None = None
        self.last_result: AdaptiveCertificationResult | None = None

        self._region_bounder: LiRPALyapunovRegionBounds | None = None
        self._abcrown_certifier: CompleteABCrownCertifier | None = None

    def _empty_regions(self) -> th.Tensor:
        return th.empty((0, 2, self.config.state_dim), dtype=th.float32, device=self.device)

    @staticmethod
    def _can_reuse_certified_regions(
        last_result: AdaptiveCertificationResult | None,
        threshold: float,
    ) -> bool:
        if last_result is None:
            return False
        return float(threshold) > float(last_result.sublevel_threshold)

    def _partition_known_regions(
        self,
        regions: th.Tensor,
        known_regions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Return ``(known, unknown)`` partitions using exact packed-region matching."""
        if len(regions) == 0:
            empty = self._empty_regions()
            return empty, empty
        if len(known_regions) == 0:
            return regions[:0], regions

        flat_known = known_regions.detach().cpu().reshape(len(known_regions), -1)
        known_keys = {tuple(float(value) for value in row.tolist()) for row in flat_known}

        flat_regions = regions.detach().cpu().reshape(len(regions), -1)
        known_mask = th.tensor(
            [tuple(float(value) for value in row.tolist()) in known_keys for row in flat_regions],
            dtype=th.bool,
            device=regions.device,
        )
        return regions[known_mask], regions[~known_mask]

    def _concat_regions(self, *region_groups: th.Tensor) -> th.Tensor:
        non_empty_groups = [regions for regions in region_groups if len(regions) > 0]
        if not non_empty_groups:
            return self._empty_regions()
        return th.cat(non_empty_groups, dim=0)

    def _set_regions(
        self,
        regions: th.Tensor,
        *,
        reset_last_result: bool,
    ) -> th.Tensor:
        self.regions = regions.to(device=self.device, dtype=th.float32)
        self.region_bounds = None
        if reset_last_result:
            self.last_result = None
        return self.regions

    def _region_volume(self, regions: th.Tensor) -> float:
        if len(regions) == 0:
            return 0.0
        widths = regions[:, 1] - regions[:, 0]
        return float(widths.prod(dim=1).sum().item())

    def _compute_volume_stats(
        self,
        result: AdaptiveCertificationResult,
    ) -> AdaptiveVolumeStats:
        certified_volume = self._region_volume(result.certified_inside_regions)
        certified_boundary_volume = self._region_volume(result.certified_boundary_regions)
        failed_inside_volume = self._region_volume(result.failed_inside_regions)
        failed_boundary_volume = self._region_volume(result.failed_boundary_regions)
        boundary_volume = self._region_volume(result.boundary_regions)
        outside_volume = self._region_volume(result.outside_regions)
        relevant_volume = certified_volume + certified_boundary_volume + failed_inside_volume + failed_boundary_volume
        unresolved_volume = failed_inside_volume + failed_boundary_volume
        unresolved_ratio = 0.0 if relevant_volume <= 0.0 else unresolved_volume / relevant_volume
        return AdaptiveVolumeStats(
            certified_volume=certified_volume,
            certified_boundary_volume=certified_boundary_volume,
            failed_inside_volume=failed_inside_volume,
            failed_boundary_volume=failed_boundary_volume,
            boundary_volume=boundary_volume,
            outside_volume=outside_volume,
            relevant_volume=relevant_volume,
            unresolved_volume=unresolved_volume,
            unresolved_ratio=unresolved_ratio,
        )

    @staticmethod
    def _normalize_spacing(spacing: str) -> str:
        normalized = spacing.strip().lower()
        if normalized not in {"geometric", "linear"}:
            raise ValueError("spacing must be either 'geometric' or 'linear'.")
        return normalized

    def _resolve_rho_values(
        self,
        rho_values: Sequence[float] | None,
        *,
        rho_min: float | None,
        rho_max: float | None,
        num_points: int,
        spacing: str,
    ) -> tuple[float, ...]:
        if rho_values is not None:
            if len(rho_values) == 0:
                raise ValueError("rho_values must not be empty.")
            normalized_values = sorted({float(rho) for rho in rho_values})
            if any(rho < 0.0 for rho in normalized_values):
                raise ValueError("rho_values must be non-negative.")
            return tuple(normalized_values)

        if rho_min is None or rho_max is None:
            raise ValueError("Either rho_values or both rho_min and rho_max must be provided.")

        if num_points <= 0:
            raise ValueError("num_points must be positive.")

        rho_lo = float(rho_min)
        rho_hi = float(rho_max)
        if rho_lo < 0.0 or rho_hi < 0.0:
            raise ValueError("rho_min and rho_max must be non-negative.")
        if rho_hi < rho_lo:
            raise ValueError("rho_max must be greater than or equal to rho_min.")
        if num_points == 1 or rho_lo == rho_hi:
            return (rho_lo,)

        normalized_spacing = self._normalize_spacing(spacing)
        if normalized_spacing == "geometric":
            if rho_lo <= 0.0:
                raise ValueError("rho_min must be positive when using geometric spacing.")
            values = th.logspace(
                math.log10(rho_lo),
                math.log10(rho_hi),
                steps=int(num_points),
                dtype=th.float64,
            )
        else:
            values = th.linspace(rho_lo, rho_hi, steps=int(num_points), dtype=th.float64)

        return tuple(float(value.item()) for value in values)

    def _build_region_builder(self) -> RegionBuilder:
        return RegionBuilder(
            bounds=self.config.cert_bounds,
            bins_per_dim=self.config.bins_per_dim,
            center_refinement_factor=self.config.center_refinement_factor,
            origin_exclusion=self.config.origin_exclusion,
            device=self.device,
        )

    def _get_region_bounder(self) -> LiRPALyapunovRegionBounds:
        if self._region_bounder is None:
            self._region_bounder = LiRPALyapunovRegionBounds(
                lyap_model=self.lyap_model,
                state_dim=self.config.state_dim,
                batch_size=self.config.batch_size,
                device=self.device,
            )
        return self._region_bounder

    def _build_abcrown_certifier(self) -> CompleteABCrownCertifier:
        return CompleteABCrownCertifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            config=self.config,
            device=self.device,
        )

    def build_root_regions(self) -> th.Tensor:
        """Build and cache the initial region partition."""
        self._set_regions(
            self._build_region_builder().build_regions(),
            reset_last_result=True,
        )
        return self.regions

    def cache_region_bounds(self, regions: th.Tensor | None = None) -> LyapunovRegionBounds:
        """Compute and cache LiRPA bounds for the current or provided regions."""
        if regions is not None:
            self.regions = regions.to(device=self.device, dtype=th.float32)
            self.region_bounds = None
        elif self.region_bounds is not None:
            return self.region_bounds
        elif self.regions is None:
            self.build_root_regions()

        if self.regions is None:
            raise RuntimeError("Regions are not initialized.")

        self.region_bounds = self._get_region_bounder().compute_bounds_for_regions(
            self.regions,
        )
        __logger__.info(
            "Cached LiRPA bounds for %d regions with V \u2208 [%s, %s].",
            len(self.regions),
            str(self.region_bounds.lower.tolist()),
            str(self.region_bounds.upper.tolist()),
        )
        return self.region_bounds

    def _ensure_region_cache(self) -> None:
        if self.regions is None:
            self.build_root_regions()
        if self.region_bounds is None:
            self.cache_region_bounds()

    def _ensure_abcrown_backend(self) -> CompleteABCrownCertifier:
        if self._abcrown_certifier is None:
            self._abcrown_certifier = self._build_abcrown_certifier()
            self._abcrown_certifier.setup_backend()
        return self._abcrown_certifier

    def classify_regions(self, rho: float) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Classify cached regions relative to the guarded rho-sublevel threshold."""
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}.")

        self._ensure_region_cache()
        if self.region_bounds is None:
            raise RuntimeError("Region bounds are not initialized.")

        threshold = float(rho + self.config.sublevel_tolerance)
        return self.region_bounds.sublevel_masks(threshold)

    def certify_inside_regions(
        self,
        rho: float,
        *,
        early_exit: bool = False,
        carried_certified_regions: th.Tensor | None = None,
        carried_certified_boundary_regions: th.Tensor | None = None,
    ) -> AdaptiveCertificationResult:
        """Certify all candidate boxes not proven outside the guarded rho-sublevel set."""
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}.")

        self._ensure_region_cache()
        if self.regions is None or self.region_bounds is None:
            raise RuntimeError("Adaptive region cache is not initialized.")

        threshold = float(rho + self.config.sublevel_tolerance)
        inside_mask, boundary_mask, outside_mask = self.region_bounds.sublevel_masks(threshold)

        inside_regions = self.regions[inside_mask]
        boundary_regions = self.regions[boundary_mask]
        outside_regions = self.regions[outside_mask]

        carried_certified_inside_regions = self._empty_regions()
        carried_certified_boundary_regions = self._empty_regions()
        pending_inside_regions = inside_regions
        pending_boundary_regions = boundary_regions

        if carried_certified_regions is not None:
            carried_certified_inside_regions, pending_inside_regions = self._partition_known_regions(
                inside_regions,
                carried_certified_regions.to(device=self.device, dtype=th.float32),
            )
        elif self._can_reuse_certified_regions(self.last_result, threshold):
            carried_certified_inside_regions = self.last_result.certified_inside_regions
            _, pending_inside_regions = self._partition_known_regions(
                inside_regions,
                carried_certified_inside_regions,
            )

        if carried_certified_boundary_regions is not None:
            carried_certified_boundary_regions, pending_boundary_regions = self._partition_known_regions(
                boundary_regions,
                carried_certified_boundary_regions.to(device=self.device, dtype=th.float32),
            )

        certified_inside_regions = carried_certified_inside_regions
        failed_inside_regions = inside_regions[:0]
        certified_boundary_regions = carried_certified_boundary_regions
        failed_boundary_regions = boundary_regions[:0]

        candidate_regions = self._concat_regions(
            pending_inside_regions,
            pending_boundary_regions,
        )
        pending_inside_count = len(pending_inside_regions)

        if len(candidate_regions) > 0:
            abcrown_certifier = self._ensure_abcrown_backend()
            batch_verification = abcrown_certifier.certify_regions(
                candidate_regions,
                rho,
                early_exit=early_exit,
            )
            is_certified = batch_verification.verified_mask.to(
                device=self.device,
                dtype=th.bool,
            )
            if is_certified.ndim != 1 or len(is_certified) != len(candidate_regions):
                raise ValueError(
                    "AdaptiveCompleteABCrownCertifier.certify_regions().verified_mask must return one boolean per candidate region. "
                    f"Expected {len(candidate_regions)} values, got shape {tuple(is_certified.shape)}."
                )
            inside_is_certified = is_certified[:pending_inside_count]
            boundary_is_certified = is_certified[pending_inside_count:]

            certified_inside_regions = self._concat_regions(
                certified_inside_regions,
                pending_inside_regions[inside_is_certified],
            )
            failed_inside_regions = pending_inside_regions[~inside_is_certified]
            certified_boundary_regions = self._concat_regions(
                certified_boundary_regions,
                pending_boundary_regions[boundary_is_certified],
            )
            failed_boundary_regions = pending_boundary_regions[~boundary_is_certified]

        __logger__.info(
            "Adaptive certification at rho=%.6f: inside=%d, boundary=%d, outside=%d, certified_inside=%d (reused=%d), certified_boundary=%d (reused=%d), failed_inside=%d, failed_boundary=%d.",
            float(rho),
            len(inside_regions),
            len(boundary_regions),
            len(outside_regions),
            len(certified_inside_regions),
            len(carried_certified_inside_regions),
            len(certified_boundary_regions),
            len(carried_certified_boundary_regions),
            len(failed_inside_regions),
            len(failed_boundary_regions),
        )

        self.last_result = AdaptiveCertificationResult(
            rho=float(rho),
            sublevel_threshold=threshold,
            region_bounds=self.region_bounds,
            inside_mask=inside_mask,
            boundary_mask=boundary_mask,
            outside_mask=outside_mask,
            inside_regions=inside_regions,
            boundary_regions=boundary_regions,
            outside_regions=outside_regions,
            certified_inside_regions=certified_inside_regions,
            failed_inside_regions=failed_inside_regions,
            certified_boundary_regions=certified_boundary_regions,
            failed_boundary_regions=failed_boundary_regions,
        )
        return self.last_result

    def _refine_pending_regions(
        self,
        result: AdaptiveCertificationResult,
    ) -> th.Tensor:
        unresolved_regions = self._concat_regions(
            result.failed_inside_regions,
            result.failed_boundary_regions,
        )
        if len(unresolved_regions) == 0:
            return self._concat_regions(
                result.certified_inside_regions,
                result.certified_boundary_regions,
                result.outside_regions,
            )

        refined_unresolved = self._build_region_builder().split_regions(unresolved_regions)
        return self._concat_regions(
            result.certified_inside_regions,
            result.certified_boundary_regions,
            result.outside_regions,
            refined_unresolved,
        )

    def _evaluate_rho_point(
        self,
        rho: float,
        *,
        unresolved_tolerance: float | None,
        max_refinement_rounds: int,
    ) -> AdaptiveParetoPoint:
        if max_refinement_rounds < 0:
            raise ValueError("max_refinement_rounds must be non-negative.")

        threshold = float(rho + self.config.sublevel_tolerance)
        carried_certified_regions = self._empty_regions()
        carried_certified_boundary_regions = self._empty_regions()
        if self._can_reuse_certified_regions(self.last_result, threshold):
            carried_certified_regions = self.last_result.certified_inside_regions

        for refinement_round in range(max_refinement_rounds + 1):
            result = self.certify_inside_regions(
                rho,
                early_exit=False,
                carried_certified_regions=carried_certified_regions,
                carried_certified_boundary_regions=carried_certified_boundary_regions,
            )
            volumes = self._compute_volume_stats(result)
            unresolved_regions = self._concat_regions(
                result.failed_inside_regions,
                result.failed_boundary_regions,
            )
            meets_tolerance = (
                unresolved_tolerance is not None
                and volumes.unresolved_ratio <= float(unresolved_tolerance)
            )
            is_fully_resolved = len(unresolved_regions) == 0

            if meets_tolerance or is_fully_resolved or refinement_round >= max_refinement_rounds:
                point = AdaptiveParetoPoint(
                    rho=float(rho),
                    result=result,
                    volumes=volumes,
                    refinement_rounds=refinement_round,
                    feasible=meets_tolerance or is_fully_resolved,
                )
                __logger__.info(
                    "Adaptive rho evaluation finished: rho=%.6f, unresolved_ratio=%.6f, rounds=%d, feasible=%s.",
                    float(rho),
                    float(volumes.unresolved_ratio),
                    refinement_round,
                    point.feasible,
                )
                return point

            carried_certified_regions = result.certified_inside_regions
            carried_certified_boundary_regions = result.certified_boundary_regions
            self._set_regions(
                self._refine_pending_regions(result),
                reset_last_result=False,
            )

        raise RuntimeError("Adaptive rho evaluation did not terminate as expected.")

    def pareto_curve(
        self,
        rho_values: Sequence[float] | None = None,
        *,
        rho_min: float | None = None,
        rho_max: float | None = None,
        num_points: int = 25,
        spacing: str = "geometric",
        unresolved_tolerance: float | None = None,
        max_refinement_rounds: int = 8,
        reset_regions: bool = True,
    ) -> tuple[AdaptiveParetoPoint, ...]:
        """Evaluate adaptive certification statistics over an ordered rho sweep."""
        if unresolved_tolerance is not None and unresolved_tolerance < 0.0:
            raise ValueError("unresolved_tolerance must be non-negative.")

        sweep_rhos = self._resolve_rho_values(
            rho_values,
            rho_min=rho_min,
            rho_max=rho_max,
            num_points=num_points,
            spacing=spacing,
        )

        if reset_regions or self.regions is None:
            self.build_root_regions()

        pareto_points: list[AdaptiveParetoPoint] = []
        for rho in sweep_rhos:
            pareto_points.append(
                self._evaluate_rho_point(
                    float(rho),
                    unresolved_tolerance=unresolved_tolerance,
                    max_refinement_rounds=max_refinement_rounds,
                )
            )

        return tuple(pareto_points)

    def certify(
        self,
        rho_values: Sequence[float] | None = None,
        *,
        unresolved_tolerance: float,
        rho_min: float | None = None,
        rho_max: float | None = None,
        num_points: int = 25,
        spacing: str = "geometric",
        max_refinement_rounds: int = 8,
        reset_regions: bool = True,
    ) -> AdaptiveCertifyResult:
        """Select the largest sampled rho whose unresolved ratio is at most ``unresolved_tolerance``."""
        if unresolved_tolerance < 0.0:
            raise ValueError("unresolved_tolerance must be non-negative.")

        pareto_points = self.pareto_curve(
            rho_values,
            rho_min=rho_min,
            rho_max=rho_max,
            num_points=num_points,
            spacing=spacing,
            unresolved_tolerance=unresolved_tolerance,
            max_refinement_rounds=max_refinement_rounds,
            reset_regions=reset_regions,
        )

        feasible_points = [point for point in pareto_points if point.feasible]
        best_point = feasible_points[-1] if feasible_points else None

        __logger__.info(
            "Adaptive certify selected rho=%.6f with unresolved_tolerance=%.6f from %d pareto points.",
            0.0 if best_point is None else float(best_point.rho),
            float(unresolved_tolerance),
            len(pareto_points),
        )

        return AdaptiveCertifyResult(
            unresolved_tolerance=float(unresolved_tolerance),
            pareto_points=pareto_points,
            best_point=best_point,
        )