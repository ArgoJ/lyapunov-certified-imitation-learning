from __future__ import annotations

import logging

import torch as th

from dataclasses import dataclass

from .lirpa_lyapunov_bounds import LyapunovRegionBounds
from .region_builder import RegionBuilder

__logger__ = logging.getLogger(__name__)

@dataclass
class RegionTable:
    ids: th.Tensor           # (N,)
    regions: th.Tensor       # (N, 2, nx)
    lower_v: th.Tensor       # (N,)
    upper_v: th.Tensor       # (N,)
    parent_ids: th.Tensor    # (N,), -1 for roots
    depth: th.Tensor         # (N,)
    core_status: th.Tensor   # (N,), -1 for cex, 0 for unknown, 1 for core
    complete_safe_max_rho: th.Tensor  # (N,)
    

    @classmethod
    def empty(cls, device: th.device, state_dim: int) -> RegionTable:
        empty_long = th.empty((0,), dtype=th.long, device=device)
        empty_float = th.empty((0,), dtype=th.float32, device=device)
        empty_regions = th.empty((0, 2, state_dim), dtype=th.float32, device=device)
        return cls(
            ids=empty_long,
            regions=empty_regions,
            lower_v=empty_float,
            upper_v=empty_float.clone(),
            parent_ids=empty_long.clone(),
            depth=empty_long.clone(),
            core_status=empty_long.clone(),
            complete_safe_max_rho=empty_float.clone(),
        )

@dataclass(frozen=True)
class RegionBatch:
    ids: th.Tensor
    regions: th.Tensor


class RegionManager:
    """Manage current certification regions and cached Lyapunov bounds."""

    def __init__(
        self,
        *,
        region_builder: RegionBuilder,
    ) -> None:
        self.region_builder = region_builder
        self.state_dim = int(region_builder.state_dim)
        self.device = region_builder.device

        self.regions: th.Tensor | None = None
        self.region_bounds: LyapunovRegionBounds | None = None
        self._cached_regions: th.Tensor | None = None
        self._region_bounds_cache: dict[tuple[float, ...], tuple[float, float]] = {}
        self._region_ids: dict[tuple[float, ...], int] = {}
        self._next_region_id = 0
        self.region_table = RegionTable.empty(device=self.device, state_dim=self.state_dim)

    def clear_regions(self) -> None:
        """Clear the current working region batch."""
        self.regions = None
        self.region_bounds = None
        self._cached_regions = None

    def _coerce_regions(self, regions: th.Tensor) -> th.Tensor:
        """Move a region batch onto the managed device and dtype."""
        return regions.to(device=self.device, dtype=th.float32)

    def set_regions(self, regions: th.Tensor) -> th.Tensor:
        """Set the current certification regions and invalidate current bounds."""
        self.regions = self._coerce_regions(regions)
        self.region_bounds = None
        self._cached_regions = None
        return self.regions

    def _resolve_regions(
        self,
        regions: th.Tensor | None = None,
        *,
        make_current: bool,
    ) -> th.Tensor:
        """Resolve a region batch, optionally promoting it to the current working set."""
        if regions is not None:
            if make_current:
                return self.set_regions(regions)
            return self._coerce_regions(regions)

        return self.ensure_regions()

    @staticmethod
    def region_key(region: th.Tensor) -> tuple[float, ...]:
        """Return an exact cache key for one packed region."""
        return tuple(float(value) for value in region.detach().cpu().reshape(-1).tolist())

    def _register_regions(
        self,
        regions: th.Tensor,
        region_bounds: LyapunovRegionBounds,
        *,
        is_root: bool,
    ) -> None:
        new_ids: list[int] = []
        new_regions: list[th.Tensor] = []
        new_lower: list[th.Tensor] = []
        new_upper: list[th.Tensor] = []
        new_parent_ids: list[int] = []
        new_depth: list[int] = []

        lower = region_bounds.lower.reshape(-1)
        upper = region_bounds.upper.reshape(-1)
        default_depth = 0 if is_root else -1

        for idx, region in enumerate(regions):
            key = self.region_key(region)
            if key in self._region_ids:
                continue

            region_id = self._next_region_id
            self._next_region_id += 1
            self._region_ids[key] = region_id

            new_ids.append(region_id)
            new_regions.append(region.detach().to(device=self.device, dtype=th.float32))
            new_lower.append(lower[idx].detach().to(device=self.device, dtype=th.float32).reshape(()))
            new_upper.append(upper[idx].detach().to(device=self.device, dtype=th.float32).reshape(()))
            new_parent_ids.append(-1)
            new_depth.append(default_depth)

        if not new_ids:
            return

        appended_table = RegionTable(
            ids=th.tensor(new_ids, dtype=th.long, device=self.device),
            regions=th.stack(new_regions, dim=0),
            lower_v=th.stack(new_lower, dim=0),
            upper_v=th.stack(new_upper, dim=0),
            parent_ids=th.tensor(new_parent_ids, dtype=th.long, device=self.device),
            depth=th.tensor(new_depth, dtype=th.long, device=self.device),
            core_status=th.zeros(len(new_ids), dtype=th.long, device=self.device),
            complete_safe_max_rho=th.full(
                (len(new_ids),),
                float("-inf"),
                dtype=th.float32,
                device=self.device,
            ),
        )

        if self.region_table.ids.numel() == 0:
            self.region_table = appended_table
            return

        self.region_table = RegionTable(
            ids=th.cat([self.region_table.ids, appended_table.ids], dim=0),
            regions=th.cat([self.region_table.regions, appended_table.regions], dim=0),
            lower_v=th.cat([self.region_table.lower_v, appended_table.lower_v], dim=0),
            upper_v=th.cat([self.region_table.upper_v, appended_table.upper_v], dim=0),
            parent_ids=th.cat([self.region_table.parent_ids, appended_table.parent_ids], dim=0),
            depth=th.cat([self.region_table.depth, appended_table.depth], dim=0),
            core_status=th.cat([self.region_table.core_status, appended_table.core_status], dim=0),
            complete_safe_max_rho=th.cat([self.region_table.complete_safe_max_rho, appended_table.complete_safe_max_rho], dim=0),
        )

    def cache_bounds_for_regions(
        self,
        regions: th.Tensor,
        region_bounds: LyapunovRegionBounds,
        *,
        is_root: bool,
    ) -> LyapunovRegionBounds:
        """Persist per-region V bounds in the local bounds cache and table."""
        lower = region_bounds.lower.detach().cpu().reshape(-1)
        upper = region_bounds.upper.detach().cpu().reshape(-1)
        for idx, region in enumerate(regions):
            self._region_bounds_cache[self.region_key(region)] = (
                float(lower[idx].item()),
                float(upper[idx].item()),
            )
        self._register_regions(regions, region_bounds, is_root=is_root)
        return region_bounds

    def lookup_cached_region_bounds(self, regions: th.Tensor) -> LyapunovRegionBounds | None:
        """Return cached V bounds for ``regions`` if every region is already known."""
        if len(regions) == 0:
            empty = th.empty((0,), dtype=th.float32, device=self.device)
            return LyapunovRegionBounds(lower=empty, upper=empty)

        cached_bounds: list[tuple[float, float]] = []
        for region in regions:
            key = self.region_key(region)
            if key not in self._region_bounds_cache:
                return None
            cached_bounds.append(self._region_bounds_cache[key])

        lower = th.tensor(
            [bound[0] for bound in cached_bounds],
            dtype=th.float32,
            device=self.device,
        )
        upper = th.tensor(
            [bound[1] for bound in cached_bounds],
            dtype=th.float32,
            device=self.device,
        )
        return LyapunovRegionBounds(lower=lower, upper=upper)

    def build_root_regions(self) -> th.Tensor:
        """Build and cache the root certification regions."""
        return self.set_regions(self.region_builder.build_regions())

    def ensure_regions(self, regions: th.Tensor | None = None) -> th.Tensor:
        """Ensure a current region batch exists and return it."""
        if regions is not None:
            return self.set_regions(regions)

        if self.regions is None:
            self.build_root_regions()

        if self.regions is None:
            raise RuntimeError("Certification regions are not initialized.")

        return self.regions

    def ensure_cached(
        self,
        regions: th.Tensor | None = None,
        *,
        make_current: bool = True,
    ) -> tuple[th.Tensor, LyapunovRegionBounds | None]:
        """Ensure a current region batch exists and return any cached V bounds."""
        current_regions = self._resolve_regions(regions, make_current=make_current)
        cached_region_bounds = self.get_cached_region_bounds(current_regions)
        if make_current and cached_region_bounds is not None:
            self.region_bounds = cached_region_bounds
            self._cached_regions = current_regions
        return current_regions, cached_region_bounds

    def cache_region_bounds(
        self,
        region_bounds: LyapunovRegionBounds,
        *,
        regions: th.Tensor | None = None,
        make_current: bool = True,
        is_root: bool | None = None,
    ) -> LyapunovRegionBounds:
        """Cache bounds for the current or provided regions and mark them current."""
        current_regions = self._resolve_regions(regions, make_current=make_current)
        if is_root is None:
            is_root = make_current and self.region_table.ids.numel() == 0

        cached_region_bounds = self.cache_bounds_for_regions(
            current_regions,
            region_bounds,
            is_root=is_root,
        )
        if make_current:
            self.region_bounds = cached_region_bounds
            self._cached_regions = current_regions
            __logger__.info(
                "Cached V bounds for %d certification regions.",
                len(current_regions),
            )
            return self.region_bounds

        return cached_region_bounds

    def ensure_region_cache(self) -> None:
        """Ensure a current region batch exists."""
        self.ensure_regions()

    def classify_regions(
        self,
        rho: float,
        *,
        sublevel_tolerance: float,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Classify cached current regions relative to the guarded rho-sublevel threshold."""
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}.")

        self.ensure_region_cache()
        if self.region_bounds is None:
            raise RuntimeError("Cached V region bounds are not initialized.")

        threshold = float(rho + sublevel_tolerance)
        return self.region_bounds.sublevel_masks(threshold)

    def get_cached_region_bounds(self, regions: th.Tensor) -> LyapunovRegionBounds | None:
        """Return cached V bounds for ``regions`` when available."""
        if self.region_bounds is not None and self._cached_regions is not None and regions is self._cached_regions:
            return self.region_bounds
        return self.lookup_cached_region_bounds(regions)

    def _lookup_region_rows(self, regions: th.Tensor) -> th.Tensor:
        """Return row indices in ``region_table`` for known regions.

        Region ids are assigned monotonically and appended in the same order,
        so the stored id also identifies the corresponding row position.
        """
        if len(regions) == 0:
            return th.empty((0,), dtype=th.long, device=self.device)

        row_indices: list[int] = []
        for region in regions:
            key = self.region_key(region)
            if key not in self._region_ids:
                raise KeyError("Region is not registered in the cache table.")
            row_indices.append(int(self._region_ids[key]))

        return th.tensor(row_indices, dtype=th.long, device=self.device)

    def get_core_status(self, regions: th.Tensor) -> th.Tensor:
        """Return cached core-certification status for ``regions``."""
        rows = self._lookup_region_rows(regions)
        return self.region_table.core_status[rows]

    def get_complete_safe_mask(self, regions: th.Tensor, rho: float) -> th.Tensor:
        """Return mask of regions already proven safe for all ``rho' <= rho``."""
        rows = self._lookup_region_rows(regions)
        return self.region_table.complete_safe_max_rho[rows] >= float(rho)

    def update_core_status(
        self,
        regions: th.Tensor,
        *,
        verified_mask: th.Tensor,
        counterexample_mask: th.Tensor,
    ) -> None:
        """Persist core-certification outcomes for ``regions``."""
        if len(regions) == 0:
            return

        rows = self._lookup_region_rows(regions)
        verified_rows = rows[verified_mask.to(device=self.device, dtype=th.bool)]
        counterexample_rows = rows[counterexample_mask.to(device=self.device, dtype=th.bool)]

        if len(verified_rows) > 0:
            self.region_table.core_status[verified_rows] = 1
        if len(counterexample_rows) > 0:
            self.region_table.core_status[counterexample_rows] = -1

    def update_complete_safe_max_rho(
        self,
        regions: th.Tensor,
        *,
        verified_mask: th.Tensor,
        rho: float,
    ) -> None:
        """Persist the largest rho for which ``regions`` were completely safe."""
        if len(regions) == 0:
            return

        rows = self._lookup_region_rows(regions)
        safe_rows = rows[verified_mask.to(device=self.device, dtype=th.bool)]
        if len(safe_rows) == 0:
            return

        safe_rho = th.full(
            (len(safe_rows),),
            float(rho),
            dtype=th.float32,
            device=self.device,
        )
        self.region_table.complete_safe_max_rho[safe_rows] = th.maximum(
            self.region_table.complete_safe_max_rho[safe_rows],
            safe_rho,
        )

    def split_regions(
        self,
        regions: th.Tensor,
        *,
        split_dims: th.Tensor | None = None,
    ) -> th.Tensor:
        """Split each region once using the managed RegionBuilder."""
        return self.region_builder.split_regions(regions, split_dims=split_dims)

    def split_regions_adjacent_to_reference(
        self,
        regions: th.Tensor,
        reference_regions: th.Tensor,
        *,
        split_dims: th.Tensor | None = None,
        adjacency_tolerance: float = 1e-6,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Split only frontier regions adjacent to references."""
        return self.region_builder.split_regions_adjacent_to_reference(
            regions,
            reference_regions,
            split_dims=split_dims,
            adjacency_tolerance=adjacency_tolerance,
        )

    def split_failed_regions_on_certification_frontier(
        self,
        failed_regions: th.Tensor,
        resolved_regions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Split unresolved regions along the certification frontier."""
        return self.split_regions_adjacent_to_reference(
            failed_regions,
            resolved_regions,
        )