from __future__ import annotations

import logging
from collections.abc import Sequence

import torch as th

__logger__ = logging.getLogger(__name__)


class RegionBuilder:
    """Build an initial grid of certification regions."""

    def __init__(
        self,
        bounds: Sequence[float],
        bins_per_dim: int | Sequence[int],
        center_refinement_factor: float | Sequence[float] = 1.0,
        origin_exclusion: float | Sequence[float] | None = 0.0,
        *,
        device: th.device = th.device("cpu"),
    ) -> None:
        self.device = device
        self.bounds = self._resolve_bounds(bounds, device)
        self.state_dim = int(self.bounds.shape[1])
        self.bins_per_dim = self._normalize_bins(bins_per_dim)
        self.center_refinement_factor = self._normalize_refinement_factors(center_refinement_factor)
        self.origin_exclusion = self._resolve_origin_exclusion(origin_exclusion)

    @staticmethod
    def _resolve_bounds(bounds: Sequence[float], device: th.device) -> th.Tensor:
        bounds_tensor = th.as_tensor(bounds, dtype=th.float32, device=device)
        if bounds_tensor.ndim != 2 or bounds_tensor.shape[0] != 2:
            raise ValueError("bounds must have shape (2, state_dim) [lb, ub].")
        if not th.all(bounds_tensor[1] > bounds_tensor[0]):
            raise ValueError("Each upper bound must be strictly greater than the lower bound.")
        return bounds_tensor

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

    def _normalize_refinement_factors(
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

    def _resolve_origin_exclusion(
        self,
        origin_exclusion: float | Sequence[float] | None,
    ) -> th.Tensor:
        default_exclusion = th.minimum(
            self.bounds.abs().max(dim=0).values * 0.01,
            th.full((self.state_dim,), 0.1, dtype=th.float32, device=self.device),
        )

        if origin_exclusion is None:
            exclusion = default_exclusion
        elif isinstance(origin_exclusion, (int, float)):
            scalar = float(origin_exclusion)
            if scalar < 0.0:
                raise ValueError("origin_exclusion must be non-negative.")
            exclusion = th.full(
                (self.state_dim,),
                scalar,
                dtype=th.float32,
                device=self.device,
            )
        else:
            exclusion = th.as_tensor(
                origin_exclusion,
                dtype=th.float32,
                device=self.device,
            ).reshape(-1)
            if exclusion.numel() != self.state_dim:
                raise ValueError("origin_exclusion must be scalar or match state_dim.")
            if (exclusion < 0.0).any():
                raise ValueError("origin_exclusion must be non-negative.")

        max_centered = th.clamp(th.minimum(-self.bounds[0], self.bounds[1]), min=0.0)
        return th.minimum(exclusion, max_centered)

    @staticmethod
    def _build_center_refined_axis_edges(
        lb: th.Tensor,
        ub: th.Tensor,
        num_bins: int,
        refinement_factor: float,
    ) -> th.Tensor:
        if refinement_factor == 1.0 or lb >= 0.0 or ub <= 0.0:
            return th.linspace(lb, ub, steps=num_bins + 1, device=lb.device)

        neg_span = float((-lb).item())
        pos_span = float(ub.item())
        total_span = neg_span + pos_span
        neg_bins = int(round(num_bins * neg_span / total_span)) if total_span > 0.0 else 0
        neg_bins = max(1, min(num_bins - 1, neg_bins))
        pos_bins = num_bins - neg_bins

        def _side_edges(span: float, bins: int, sign: float, device: th.device) -> th.Tensor:
            if bins <= 0 or span <= 0.0:
                return th.zeros(1, dtype=th.float32, device=device)
            if bins == 1:
                distances = th.tensor([span], dtype=th.float32, device=device)
            else:
                powers = th.arange(bins - 1, -1, -1, dtype=th.float32, device=device)
                weights = refinement_factor ** powers
                widths = span * weights / weights.sum()
                distances = th.cumsum(widths, dim=0)
            return sign * distances

        neg_edges = _side_edges(neg_span, neg_bins, -1.0, lb.device)
        pos_edges = _side_edges(pos_span, pos_bins, 1.0, ub.device)

        return th.cat([
            th.tensor([lb.item()], dtype=th.float32, device=lb.device),
            neg_edges.flip(0)[1:],
            th.zeros(1, dtype=th.float32, device=lb.device),
            pos_edges[:-1],
            th.tensor([ub.item()], dtype=th.float32, device=lb.device),
        ])

    @staticmethod
    def _pack_regions(lbs: th.Tensor, ubs: th.Tensor) -> th.Tensor:
        return th.stack([lbs, ubs], dim=1)

    def _validate_regions(self, regions: th.Tensor) -> th.Tensor:
        validated = regions.to(device=self.device, dtype=th.float32)
        if validated.ndim != 3 or validated.shape[1] != 2 or validated.shape[2] != self.state_dim:
            raise ValueError(
                f"regions must have shape (N, 2, {self.state_dim}); got {tuple(validated.shape)}."
            )
        if len(validated) > 0 and not th.all(validated[:, 1] > validated[:, 0]):
            raise ValueError("Each region upper bound must be strictly greater than its lower bound.")
        return validated

    def split_regions(
        self,
        regions: th.Tensor,
        *,
        split_dims: th.Tensor | None = None,
    ) -> th.Tensor:
        """Split each region once along its widest, or explicitly provided, dimension."""
        regions = self._validate_regions(regions)
        if len(regions) == 0:
            return regions.clone()

        mids = 0.5 * (regions[:, 0] + regions[:, 1])
        widths = regions[:, 1] - regions[:, 0]

        if split_dims is None:
            split_dims = th.argmax(widths, dim=1)
        else:
            split_dims = split_dims.to(device=self.device, dtype=th.long).reshape(-1)
            if split_dims.numel() != len(regions):
                raise ValueError("split_dims must contain exactly one split dimension per region.")
            if ((split_dims < 0) | (split_dims >= self.state_dim)).any():
                raise ValueError("split_dims contains an out-of-range dimension index.")

        low_regions = regions.clone()
        high_regions = regions.clone()
        row_idx = th.arange(len(regions), device=self.device)

        low_regions[row_idx, 1, split_dims] = mids[row_idx, split_dims]
        high_regions[row_idx, 0, split_dims] = mids[row_idx, split_dims]

        refined = th.cat([low_regions, high_regions], dim=0)
        __logger__.info(
            "Adaptively split %d regions into %d subregions.",
            len(regions),
            len(refined),
        )
        return refined

    def split_region(self, region: th.Tensor, *, split_dim: int | None = None) -> th.Tensor:
        """Split a single packed region and return the two resulting subregions."""
        region = region.to(device=self.device, dtype=th.float32)
        if region.shape != (2, self.state_dim):
            raise ValueError(
                f"region must have shape (2, {self.state_dim}); got {tuple(region.shape)}."
            )

        split_dims = None
        if split_dim is not None:
            split_dims = th.tensor([int(split_dim)], device=self.device)
        return self.split_regions(region.unsqueeze(0), split_dims=split_dims)

    def build_regions(self) -> th.Tensor:
        """Construct packed root regions after optional origin-hole filtering."""
        lb_axes = []
        ub_axes = []
        for idx in range(self.state_dim):
            edges = self._build_center_refined_axis_edges(
                lb=self.bounds[0, idx],
                ub=self.bounds[1, idx],
                num_bins=int(self.bins_per_dim[idx]),
                refinement_factor=float(self.center_refinement_factor[idx]),
            )
            lb_axes.append(edges[:-1])
            ub_axes.append(edges[1:])

        lb_mesh = th.meshgrid(*lb_axes, indexing="ij")
        ub_mesh = th.meshgrid(*ub_axes, indexing="ij")
        lbs = th.stack([axis.flatten() for axis in lb_mesh], dim=1)
        ubs = th.stack([axis.flatten() for axis in ub_mesh], dim=1)

        overlaps_origin_per_dim = (lbs < self.origin_exclusion.unsqueeze(0)) & (
            ubs > -self.origin_exclusion.unsqueeze(0)
        )
        overlaps_origin = overlaps_origin_per_dim.all(dim=1)
        valid_mask = ~overlaps_origin

        __logger__.info(
            "Adaptive region split built %d/%d root regions after origin exclusion %s.",
            int(valid_mask.sum().item()),
            int(valid_mask.numel()),
            [float(value) for value in self.origin_exclusion.detach().cpu().tolist()],
        )

        return self._pack_regions(lbs[valid_mask], ubs[valid_mask])