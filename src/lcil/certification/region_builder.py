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
        self._check_exclusion_ratio()

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

    def _check_exclusion_ratio(self, threshold: float = 0.75) -> None:
        """Log warnings if the origin exclusion is too large compared to the overall bounds."""
        bounds_width = self.bounds[1] - self.bounds[0]
        exclusion_width = self.origin_exclusion * 2
        dim_ratios = (exclusion_width / bounds_width)
        max_dim_ratio = dim_ratios.max().item()

        if max_dim_ratio >= 1.0:
            idx = (dim_ratios >= 1.0).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                f"The origin exclusion covers 100% or more of the bounds in dims: {idx}! "
                "Please reduce the origin_exclusion or increase the bounds."
            )
        
        if max_dim_ratio >= threshold:
            __logger__.warning(
                "High Origin Exclusion: In at least one dimension, the exclusion covers %.1f%% of the bounds!",
                max_dim_ratio * 100
            )

        volume_ratio = (exclusion_width.prod() / bounds_width.prod()).item()
        if volume_ratio >= threshold:
            __logger__.warning(
                "High Origin Exclusion Volume: The total exclusion volume covers %.1f%% of the entire state space!",
                volume_ratio * 100
            )

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

    def _subtract_origin_exclusion_from_region(
        self,
        lb: th.Tensor,
        ub: th.Tensor,
    ) -> list[tuple[th.Tensor, th.Tensor]]:
        """Subtract the centered origin-exclusion box from one region.

        Returns packed subregions that tile ``[lb, ub]`` outside the configured
        origin-exclusion box exactly. If the region does not intersect the hole,
        the original region is returned unchanged.
        """
        inner_lb = lb.clone()
        inner_ub = ub.clone()

        active_dims = self.origin_exclusion > 0.0
        if not bool(active_dims.any().item()):
            return [(lb, ub)]

        inner_lb[active_dims] = th.maximum(lb[active_dims], -self.origin_exclusion[active_dims])
        inner_ub[active_dims] = th.minimum(ub[active_dims], self.origin_exclusion[active_dims])

        if not bool((inner_ub[active_dims] > inner_lb[active_dims]).all().item()):
            return [(lb, ub)]

        # Dimensions without active exclusion should remain unchanged and not
        # trigger any extra splits.
        inner_lb[~active_dims] = lb[~active_dims]
        inner_ub[~active_dims] = ub[~active_dims]

        subregions: list[tuple[th.Tensor, th.Tensor]] = []
        for dim in range(self.state_dim):
            prefix_lb = lb.clone()
            prefix_ub = ub.clone()
            if dim > 0:
                prefix_lb[:dim] = inner_lb[:dim]
                prefix_ub[:dim] = inner_ub[:dim]

            lower_lb = prefix_lb.clone()
            lower_ub = prefix_ub.clone()
            lower_ub[dim] = inner_lb[dim]
            if bool((lower_ub > lower_lb).all().item()):
                subregions.append((lower_lb, lower_ub))

            upper_lb = prefix_lb.clone()
            upper_ub = prefix_ub.clone()
            upper_lb[dim] = inner_ub[dim]
            if bool((upper_ub > upper_lb).all().item()):
                subregions.append((upper_lb, upper_ub))

        return subregions

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
        __logger__.debug(
            "Split %d regions into %d subregions.",
            len(regions),
            len(refined),
        )
        return refined

    def _face_adjacency_mask(
        self,
        regions: th.Tensor,
        reference_regions: th.Tensor,
        *,
        tolerance: float = 1e-6,
    ) -> th.Tensor:
        """Return a mask for regions sharing a boundary face with any reference region."""
        regions = self._validate_regions(regions)
        reference_regions = self._validate_regions(reference_regions)

        if len(regions) == 0:
            return th.zeros((0,), dtype=th.bool, device=self.device)
        if len(reference_regions) == 0:
            return th.zeros((len(regions),), dtype=th.bool, device=self.device)

        tol = float(tolerance)
        region_lb = regions[:, None, 0, :]
        region_ub = regions[:, None, 1, :]
        reference_lb = reference_regions[None, :, 0, :]
        reference_ub = reference_regions[None, :, 1, :]

        touches = th.isclose(region_ub, reference_lb, atol=tol, rtol=0.0) | th.isclose(
            reference_ub,
            region_lb,
            atol=tol,
            rtol=0.0,
        )
        overlaps = (region_lb < (reference_ub - tol)) & (reference_lb < (region_ub - tol))

        adjacent_pairs = (touches.sum(dim=2) == 1) & (touches | overlaps).all(dim=2)
        return adjacent_pairs.any(dim=1)

    def split_regions_adjacent_to_reference(
        self,
        regions: th.Tensor,
        reference_regions: th.Tensor,
        *,
        split_dims: th.Tensor | None = None,
        adjacency_tolerance: float = 1e-6,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Split only frontier regions adjacent to references.

        Parameters
        ----------
        regions : th.Tensor
            Candidate packed regions with shape ``(N, 2, state_dim)``.
        reference_regions : th.Tensor
            Packed reference regions whose boundary defines the frontier.
        split_dims : th.Tensor | None, optional
            Optional explicit split dimension per input region.
        adjacency_tolerance : float, optional
            Absolute tolerance used when deciding whether two regions share a
            boundary face.

        Returns
        -------
        tuple[th.Tensor, th.Tensor]
            ``(frontier_children, terminal_regions)`` where ``frontier_children``
            are the split children that still border ``reference_regions`` and
            ``terminal_regions`` contains all remaining regions that should not
            be retried in the frontier expansion.
        """
        regions = self._validate_regions(regions)
        reference_regions = self._validate_regions(reference_regions)

        if len(regions) == 0:
            empty = regions.clone()
            return empty, empty

        if split_dims is not None:
            split_dims = split_dims.to(device=self.device, dtype=th.long).reshape(-1)
            if split_dims.numel() != len(regions):
                raise ValueError("split_dims must contain exactly one split dimension per region.")
            if ((split_dims < 0) | (split_dims >= self.state_dim)).any():
                raise ValueError("split_dims contains an out-of-range dimension index.")

        if len(reference_regions) == 0:
            split_children = self.split_regions(regions, split_dims=split_dims)
            return split_children, regions[:0]

        adjacent_mask = self._face_adjacency_mask(
            regions,
            reference_regions,
            tolerance=adjacency_tolerance,
        )
        frontier_parents = regions[adjacent_mask]
        terminal_regions = regions[~adjacent_mask]

        if len(frontier_parents) == 0:
            return regions[:0], terminal_regions

        frontier_split_dims = None if split_dims is None else split_dims[adjacent_mask]
        split_children = self.split_regions(frontier_parents, split_dims=frontier_split_dims)
        child_frontier_mask = self._face_adjacency_mask(
            split_children,
            reference_regions,
            tolerance=adjacency_tolerance,
        )

        frontier_children = split_children[child_frontier_mask]
        terminal_children = split_children[~child_frontier_mask]
        if len(terminal_children) > 0:
            terminal_regions = (
                terminal_children
                if len(terminal_regions) == 0
                else th.cat([terminal_regions, terminal_children], dim=0)
            )

        __logger__.debug(
            "Frontier split kept %d/%d child regions adjacent to %d reference regions.",
            len(frontier_children),
            len(split_children),
            len(reference_regions),
        )
        return frontier_children, terminal_regions

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
        """Construct packed root regions after exact origin-hole carving."""
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

        region_lbs: list[th.Tensor] = []
        region_ubs: list[th.Tensor] = []
        for region_lb, region_ub in zip(lbs, ubs, strict=True):
            for sub_lb, sub_ub in self._subtract_origin_exclusion_from_region(region_lb, region_ub):
                region_lbs.append(sub_lb)
                region_ubs.append(sub_ub)

        if region_lbs:
            packed_regions = self._pack_regions(th.stack(region_lbs, dim=0), th.stack(region_ubs, dim=0))
        else:
            packed_regions = th.empty((0, 2, self.state_dim), dtype=th.float32, device=self.device)

        __logger__.info(
            "Built %d root regions from %d base grid cells after origin exclusion %s in %s.",
            len(packed_regions),
            len(lbs),
            [round(v, 3) for v in self.origin_exclusion.detach().cpu().tolist()],
            [[round(v, 3) for v in upper_lower] for upper_lower in self.bounds.detach().cpu().tolist()]
        )

        return packed_regions