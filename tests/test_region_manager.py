import unittest

import torch as th

from lcil.certification.lirpa_lyapunov_bounds import LyapunovRegionBounds
from lcil.certification.region_manager import RegionManager


class _StubRegionBuilder:
    def __init__(self, root_regions: th.Tensor) -> None:
        self.device = th.device("cpu")
        self.state_dim = int(root_regions.shape[-1])
        self._root_regions = root_regions.to(device=self.device, dtype=th.float32)
        self.build_calls = 0
        self.last_split_regions_args: tuple[th.Tensor, th.Tensor, th.Tensor | None, float] | None = None
        self._split_frontier_return = (
            th.empty((0, 2, self.state_dim), dtype=th.float32, device=self.device),
            th.empty((0, 2, self.state_dim), dtype=th.float32, device=self.device),
        )

    def build_regions(self) -> th.Tensor:
        self.build_calls += 1
        return self._root_regions.clone()

    def split_regions(
        self,
        regions: th.Tensor,
        *,
        split_dims: th.Tensor | None = None,
    ) -> th.Tensor:
        del split_dims
        return regions.clone()

    def split_regions_adjacent_to_reference(
        self,
        regions: th.Tensor,
        reference_regions: th.Tensor,
        *,
        split_dims: th.Tensor | None = None,
        adjacency_tolerance: float = 1e-6,
    ) -> tuple[th.Tensor, th.Tensor]:
        self.last_split_regions_args = (
            regions.clone(),
            reference_regions.clone(),
            None if split_dims is None else split_dims.clone(),
            adjacency_tolerance,
        )
        pending, terminal = self._split_frontier_return
        return pending.clone(), terminal.clone()

    def set_frontier_split_return(
        self,
        pending: th.Tensor,
        terminal: th.Tensor,
    ) -> None:
        self._split_frontier_return = (
            pending.to(device=self.device, dtype=th.float32),
            terminal.to(device=self.device, dtype=th.float32),
        )


class TestRegionManager(unittest.TestCase):
    def setUp(self) -> None:
        self.root_regions = th.tensor(
            [
                [[-1.0, -1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        self.builder = _StubRegionBuilder(self.root_regions)
        self.manager = RegionManager(region_builder=self.builder)

    def test_ensure_regions_builds_root_regions_on_demand(self) -> None:
        regions = self.manager.ensure_regions()

        th.testing.assert_close(regions, self.root_regions)
        self.assertIs(self.manager.regions, regions)
        self.assertEqual(self.builder.build_calls, 1)

    def test_cache_region_bounds_registers_root_regions_and_marks_them_current(self) -> None:
        root_bounds = LyapunovRegionBounds(
            lower=th.tensor([0.5, 1.5], dtype=th.float32),
            upper=th.tensor([0.75, 2.0], dtype=th.float32),
        )

        cached_bounds = self.manager.cache_region_bounds(root_bounds)

        self.assertIs(cached_bounds, self.manager.region_bounds)
        self.assertIs(self.manager._cached_regions, self.manager.regions)
        th.testing.assert_close(self.manager.region_table.regions, self.root_regions)
        th.testing.assert_close(self.manager.region_table.lower_v, root_bounds.lower)
        th.testing.assert_close(self.manager.region_table.upper_v, root_bounds.upper)
        th.testing.assert_close(
            self.manager.region_table.ids,
            th.tensor([0, 1], dtype=th.long),
        )
        th.testing.assert_close(
            self.manager.region_table.parent_ids,
            th.tensor([-1, -1], dtype=th.long),
        )
        th.testing.assert_close(
            self.manager.region_table.depth,
            th.tensor([0, 0], dtype=th.long),
        )

    def test_ensure_cached_for_non_current_regions_preserves_current_state(self) -> None:
        root_bounds = LyapunovRegionBounds(
            lower=th.tensor([0.5, 1.5], dtype=th.float32),
            upper=th.tensor([0.75, 2.0], dtype=th.float32),
        )
        self.manager.cache_region_bounds(root_bounds)
        current_regions = self.manager.regions
        current_bounds = self.manager.region_bounds

        child_regions = th.tensor(
            [
                [[-1.0, -1.0], [-0.5, -0.5]],
                [[-0.5, -0.5], [0.0, 0.0]],
            ],
            dtype=th.float32,
        )
        child_bounds = LyapunovRegionBounds(
            lower=th.tensor([0.25, 0.4], dtype=th.float32),
            upper=th.tensor([0.5, 0.75], dtype=th.float32),
        )
        self.manager.cache_region_bounds(
            child_bounds,
            regions=child_regions,
            make_current=False,
            is_root=False,
        )

        resolved_regions, cached_child_bounds = self.manager.ensure_cached(
            child_regions.clone(),
            make_current=False,
        )

        th.testing.assert_close(resolved_regions, child_regions)
        self.assertIs(self.manager.regions, current_regions)
        self.assertIs(self.manager.region_bounds, current_bounds)
        self.assertIs(self.manager._cached_regions, current_regions)
        th.testing.assert_close(cached_child_bounds.lower, child_bounds.lower)
        th.testing.assert_close(cached_child_bounds.upper, child_bounds.upper)
        th.testing.assert_close(
            self.manager.region_table.depth[-2:],
            th.tensor([-1, -1], dtype=th.long),
        )

    def test_partition_regions_by_sublevel_returns_relevant_and_irrelevant_regions(self) -> None:
        regions = th.tensor(
            [
                [[-1.0, -1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [2.0, 2.0]],
            ],
            dtype=th.float32,
        )
        region_bounds = LyapunovRegionBounds(
            lower=th.tensor([0.5, 1.0, 1.1], dtype=th.float32),
            upper=th.tensor([0.75, 1.25, 1.4], dtype=th.float32),
        )

        relevant, irrelevant = self.manager.partition_regions_by_sublevel(
            regions,
            region_bounds,
            rho=1.0,
            sublevel_tolerance=0.0,
        )

        th.testing.assert_close(relevant, regions[:2])
        th.testing.assert_close(irrelevant, regions[2:])

    def test_split_failed_regions_on_certification_frontier_delegates_to_builder(self) -> None:
        failed_regions = th.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        resolved_regions = th.tensor(
            [
                [[-1.0, -1.0], [0.0, 0.0]],
            ],
            dtype=th.float32,
        )
        pending_regions = th.tensor(
            [
                [[0.0, 0.0], [0.5, 0.5]],
            ],
            dtype=th.float32,
        )
        terminal_regions = th.tensor(
            [
                [[0.5, 0.5], [1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        self.builder.set_frontier_split_return(pending_regions, terminal_regions)

        pending, terminal = self.manager.split_failed_regions_on_certification_frontier(
            failed_regions,
            resolved_regions,
        )

        th.testing.assert_close(pending, pending_regions)
        th.testing.assert_close(terminal, terminal_regions)
        self.assertIsNotNone(self.builder.last_split_regions_args)
        split_failed, split_resolved, split_dims, adjacency_tolerance = self.builder.last_split_regions_args
        th.testing.assert_close(split_failed, failed_regions)
        th.testing.assert_close(split_resolved, resolved_regions)
        self.assertIsNone(split_dims)
        self.assertEqual(adjacency_tolerance, 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)