import unittest

import torch as th

from lcil.certification.lirpa_lyapunov_bounds import LyapunovRegionBounds
from lcil.certification.progress import CertificationProgress, ProgressLevel
from lcil.certification.region_manager import RegionManager, RegionTable, CoreStatus


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

        self.manager.cache_region_bounds(
            region_bounds,
            regions=regions,
            make_current=False,
            is_root=True,
        )

        partition = self.manager.partition_certification_regions(
            regions,
            region_bounds=region_bounds,
            rho=1.0,
            sublevel_tolerance=0.0,
        )

        self.assertTrue(partition.has_relevant_regions)
        th.testing.assert_close(partition.inside_core_unchecked_regions, regions[:1])
        th.testing.assert_close(partition.boundary_core_unchecked_regions, regions[1:2])
        th.testing.assert_close(partition.irrelevant_regions, regions[2:])

    def test_partition_certification_regions_progress_accounting(self) -> None:
        # Create 11 distinct regions to cover all partition subsets:
        # 0: inside, complete safe (for rho=1.0)
        # 1: inside, core safe
        # 2: inside, core counterexample
        # 3: inside, core unknown
        # 4: inside, core unchecked
        # 5: boundary, complete safe (for rho=1.0)
        # 6: boundary, core safe
        # 7: boundary, core counterexample (complete candidate)
        # 8: boundary, core unknown (complete candidate)
        # 9: boundary, core unchecked
        # 10: irrelevant (outside sublevel set)
        regions = th.tensor(
            [
                [[float(i), float(i)], [float(i) + 0.5, float(i) + 0.5]]
                for i in range(11)
            ],
            dtype=th.float32,
        )
        lower_v = th.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.5], dtype=th.float32)
        upper_v = th.tensor([0.7, 0.8, 0.85, 0.9, 0.95, 1.2, 1.3, 1.25, 1.35, 1.4, 2.0], dtype=th.float32)
        region_bounds = LyapunovRegionBounds(lower=lower_v, upper=upper_v)

        self.manager.cache_region_bounds(
            region_bounds,
            regions=regions,
            make_current=True,
            is_root=True,
        )

        # Region 0: complete safe at rho=1.0
        self.manager.update_complete_safe_max_rho(regions[0:1], verified_mask=th.tensor([True]), rho=1.0)
        # Regions 1, 2, 3: core safe, cex, unknown
        self.manager.update_core_status(
            regions[1:4],
            verified_mask=th.tensor([True, False, False]),
            counterexample_mask=th.tensor([False, True, False]),
            unknown_mask=th.tensor([False, False, True]),
        )
        # Region 5: boundary, complete safe at rho=1.0
        self.manager.update_complete_safe_max_rho(regions[5:6], verified_mask=th.tensor([True]), rho=1.0)
        # Region 6, 7, 8: boundary, core safe, cex, unknown
        self.manager.update_core_status(
            regions[6:9],
            verified_mask=th.tensor([True, False, False]),
            counterexample_mask=th.tensor([False, True, False]),
            unknown_mask=th.tensor([False, False, True]),
        )

        partition = self.manager.partition_certification_regions(
            regions,
            region_bounds=region_bounds,
            rho=1.0,
            sublevel_tolerance=0.0,
        )

        # 1. Verify partitioning correctness and mutual exclusivity
        th.testing.assert_close(partition.cached_complete_safe_regions, regions[[0, 5]])
        th.testing.assert_close(partition.cached_core_safe_regions, regions[[1, 6]])
        th.testing.assert_close(partition.cached_inside_counterexample_regions, regions[2:3])
        th.testing.assert_close(partition.cached_inside_unknown_regions, regions[3:4])
        th.testing.assert_close(partition.inside_core_unchecked_regions, regions[4:5])
        th.testing.assert_close(partition.boundary_complete_candidate_regions, regions[7:9])
        th.testing.assert_close(partition.boundary_core_unchecked_regions, regions[9:10])
        th.testing.assert_close(partition.irrelevant_regions, regions[10:11])

        self.assertEqual(partition.n_cached_resolved, 4)
        self.assertEqual(partition.n_cached_unresolved, 2)
        self.assertEqual(partition.n_irrelevant, 1)
        self.assertEqual(partition.n_pending_verification, 4)
        self.assertEqual(partition.total_partitioned, len(regions))

        # 2. Test rich progress bar integration and verify pending is never negative
        progress = CertificationProgress(ProgressLevel.ALL)
        progress.start_recursive("Region Splits", max_depth=1, force_display=True)
        progress.update_recursive(n_pending=len(regions))
        self.assertEqual(progress._rec_pending, 11)

        # Pass cached/irrelevant outcomes to progress (matching recursive certifier logic)
        if partition.n_irrelevant > 0:
            progress.add_recursive_counts(irrelevant=partition.n_irrelevant, pending=-partition.n_irrelevant)
        if len(partition.cached_complete_safe_regions) > 0:
            progress.add_recursive_counts(
                resolved=len(partition.cached_complete_safe_regions),
                pending=-len(partition.cached_complete_safe_regions),
            )
        if len(partition.cached_core_safe_regions) > 0:
            progress.add_recursive_counts(
                resolved=len(partition.cached_core_safe_regions),
                pending=-len(partition.cached_core_safe_regions),
            )
        if len(partition.cached_inside_counterexample_regions) > 0:
            progress.add_recursive_counts(
                unresolved=len(partition.cached_inside_counterexample_regions),
                pending=-len(partition.cached_inside_counterexample_regions),
            )
        if len(partition.cached_inside_unknown_regions) > 0:
            progress.add_recursive_counts(
                unresolved=len(partition.cached_inside_unknown_regions),
                pending=-len(partition.cached_inside_unknown_regions),
            )

        # Check progress state after cached partition processing:
        # Pending must be non-negative and equal exactly the regions still awaiting certification
        self.assertGreaterEqual(progress._rec_pending, 0)
        self.assertEqual(progress._rec_pending, partition.n_pending_verification)
        self.assertEqual(progress._rec_pending, 4)
        self.assertEqual(progress._rec_resolved, 4)
        self.assertEqual(progress._rec_unresolved, 2)
        self.assertEqual(progress._rec_irrelevant, 1)

        # 3. Simulate processing of remaining pending regions
        # Inside unchecked region verified:
        progress.add_recursive_counts(resolved=1, pending=-1)
        self.assertEqual(progress._rec_pending, 3)

        # Boundary unchecked region verified in core cert:
        progress.add_recursive_counts(resolved=1, pending=-1)
        self.assertEqual(progress._rec_pending, 2)

        # Two boundary candidates certified in complete cert (1 safe, 1 cex):
        progress.add_recursive_counts(resolved=1, pending=-1)
        self.assertEqual(progress._rec_pending, 1)
        progress.add_recursive_counts(unresolved=1, pending=-1)
        self.assertEqual(progress._rec_pending, 0)
        self.assertGreaterEqual(progress._rec_pending, 0)

        # Total accounts for all 11 regions with zero remaining pending
        self.assertEqual(progress._rec_resolved, 7)
        self.assertEqual(progress._rec_unresolved, 3)
        self.assertEqual(progress._rec_irrelevant, 1)
        self.assertEqual(
            progress._rec_resolved + progress._rec_unresolved + progress._rec_irrelevant,
            len(regions),
        )

    def test_boundary_core_failure_does_not_double_count_in_progress(self) -> None:
        """Verify that when boundary regions fail core cert and proceed to complete cert,
        pending is not decremented twice into negative values."""
        region = th.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=th.float32)
        region_bounds = LyapunovRegionBounds(
            lower=th.tensor([0.5], dtype=th.float32),
            upper=th.tensor([1.5], dtype=th.float32),
        )
        self.manager.cache_region_bounds(region_bounds, regions=region, make_current=True, is_root=True)
        partition = self.manager.partition_certification_regions(
            region,
            region_bounds=region_bounds,
            rho=1.0,
            sublevel_tolerance=0.0,
        )

        self.assertEqual(len(partition.boundary_core_unchecked_regions), 1)

        progress = CertificationProgress(ProgressLevel.ALL)
        progress.start_recursive("Region Splits", max_depth=1, force_display=True)
        progress.update_recursive(n_pending=len(region))
        self.assertEqual(progress._rec_pending, 1)

        # When a boundary region fails core cert, it is forwarded to complete candidates
        # WITHOUT updating progress (pending remains 1, unresolved remains 0).
        complete_candidates = partition.boundary_core_unchecked_regions
        self.assertEqual(progress._rec_pending, 1)

        # Complete cert finishes (resolves region) -> pending decrements once to 0
        progress.add_recursive_counts(resolved=len(complete_candidates), pending=-len(complete_candidates))
        self.assertEqual(progress._rec_pending, 0)
        self.assertGreaterEqual(progress._rec_pending, 0)

    def test_progress_bar_pending_never_negative(self) -> None:
        """Test defensive clamping ensuring pending is never negative."""
        progress = CertificationProgress(ProgressLevel.ALL)
        progress.start_recursive("Region Splits", max_depth=1, force_display=True)
        progress.update_recursive(n_pending=2)
        self.assertEqual(progress._rec_pending, 2)

        # Subtract more than pending
        progress.add_recursive_counts(resolved=5, pending=-5)
        self.assertEqual(progress._rec_pending, 0)
        self.assertGreaterEqual(progress._rec_pending, 0)

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

    def test_get_best_fallback_rho_calculates_max_volume_without_holes(self) -> None:
        ids = th.tensor([0, 1, 2, 3, 4], dtype=th.long)
        parent_ids = th.tensor([-1, 0, 0, 0, 0], dtype=th.long)
        
        regions = th.tensor([
            [[0.0, 0.0], [3.0, 3.0]], # Parent
            [[0.0, 0.0], [1.0, 1.0]], # Leaf 1, Vol = 1
            [[1.0, 0.0], [3.0, 1.0]], # Leaf 2, Vol = 2
            [[0.0, 1.0], [2.0, 3.0]], # Leaf 3, Vol = 4
            [[2.0, 1.0], [3.0, 3.0]], # Leaf 4, Vol = 2
        ], dtype=th.float32)
        
        lower_v = th.tensor([0.1, 0.1, 0.3, 0.5, 0.6], dtype=th.float32)
        upper_v = th.tensor([0.9, 0.3, 0.5, 0.8, 0.9], dtype=th.float32)
        
        core_status = th.tensor([
            CoreStatus.UNCHECKED,
            CoreStatus.SAFE,       # L1
            CoreStatus.UNCHECKED,  # L2
            CoreStatus.SAFE,       # L3
            CoreStatus.UNCHECKED,  # L4
        ], dtype=th.long)
        
        complete_safe_max_rho = th.tensor([
            -1.0,
            -1.0,
            1.0, # L2 completely safe up to rho=1.0
            -1.0,
            -1.0,
        ], dtype=th.float32)
        
        depth = th.tensor([0, 1, 1, 1, 1], dtype=th.long)
        
        self.manager.region_table = RegionTable(
            ids=ids,
            regions=regions,
            lower_v=lower_v,
            upper_v=upper_v,
            parent_ids=parent_ids,
            depth=depth,
            core_status=core_status,
            complete_safe_max_rho=complete_safe_max_rho,
        )
        
        best_rho = self.manager.get_best_fallback_rho(rho_min=0.0, sublevel_tolerance=0.0)
        
        # Candidate rhos will include lower_v, upper_v, complete_safe_max_rho
        # rho=0.5 -> has holes (L3 is relevant but upper=0.8 > 0.5, so not certified)
        # rho=0.6 -> has holes (L4 is relevant but not certified)
        # rho=0.3 -> L1 (upper=0.3<=0.3, safe), L2 (complete_safe_max_rho=1.0 >= 0.3). No holes. Vol = 3.
        self.assertAlmostEqual(best_rho, 0.3, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)