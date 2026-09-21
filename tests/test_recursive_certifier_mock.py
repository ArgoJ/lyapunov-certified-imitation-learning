import unittest
from unittest import mock

import numpy as np
import torch as th

from certification_mock_common import (
    CertificationMockedABCrownTestCase,
    LyapunovRegionBounds,
    RecursiveCertifier,
    _MockVerificationResult,
    _RecordingMockRegionCertifier,
    _StatusAwareMockRegionCertifier,
    _complete_candidate_partition,
)
from lcil.certification.abcrown_region_certifier import EarlyExitLevel
from lcil.certification.region_manager import CertificationRegionPartition
from shared_utils import _IdentityDynamics, _QuadraticLyapunov, _ZeroPolicy


class TestRecursiveCertifierMock(CertificationMockedABCrownTestCase):
    """Unit tests specifically verifying the RecursiveCertifier component.

    Tests region processing order (Core -> Complete), boundary routing,
    caching across recursive passes, multi-depth recursion, and splitting logic.
    """

    def _make_certifier(
        self,
        *,
        max_recursion_depth: int = 3,
        skip_core_cert: bool = False,
    ) -> RecursiveCertifier:
        config = self.make_config(
            state_dim=3,
            cert_bounds=np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=np.float32),
            kappa=1e-6,
            rho_min=1e-6,
            bins_per_dim=4,
            center_refinement_factor=0.7,
            origin_exclusion=0.0,
            max_recursion_depth=max_recursion_depth,
            skip_core_cert=skip_core_cert,
        )
        return self.make_recursive_certifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
        )

    def test_certify_recursive_regions_skip_core_routes_directly_to_complete(self) -> None:
        certifier = self._make_certifier(skip_core_cert=True)
        certifier.regions = th.tensor(
            [
                [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        certifier.region_manager.cache_region_bounds(
            LyapunovRegionBounds(
                lower=th.zeros((len(certifier.regions),), dtype=th.float32),
                upper=th.zeros((len(certifier.regions),), dtype=th.float32),
            ),
            regions=certifier.regions,
            make_current=True,
        )

        empty = certifier.regions[:0]
        partition = CertificationRegionPartition(
            irrelevant_regions=empty,
            cached_complete_safe_regions=empty,
            cached_core_safe_regions=empty,
            cached_inside_counterexample_regions=empty,
            cached_inside_unknown_regions=empty,
            inside_core_unchecked_regions=empty,
            boundary_core_unchecked_regions=certifier.regions,
            boundary_complete_candidate_regions=empty,
        )
        complete_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(
                    verified=True,
                    counterexample_found=False,
                    status="safe",
                ),
                _MockVerificationResult(
                    verified=True,
                    counterexample_found=False,
                    status="safe",
                ),
            ]
        )

        with mock.patch.object(
            certifier.region_manager,
            "partition_certification_regions",
            return_value=partition,
        ), mock.patch.object(
            certifier,
            "_get_region_certifier",
            return_value=complete_certifier,
        ), mock.patch.object(
            certifier,
            "_get_core_region_certifier",
            side_effect=AssertionError("boundary core certifier should be skipped"),
        ):
            result = certifier._certify_recursive_regions(
                rho=1.0,
                early_exit=False,
            )

        self.assertEqual(len(complete_certifier.batches), 1)
        self.assertTrue(th.allclose(complete_certifier.batches[0], certifier.regions))
        self.assertEqual(result.resolved.shape[0], 2)
        self.assertEqual(result.unresolved.shape[0], 0)

    def test_certify_recursive_regions_reuses_cached_core_unknown_regions(self) -> None:
        certifier = self._make_certifier()
        certifier.regions = th.tensor(
            [
                [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        certifier.region_manager.cache_region_bounds(
            LyapunovRegionBounds(
                lower=th.tensor([0.0, 0.0], dtype=th.float32),
                upper=th.tensor([0.2, 0.3], dtype=th.float32),
            ),
            regions=certifier.regions,
            make_current=True,
        )
        core_certifier = _StatusAwareMockRegionCertifier(
            [
                _MockVerificationResult(
                    verified=False,
                    counterexample_found=False,
                    status="unknown",
                ),
                _MockVerificationResult(
                    verified=False,
                    counterexample_found=False,
                    status="unknown",
                ),
            ]
        )

        with mock.patch.object(
            certifier,
            "_get_core_region_certifier",
            return_value=core_certifier,
        ), mock.patch.object(
            certifier.region_manager,
            "split_regions",
            side_effect=lambda failed_bs: failed_bs[:0],
        ):
            first_result = certifier._certify_recursive_regions(
                rho=1.0,
                show_progress=False,
                early_exit=False,
            )
            second_result = certifier._certify_recursive_regions(
                rho=1.2,
                show_progress=False,
                early_exit=False,
            )

        self.assertEqual(core_certifier.calls, 2)
        self.assertEqual(first_result.unresolved.shape[0], 2)
        self.assertEqual(second_result.unresolved.shape[0], 2)
        self.assertTrue(th.equal(first_result.unresolved, certifier.regions))
        self.assertTrue(th.equal(second_result.unresolved, certifier.regions))

    def test_certify_recursive_regions_keeps_splitting_unknown_regions(self) -> None:
        certifier = self._make_certifier(max_recursion_depth=1)
        certifier.regions = th.tensor(
            [
                [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        core_certifier = _StatusAwareMockRegionCertifier(
            [
                _MockVerificationResult(
                    verified=True,
                    counterexample_found=False,
                    status="safe",
                ),
                _MockVerificationResult(
                    verified=False,
                    counterexample_found=False,
                    status="unknown",
                ),
            ]
        )

        certifier.region_manager.cache_region_bounds(
            LyapunovRegionBounds(
                lower=th.zeros((len(certifier.regions),), dtype=th.float32),
                upper=th.zeros((len(certifier.regions),), dtype=th.float32),
            ),
            regions=certifier.regions,
            make_current=True,
        )

        with mock.patch.object(
            certifier,
            "_get_core_region_certifier",
            return_value=core_certifier,
        ), mock.patch.object(
            certifier.region_manager,
            "split_regions",
            side_effect=lambda failed_bs: failed_bs[:0],
        ) as split_mock:
            result = certifier._certify_recursive_regions(
                rho=1.0,
                early_exit=False,
            )

        self.assertFalse(result.global_success)
        self.assertEqual(core_certifier.calls, 2)
        split_mock.assert_called_once()
        (split_failed_bs,) = split_mock.call_args.args
        self.assertTrue(th.allclose(split_failed_bs, certifier.regions[1:2]))

    def test_certify_recursive_regions_marks_all_resolved_when_all_are_safe(self) -> None:
        certifier = self._make_certifier(max_recursion_depth=0)
        certifier.regions = th.tensor(
            [
                [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        region_certifier = _StatusAwareMockRegionCertifier(
            [
                _MockVerificationResult(
                    verified=True,
                    counterexample_found=False,
                    status="safe",
                ),
                _MockVerificationResult(
                    verified=True,
                    counterexample_found=False,
                    status="verified",
                ),
            ]
        )

        certifier.region_manager.cache_region_bounds(
            LyapunovRegionBounds(
                lower=th.zeros((len(certifier.regions),), dtype=th.float32),
                upper=th.zeros((len(certifier.regions),), dtype=th.float32),
            ),
            regions=certifier.regions,
            make_current=True,
        )

        with mock.patch.object(
            certifier.region_manager,
            "partition_certification_regions",
            return_value=_complete_candidate_partition(certifier.regions),
        ), mock.patch.object(
            certifier,
            "_get_region_certifier",
            return_value=region_certifier,
        ):
            result = certifier._certify_recursive_regions(
                rho=1.0,
                early_exit=True,
            )

        self.assertTrue(result.global_success)
        self.assertEqual(result.resolved.shape[0], 2)
        self.assertEqual(result.unresolved.shape[0], 0)
        self.assertEqual(region_certifier.calls, 2)

    def test_certify_recursive_regions_multi_depth_core_complete_split_order(self) -> None:
        """Verify that Core runs first and splits unresolved regions, with Complete certification
        only running on remaining boundary regions at the leaf level."""
        certifier = self._make_certifier(max_recursion_depth=1)

        root_region = th.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]], dtype=th.float32)
        child1 = th.tensor([[[-1.0, -1.0, -1.0], [0.0, 1.0, 1.0]]], dtype=th.float32)
        child2 = th.tensor([[[0.0, -1.0, -1.0], [1.0, 1.0, 1.0]]], dtype=th.float32)
        split_children = th.cat([child1, child2], dim=0)

        certifier.regions = root_region

        # Depth 0: Core fails (unknown) -> Complete fails (unknown) -> split
        # Depth 1 (leaf): Core resolves child1, fails child2 -> Complete resolves child2!
        core_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),  # root
                _MockVerificationResult(verified=True, counterexample_found=False, status="safe"),      # child1
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),  # child2
            ]
        )
        complete_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),  # root (depth 0, is_leaf=False)
                _MockVerificationResult(verified=True, counterexample_found=False, status="safe"),      # child2 (depth 1, is_leaf=True)
            ]
        )

        bounder = certifier._get_region_bounder()

        with mock.patch.object(
            certifier, "_get_core_region_certifier", return_value=core_certifier
        ), mock.patch.object(
            certifier, "_get_region_certifier", return_value=complete_certifier
        ), mock.patch.object(
            certifier.region_manager, "split_regions", return_value=split_children
        ) as split_mock, mock.patch.object(
            bounder,
            "compute_bounds_for_regions",
            side_effect=lambda bs, *args, **kwargs: LyapunovRegionBounds(
                lower=th.full((len(bs),), 0.1, dtype=th.float32),
                upper=th.full((len(bs),), 0.5, dtype=th.float32),
            ),
        ):
            result = certifier._certify_recursive_regions(
                rho=1.0,
                early_exit=False,
            )

        self.assertTrue(result.global_success)
        self.assertEqual(len(result.resolved), 2)
        self.assertEqual(len(result.unresolved), 0)

        # Core was called on root (batch 0) and both children (batch 1)
        self.assertEqual(len(core_certifier.batches), 2)
        self.assertEqual(len(core_certifier.batches[0]), 1)
        self.assertEqual(len(core_certifier.batches[1]), 2)
        self.assertEqual(core_certifier.calls, 3)

        # Complete was called on root (batch 0, is_leaf=False) and leaf child2 (batch 1, is_leaf=True)
        self.assertEqual(len(complete_certifier.batches), 2)
        self.assertEqual(len(complete_certifier.batches[0]), 1)
        self.assertTrue(th.equal(complete_certifier.batches[0], root_region))
        self.assertEqual(len(complete_certifier.batches[1]), 1)
        self.assertTrue(th.equal(complete_certifier.batches[1], child2))
        self.assertEqual(complete_certifier.calls, 2)
        self.assertEqual(complete_certifier.is_leaf_calls, [False, True])

        # Splitting was called exactly once on root region
        self.assertEqual(split_mock.call_count, 1)

    def test_certify_recursive_regions_stops_at_max_depth_without_extra_split(self) -> None:
        """Verify that recursion halts at max_depth and does not perform additional splits."""
        certifier = self._make_certifier(max_recursion_depth=1)

        root_region = th.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]], dtype=th.float32)
        child1 = th.tensor([[[-1.0, -1.0, -1.0], [0.0, 1.0, 1.0]]], dtype=th.float32)
        child2 = th.tensor([[[0.0, -1.0, -1.0], [1.0, 1.0, 1.0]]], dtype=th.float32)
        split_children = th.cat([child1, child2], dim=0)
        certifier.regions = root_region

        # Depth 0: Core fails root -> Complete fails root -> split
        # Depth 1 (max_depth): Core resolves child1, fails child2 -> Complete fails child2
        # Max depth reached -> no more splits, child2 returned in unresolved
        core_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),
                _MockVerificationResult(verified=True, counterexample_found=False, status="safe"),
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),
            ]
        )
        complete_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),
            ]
        )

        bounder = certifier._get_region_bounder()

        with mock.patch.object(
            certifier, "_get_core_region_certifier", return_value=core_certifier
        ), mock.patch.object(
            certifier, "_get_region_certifier", return_value=complete_certifier
        ), mock.patch.object(
            certifier.region_manager, "split_regions", return_value=split_children
        ) as split_mock, mock.patch.object(
            bounder,
            "compute_bounds_for_regions",
            side_effect=lambda bs, *args, **kwargs: LyapunovRegionBounds(
                lower=th.full((len(bs),), 0.1, dtype=th.float32),
                upper=th.full((len(bs),), 0.5, dtype=th.float32),
            ),
        ):
            result = certifier._certify_recursive_regions(
                rho=1.0,
                early_exit=False,
            )

        self.assertFalse(result.global_success)
        self.assertEqual(len(result.resolved), 1)
        self.assertEqual(len(result.unresolved), 1)
        self.assertTrue(th.equal(result.unresolved, child2))
        self.assertEqual(split_mock.call_count, 1)
        self.assertEqual(complete_certifier.calls, 2)
        self.assertEqual(complete_certifier.is_leaf_calls, [False, True])

    def test_certify_recursive_regions_early_exit_on_counterexample_stops_without_splitting(self) -> None:
        """Verify that a counterexample under early_exit stops recursion immediately without splitting,
        while the core check runs through completely across all regions."""
        certifier = self._make_certifier(max_recursion_depth=0)

        root_regions = th.tensor(
            [
                [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        certifier.regions = root_regions

        # Core check verifies both regions: region 0 has counterexample, region 1 is safe.
        # Core check must run through completely (both calls executed).
        core_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(verified=False, counterexample_found=True, status="unsafe-pgd"),
                _MockVerificationResult(verified=True, counterexample_found=False, status="safe"),
            ]
        )
        # Complete check only needs to check the unresolved region (region 0) and early exits on counterexample.
        complete_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(verified=False, counterexample_found=True, status="unsafe-pgd"),
            ]
        )

        bounder = certifier._get_region_bounder()

        with mock.patch.object(
            certifier, "_get_core_region_certifier", return_value=core_certifier
        ), mock.patch.object(
            certifier, "_get_region_certifier", return_value=complete_certifier
        ), mock.patch.object(
            certifier.region_manager, "split_regions", side_effect=AssertionError("Should not split on cex")
        ) as split_mock, mock.patch.object(
            bounder,
            "compute_bounds_for_regions",
            side_effect=lambda bs, *args, **kwargs: LyapunovRegionBounds(
                lower=th.full((len(bs),), 0.1, dtype=th.float32),
                upper=th.full((len(bs),), 0.5, dtype=th.float32),
            ),
        ):
            result = certifier._certify_recursive_regions(
                rho=1.0,
                early_exit=True,
            )

        self.assertFalse(result.global_success)
        self.assertTrue(result.counterexample_found)
        self.assertEqual(core_certifier.calls, 2)
        self.assertEqual(len(result.resolved), 1)
        self.assertEqual(len(result.unresolved), 1)
        self.assertEqual(split_mock.call_count, 0)

    def test_certify_recursive_regions_passes_is_leaf_at_max_depth(self) -> None:
        """Verify that is_leaf=False for depth < max_depth and is_leaf=True at depth == max_depth."""
        certifier = self._make_certifier(max_recursion_depth=1)

        root_region = th.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]], dtype=th.float32)
        child = th.tensor([[[-1.0, -1.0, -1.0], [0.0, 1.0, 1.0]]], dtype=th.float32)
        certifier.regions = root_region

        core_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),
            ]
        )
        complete_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),
                _MockVerificationResult(verified=True, counterexample_found=False, status="safe"),
            ]
        )

        bounder = certifier._get_region_bounder()

        with mock.patch.object(
            certifier, "_get_core_region_certifier", return_value=core_certifier
        ), mock.patch.object(
            certifier, "_get_region_certifier", return_value=complete_certifier
        ), mock.patch.object(
            certifier.region_manager, "split_regions", return_value=child
        ), mock.patch.object(
            bounder,
            "compute_bounds_for_regions",
            side_effect=lambda bs, *args, **kwargs: LyapunovRegionBounds(
                lower=th.full((len(bs),), 0.1, dtype=th.float32),
                upper=th.full((len(bs),), 0.5, dtype=th.float32),
            ),
        ):
            result = certifier._certify_recursive_regions(
                rho=1.0,
                early_exit=False,
            )

        self.assertTrue(result.global_success)
        self.assertEqual(complete_certifier.is_leaf_calls, [False, True])

    def test_non_leaf_regions_failing_core_check_invoke_complete_with_is_leaf_false(self) -> None:
        """Verify that at depth < max_depth, failed core regions invoke Complete certification with is_leaf=False."""
        certifier = self._make_certifier(max_recursion_depth=1)
        root_region = th.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]], dtype=th.float32)
        child = th.tensor([[[-1.0, -1.0, -1.0], [0.0, 1.0, 1.0]]], dtype=th.float32)
        certifier.regions = root_region

        core_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),  # depth 0
                _MockVerificationResult(verified=True, counterexample_found=False, status="safe"),      # depth 1
            ]
        )
        complete_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(verified=False, counterexample_found=False, status="unknown"),  # depth 0
            ]
        )

        bounder = certifier._get_region_bounder()

        with mock.patch.object(
            certifier, "_get_core_region_certifier", return_value=core_certifier
        ), mock.patch.object(
            certifier, "_get_region_certifier", return_value=complete_certifier
        ), mock.patch.object(
            certifier.region_manager, "split_regions", return_value=child
        ) as split_mock, mock.patch.object(
            bounder,
            "compute_bounds_for_regions",
            side_effect=lambda bs, *args, **kwargs: LyapunovRegionBounds(
                lower=th.full((len(bs),), 0.1, dtype=th.float32),
                upper=th.full((len(bs),), 0.5, dtype=th.float32),
            ),
        ):
            result = certifier._certify_recursive_regions(
                rho=1.0,
                early_exit=True,
            )

        self.assertTrue(result.global_success)
        self.assertEqual(split_mock.call_count, 1)
        self.assertEqual(complete_certifier.calls, 1)
        self.assertEqual(complete_certifier.is_leaf_calls, [False])

    def test_run_core_certification_runs_through_all_regions_even_when_counterexample_found(self) -> None:
        certifier = self._make_certifier()
        regions = th.tensor(
            [
                [[-2.0, -2.0, -2.0], [-1.0, -1.0, -1.0]],
                [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        certifier.region_manager.cache_region_bounds(
            LyapunovRegionBounds(
                lower=th.zeros((len(regions),), dtype=th.float32),
                upper=th.zeros((len(regions),), dtype=th.float32),
            ),
            regions=regions,
            make_current=True,
        )
        core_certifier = _RecordingMockRegionCertifier(
            [
                _MockVerificationResult(verified=True, counterexample_found=False, status="safe"),
                _MockVerificationResult(verified=False, counterexample_found=True, status="unsafe-pgd"),
                _MockVerificationResult(verified=True, counterexample_found=False, status="safe"),
            ]
        )

        with mock.patch.object(certifier, "_get_core_region_certifier", return_value=core_certifier):
            update = certifier._run_core_certification(
                regions,
                rho=1.0,
                early_exit=EarlyExitLevel.ON_COUNTEREXAMPLE,
            )

        self.assertIsNotNone(update)
        self.assertEqual(core_certifier.calls, 3)
        self.assertEqual(len(update.verified_regions), 2)
        self.assertEqual(len(update.failed_regions), 1)
        self.assertTrue(th.equal(update.failed_regions, regions[1:2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
