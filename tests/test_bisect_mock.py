import importlib
import unittest
from dataclasses import dataclass
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn
from plot_assertions_mixin import PlotAssertionsMixin


from shared_utils import (
    _DirectionalScaleDynamics,
    _IdentityDynamics,
    _NegativeQuadraticLyapunov,
    _QuadraticLyapunov,
    _ZeroDynamics,
    _ZeroPolicy,
)
from certification_mock_common import (
    BisectCertifier,
    CertificationMockedABCrownTestCase,
)
from lcil.certification.lirpa_lyapunov_bounds import LyapunovRegionBounds
from lcil.certification.region_manager import CertificationRegionPartition
from lcil.utils.lcil_plt import certified_regions_2d, lyapunov_cert_regions


@dataclass(frozen=True)
class _MockVerificationResult:
    verified: bool
    counterexample_found: bool
    status: str


@dataclass(frozen=True)
class _MockBatchVerification:
    verified_mask: th.Tensor
    counterexample_mask: th.Tensor
    unknown_mask: th.Tensor

    @property
    def failed_mask(self) -> th.Tensor:
        return self.counterexample_mask | self.unknown_mask

    @property
    def any_counterexample(self) -> bool:
        return bool(self.counterexample_mask.any().item())


class _StatusAwareMockRegionCertifier:
    def __init__(self, results: list[_MockVerificationResult]):
        self.results = list(results)
        self.calls = 0

    def verify_region(self, region: th.Tensor, rho: float) -> _MockVerificationResult:
        del region, rho
        if self.calls >= len(self.results):
            raise AssertionError("verify_region called more often than expected.")
        result = self.results[self.calls]
        self.calls += 1
        return result

    def certify_regions(
        self,
        regions: th.Tensor,
        rho: float,
        *,
        early_exit: bool = False,
        show_progress: bool = False,
    ) -> _MockBatchVerification:
        del show_progress
        verified_mask = th.zeros((len(regions),), dtype=th.bool)
        counterexample_mask = th.zeros((len(regions),), dtype=th.bool)
        unknown_mask = th.zeros((len(regions),), dtype=th.bool)

        for idx, region in enumerate(regions):
            result = self.verify_region(region, rho)
            verified_mask[idx] = result.verified
            counterexample_mask[idx] = result.counterexample_found
            unknown_mask[idx] = (not result.verified and not result.counterexample_found)
            if early_exit and result.counterexample_found:
                break

        return _MockBatchVerification(
            verified_mask=verified_mask,
            counterexample_mask=counterexample_mask,
            unknown_mask=unknown_mask,
        )


class _RecordingMockRegionCertifier(_StatusAwareMockRegionCertifier):
    def __init__(self, results: list[_MockVerificationResult]):
        super().__init__(results)
        self.batches: list[th.Tensor] = []

    def certify_regions(
        self,
        regions: th.Tensor,
        rho: float,
        *,
        early_exit: bool = False,
        show_progress: bool = False,
    ) -> _MockBatchVerification:
        self.batches.append(regions.clone())
        return super().certify_regions(
            regions,
            rho,
            early_exit=early_exit,
            show_progress=show_progress,
        )


def _complete_candidate_partition(regions: th.Tensor) -> CertificationRegionPartition:
    empty = regions[:0]
    return CertificationRegionPartition(
        irrelevant_regions=empty,
        cached_complete_safe_regions=empty,
        cached_core_safe_regions=empty,
        cached_inside_counterexample_regions=empty,
        cached_inside_unknown_regions=empty,
        inside_core_unchecked_regions=empty,
        boundary_core_unchecked_regions=empty,
        boundary_complete_candidate_regions=regions,
    )


class TestBisectCertifier(PlotAssertionsMixin, CertificationMockedABCrownTestCase):

    @classmethod
    def _make_certifier(
        cls,
        lyap_model: nn.Module,
        *,
        dyn_model: nn.Module | None = None,
        kappa: float = 0.1,
        rho_min: float = 1e-6,
    ) -> BisectCertifier:
        config = cls.make_config(
            state_dim=3,
            cert_bounds=np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=np.float32),
            kappa=kappa,
            rho_min=rho_min,
            bins_per_dim=4,
            center_refinement_factor=0.7,
            origin_exclusion=0.0,
            max_scale_steps=6,
            max_bisection_steps=6,
            lirpa_method="alpha-crown",
            condition_tolerance=1e-6,
            max_recursion_depth=3,
        )
        return cls.make_bisect_certifier(
            policy_model=_ZeroPolicy(),
            lyap_model=lyap_model,
            dyn_model=_ZeroDynamics() if dyn_model is None else dyn_model,
            config=config,
        )

    def _assert_region_plot_written(
        self,
        certification_result,
        stem: str,
    ) -> None:
        self._assert_plot_written(
            plot_fn=certified_regions_2d,
            stem=stem,
            plot_kwargs={
                "certification_result": certification_result,
                "state_labels": ["x0", "x1", "x2"],
            },
        )

    @staticmethod
    def _to_numpy_lyapunov(
        lyap_model: nn.Module,
        state_dim: int,
    ):
        def _lyapunov_func(x: np.ndarray) -> np.ndarray | float:
            x_array = np.asarray(x, dtype=np.float32)

            if x_array.shape[-1] == state_dim:
                x_lifted = x_array
            elif x_array.shape[-1] < state_dim:
                # mpc_datagen may evaluate pairwise projections with reduced tail dimensions.
                # Preserve existing coordinates and pad missing higher dimensions with zeros.
                x_lifted = np.zeros((*x_array.shape[:-1], state_dim), dtype=np.float32)
                x_lifted[..., : x_array.shape[-1]] = x_array
            else:
                raise ValueError(
                    f"Lyapunov input has invalid shape {x_array.shape}; expected last dim "
                    f"<= {state_dim}."
                )

            x_tensor = th.as_tensor(x_lifted, dtype=th.float32)
            if x_tensor.ndim == 1:
                x_tensor = x_tensor.unsqueeze(0)

            with th.no_grad():
                values = lyap_model(x_tensor).reshape(-1)

            values_np = values.detach().cpu().numpy()
            if x_lifted.ndim == 1:
                return float(values_np[0])
            return values_np

        return _lyapunov_func

    def _assert_lyapunov_plot_written(
        self,
        lyap_model: nn.Module,
        certification_result,
        stem: str,
    ) -> None:
        lyap_func = self._to_numpy_lyapunov(
            lyap_model=lyap_model,
            state_dim=3,
        )
        self._assert_plot_written(
            plot_fn=lyapunov_cert_regions,
            stem=stem,
            plot_kwargs={
                "dataset": None,
                "lyapunov_func": lyap_func,
                "state_labels": ["x0", "x1", "x2"],
                "certification_result": certification_result,
            },
        )

    def test_quadratic_lyapunov_with_identity_dynamics_certifies_all_regions(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=1e-6,
        )
        result = certifier.certify(rho_estimate=1.0)

        self.assertTrue(result.global_success)
        self.assertTrue(result.partial_success)
        self.assertGreaterEqual(result.rho, 1.0)
        self.assertEqual(result.uncertified_regions.shape[0], 0)
        self.assertGreater(result.certified_sublevel_regions.shape[0], 0)
        self._assert_region_plot_written(
            certification_result=result,
            stem="quadratic_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem="quadratic_lyapunov",
        )

    def test_negative_quadratic_produces_counterexamples(self) -> None:
        certifier = self._make_certifier(_NegativeQuadraticLyapunov())
        result = certifier.certify(rho_estimate=1.0)

        self.assertFalse(result.global_success)
        self.assertFalse(result.partial_success)
        self.assertAlmostEqual(result.rho, 1.0, places=6)
        self.assertEqual(result.certified_sublevel_regions.shape[0], 0)
        self.assertGreaterEqual(result.uncertified_regions.shape[0], 0)
        self.assertGreaterEqual(result.outside_sublevel_regions.shape[0], 0)
        self._assert_region_plot_written(
            certification_result=result,
            stem="negative_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem="negative_lyapunov",
        )

    def test_mixed_lyapunov_has_safe_and_unsafe_regions(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_DirectionalScaleDynamics(base_scale=0.8, axis_gain=0.4),
            kappa=1e-6,
            rho_min=0.9,
        )
        result = certifier.certify(rho_estimate=1.0)

        self.assertFalse(result.global_success)
        self.assertTrue(result.partial_success)
        self.assertGreater(result.certified_sublevel_regions.shape[0], 0)
        self.assertGreater(result.uncertified_regions.shape[0], 0)

        certified_centers = result.certified_sublevel_regions.mean(axis=1)
        failed_centers = result.uncertified_regions.mean(axis=1)
        self.assertTrue(np.any(certified_centers[:, 0] < 0.0))
        self.assertTrue(np.any(failed_centers[:, 0] > 0.0))
        self._assert_region_plot_written(
            certification_result=result,
            stem="mixed_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem="mixed_lyapunov",
        )

    def test_is_rho_certified_stops_immediately_on_direct_counterexample(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=1e-6,
        )
        certifier.regions = th.tensor(
            [
                [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
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
                    verified=False,
                    counterexample_found=True,
                    status="unsafe-pgd",
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
            self.assertFalse(certifier.is_rho_certified(rho=1.0))

        self.assertEqual(region_certifier.calls, 2)

    def test_is_rho_certified_keeps_splitting_unknown_regions(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=1e-6,
        )
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
                    verified=False,
                    counterexample_found=False,
                    status="unknown",
                ),
            ]
        )

        def _split_failed_regions(failed_bs: th.Tensor, resolved_bs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
            del resolved_bs
            return failed_bs[:0], failed_bs

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
        ), mock.patch.object(
            certifier.region_manager,
            "split_failed_regions_on_certification_frontier",
            side_effect=_split_failed_regions,
        ) as split_mock:
            self.assertFalse(certifier.is_rho_certified(rho=1.0))

        self.assertEqual(region_certifier.calls, 2)
        split_mock.assert_called_once()
        split_failed_bs, split_resolved_bs = split_mock.call_args.args
        self.assertTrue(th.allclose(split_failed_bs, certifier.regions[1:2]))
        self.assertTrue(th.allclose(split_resolved_bs, certifier.regions[:1]))

    def test_is_rho_certified_marks_all_regions_certified_when_all_are_safe(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=1e-6,
        )
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
                show_progress=False,
                early_exit=True,
            )

        self.assertTrue(result.global_success)
        self.assertEqual(result.resolved.shape[0], 2)
        self.assertEqual(result.unresolved.shape[0], 0)
        self.assertEqual(region_certifier.calls, 2)

    def test_boundary_regions_use_core_before_complete_by_default(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=1e-6,
        )
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
            ]
        )
        core_certifier = _RecordingMockRegionCertifier(
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
            return_value=core_certifier,
        ):
            result = certifier._process_regions(
                certifier.regions,
                rho=1.0,
                early_exit=False,
            )

        self.assertEqual(len(complete_certifier.batches), 1)
        self.assertEqual(len(core_certifier.batches), 1)
        self.assertTrue(th.allclose(core_certifier.batches[0], certifier.regions))
        self.assertTrue(th.allclose(complete_certifier.batches[0], certifier.regions[1:2]))
        self.assertEqual(result.resolved.shape[0], 2)
        self.assertEqual(result.unresolved.shape[0], 0)

    def test_skip_boundary_core_cert_routes_boundary_regions_directly_to_complete(self) -> None:
        config = self.make_config(
            state_dim=3,
            cert_bounds=np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=np.float32),
            kappa=1e-6,
            rho_min=1e-6,
            bins_per_dim=4,
            center_refinement_factor=0.7,
            origin_exclusion=0.0,
            max_scale_steps=6,
            max_bisection_steps=6,
            lirpa_method="alpha-crown",
            condition_tolerance=1e-6,
            max_recursion_depth=3,
            skip_boundary_core_cert=True,
        )
        certifier = self.make_bisect_certifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
        )
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
            result = certifier._process_regions(
                certifier.regions,
                rho=1.0,
                early_exit=False,
            )

        self.assertEqual(len(complete_certifier.batches), 1)
        self.assertTrue(th.allclose(complete_certifier.batches[0], certifier.regions))
        self.assertEqual(result.resolved.shape[0], 2)
        self.assertEqual(result.unresolved.shape[0], 0)

    def test_is_rho_certified_reuses_cached_core_unknown_regions(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=1e-6,
        )
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
            "split_failed_regions_on_certification_frontier",
            side_effect=lambda failed_bs, resolved_bs: (failed_bs[:0], failed_bs),
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

if __name__ == "__main__":
    unittest.main(verbosity=2)
