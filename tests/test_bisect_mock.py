import importlib
import unittest
from dataclasses import dataclass
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn
from plot_assertions_mixin import PlotAssertionsMixin

from certification_mock_common import (
    BisectCertifier,
    CertificationMockedABCrownTestCase,
    LyapunovCertificationConfig,
    LyapunovMultiOutputVerifier,
    LyapunovRegionBounds,
    LyapunovVerifier,
    RecursiveCertificationResult,
    _DescendingLinearLyapunov,
    _DirectionalScaleDynamics,
    _IdentityDynamics,
    _MixedLyapunov,
    _NegativeQuadraticLyapunov,
    _QuadraticLyapunov,
    _ShiftDynamics,
    _ZeroDynamics,
    _ZeroPolicy,
)

plot_module = importlib.import_module("lcil.utils.plot")
certified_regions_2d = plot_module.certified_regions_2d
lyapunov_cert_regions = plot_module.lyapunov_cert_regions


class _StaticRegionBounder:
    def __init__(self, lower: th.Tensor, upper: th.Tensor):
        self.bounds = LyapunovRegionBounds(lower=lower, upper=upper)

    def compute_bounds_for_regions(self, regions: th.Tensor, *, method: str | None = None):
        del regions, method
        return self.bounds


class _RecursiveMockCertifier(BisectCertifier):
    def __init__(self, *args, width_limit: float, rho_limit: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.width_limit = float(width_limit)
        self.rho_limit = float(rho_limit)
        self.certify_calls: list[tuple[int, bool]] = []

    def setup_backend(self, *args, **kwargs) -> None:
        del args, kwargs

    def _partition_regions_by_sublevel(
        self,
        bs: th.Tensor,
        rho: float,
    ) -> tuple[th.Tensor, th.Tensor]:
        del rho
        return bs, bs[:0]

    def _certify_batched_regions(
        self,
        bs: th.Tensor,
        rho: float,
        early_exit: bool = True,
        *args,
        **kwargs,
    ) -> th.Tensor:
        del args, kwargs
        self.certify_calls.append((len(bs), early_exit))
        widths = (bs[:, 1] - bs[:, 0]).amax(dim=1)
        return (widths <= self.width_limit) & (rho <= self.rho_limit)


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
            cert_method="alpha-crown",
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
            kappa=0.0,
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
            kappa=0.0,
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
            kappa=0.0,
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

        with mock.patch.object(
            certifier,
            "_partition_regions_by_sublevel",
            return_value=(certifier.regions, certifier.regions[:0]),
        ), mock.patch.object(
            certifier,
            "_get_region_certifier",
            return_value=region_certifier,
        ), mock.patch.object(
            certifier,
            "_split_failed_regions_on_certification_frontier",
            side_effect=AssertionError("splitting should not happen after a direct counterexample"),
        ):
            self.assertFalse(certifier.is_rho_certified(rho=1.0))

        self.assertEqual(region_certifier.calls, 2)

    def test_is_rho_certified_keeps_splitting_unknown_regions(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=0.0,
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

        with mock.patch.object(
            certifier,
            "_partition_regions_by_sublevel",
            return_value=(certifier.regions, certifier.regions[:0]),
        ), mock.patch.object(
            certifier,
            "_get_region_certifier",
            return_value=region_certifier,
        ), mock.patch.object(
            certifier,
            "_split_failed_regions_on_certification_frontier",
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
            kappa=0.0,
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

        with mock.patch.object(
            certifier,
            "_partition_regions_by_sublevel",
            return_value=(certifier.regions, certifier.regions[:0]),
        ), mock.patch.object(
            certifier,
            "_get_region_certifier",
            return_value=region_certifier,
        ), mock.patch.object(
            certifier,
            "_split_failed_regions_on_certification_frontier",
            side_effect=AssertionError("splitting should not happen when all regions are safe"),
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

    def test_negative_values_are_not_hidden_by_origin_guard(self) -> None:
        certification_verifier = LyapunovVerifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_NegativeQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            lbx=th.tensor([[-2.0]], dtype=th.float32),
            ubx=th.tensor([[2.0]], dtype=th.float32),
            kappa=0.1,
            sublevel_tolerance=1e-6,
        )

        x = th.tensor([[1.0]], dtype=th.float32)
        rho = th.tensor([[2.0]], dtype=th.float32)

        self.assertGreater(float(certification_verifier(x, rho).item()), 0.0)

    def test_certification_verifier_enforces_hard_invariance(self) -> None:
        certification_verifier = LyapunovVerifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_DescendingLinearLyapunov(),
            dyn_model=_ShiftDynamics(shift=2.0),
            lbx=th.tensor([[0.0]], dtype=th.float32),
            ubx=th.tensor([[2.0]], dtype=th.float32),
            kappa=0.0,
            sublevel_tolerance=1e-6,
        )

        x = th.tensor([[1.0]], dtype=th.float32)
        rho = th.tensor([[2.0]], dtype=th.float32)

        self.assertGreater(float(certification_verifier(x, rho).item()), 0.0)

    def test_certification_verifier_checks_levelset_boundary_conservatively(self) -> None:
        certification_verifier = LyapunovVerifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_DescendingLinearLyapunov(),
            dyn_model=_ShiftDynamics(shift=3.0),
            lbx=th.tensor([[0.0]], dtype=th.float32),
            ubx=th.tensor([[2.0]], dtype=th.float32),
            kappa=0.0,
            sublevel_tolerance=1e-6,
        )

        x = th.tensor([[0.0]], dtype=th.float32)
        rho = th.tensor([[2.0]], dtype=th.float32)

        self.assertGreater(float(certification_verifier(x, rho).item()), 0.0)

    def test_bisect_multi_output_verifier_returns_condition_v_and_xnext(self) -> None:
        verifier = LyapunovMultiOutputVerifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_DescendingLinearLyapunov(),
            dyn_model=_ShiftDynamics(shift=2.0),
            lbx=th.tensor([[0.0]], dtype=th.float32),
            ubx=th.tensor([[2.0]], dtype=th.float32),
            kappa=0.0,
            sublevel_tolerance=1e-6,
        )

        outputs = verifier(
            th.tensor([[1.0]], dtype=th.float32),
            th.tensor([[2.0]], dtype=th.float32),
        )

        self.assertEqual(outputs.shape, (1, 3))
        self.assertAlmostEqual(float(outputs[0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(outputs[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(outputs[0, 2]), 3.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
