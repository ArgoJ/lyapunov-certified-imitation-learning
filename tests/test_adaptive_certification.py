import unittest
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn

from lcil.certification.adaptive import (
    AdaptiveABCrownRegionCertifier,
    AdaptiveCertificationConfig,
    AdaptiveCertifier,
    AdaptiveCertificationResult,
    LiRPALyapunovRegionBounds,
    LyapunovRegionBounds,
    RegionBuilder,
)


class _LinearLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return 2.0 * x[:, :1] - x[:, 1:2]


class _IdentityLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x[:, :1]


class _ZeroPolicy(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)


class _IdentityDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return x


def _make_result(
    rho: float,
    certified_inside_regions: th.Tensor,
    boundary_regions: th.Tensor,
    outside_regions: th.Tensor,
    failed_inside_regions: th.Tensor | None = None,
) -> AdaptiveCertificationResult:
    if failed_inside_regions is None:
        failed_inside_regions = certified_inside_regions[:0]

    inside_regions = th.cat([certified_inside_regions, failed_inside_regions], dim=0)
    all_regions = th.cat([inside_regions, boundary_regions, outside_regions], dim=0)
    inside_count = len(inside_regions)
    boundary_count = len(boundary_regions)
    total_count = len(all_regions)

    inside_mask = th.zeros((total_count,), dtype=th.bool)
    boundary_mask = th.zeros((total_count,), dtype=th.bool)
    outside_mask = th.zeros((total_count,), dtype=th.bool)
    inside_mask[:inside_count] = True
    boundary_mask[inside_count:inside_count + boundary_count] = True
    outside_mask[inside_count + boundary_count:] = True

    zero_bounds = LyapunovRegionBounds(
        lower=th.zeros((total_count,), dtype=th.float32),
        upper=th.zeros((total_count,), dtype=th.float32),
    )
    return AdaptiveCertificationResult(
        rho=float(rho),
        sublevel_threshold=float(rho + 1e-6),
        region_bounds=zero_bounds,
        inside_mask=inside_mask,
        boundary_mask=boundary_mask,
        outside_mask=outside_mask,
        inside_regions=inside_regions,
        boundary_regions=boundary_regions,
        outside_regions=outside_regions,
        certified_inside_regions=certified_inside_regions,
        failed_inside_regions=failed_inside_regions,
    )


class TestAdaptiveCertificationConfig(unittest.TestCase):
    def test_normalizes_lirpa_method(self) -> None:
        config = AdaptiveCertificationConfig(
            state_dim=2,
            cert_bounds=np.array([[-1.0, -2.0], [1.0, 2.0]], dtype=np.float32),
            kappa=0.1,
            bins_per_dim=2,
            lirpa_bound_method="  IBP  ",
        )

        self.assertEqual(config.lirpa_bound_method, "ibp")
        self.assertEqual(config.bins_per_dim, (2, 2))

    def test_is_standalone_and_keeps_only_adaptive_fields(self) -> None:
        config = AdaptiveCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            kappa=0.1,
            bins_per_dim=2,
        )

        self.assertFalse(hasattr(config, "rho_min"))
        self.assertFalse(hasattr(config, "max_bisection_steps"))


class TestRegionBuilder(unittest.TestCase):
    def test_build_regions_keeps_root_box_when_origin_exclusion_is_zero(self) -> None:
        splitter = RegionBuilder(
            bounds=th.tensor([[-1.0], [1.0]], dtype=th.float32),
            bins_per_dim=1,
            origin_exclusion=0.0,
        )

        regions = splitter.build_regions()

        self.assertEqual(tuple(regions.shape), (1, 2, 1))
        self.assertTrue(
            th.allclose(
                regions[:, :, 0].cpu(),
                th.tensor([[-1.0, 1.0]], dtype=th.float32),
            )
        )

    def test_build_regions_filters_center_hole_from_uniform_grid(self) -> None:
        splitter = RegionBuilder(
            bounds=th.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=th.float32),
            bins_per_dim=(4, 4),
            center_refinement_factor=1.0,
            origin_exclusion=0.5,
        )

        regions = splitter.build_regions()

        self.assertEqual(tuple(regions.shape), (12, 2, 2))
        lbs = regions[:, 0]
        ubs = regions[:, 1]
        overlaps_origin = ((lbs < 0.5) & (ubs > -0.5)).all(dim=1)
        self.assertFalse(bool(overlaps_origin.any().item()))

    def test_split_regions_splits_each_box_along_its_widest_dimension(self) -> None:
        splitter = RegionBuilder(
            bounds=th.tensor([[-4.0, -4.0], [4.0, 4.0]], dtype=th.float32),
            bins_per_dim=(2, 2),
        )
        regions = th.tensor(
            [
                [[-4.0, -1.0], [2.0, 1.0]],
                [[-1.0, -3.0], [1.0, 3.0]],
            ],
            dtype=th.float32,
        )

        refined = splitter.split_regions(regions)

        expected = th.tensor(
            [
                [[-4.0, -1.0], [-1.0, 1.0]],
                [[-1.0, -3.0], [1.0, 0.0]],
                [[-1.0, -1.0], [2.0, 1.0]],
                [[-1.0, 0.0], [1.0, 3.0]],
            ],
            dtype=th.float32,
        )
        self.assertTrue(th.allclose(refined.cpu(), expected))

    def test_split_region_uses_explicit_dimension_when_provided(self) -> None:
        splitter = RegionBuilder(
            bounds=th.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=th.float32),
            bins_per_dim=(2, 2),
        )
        region = th.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=th.float32)

        refined = splitter.split_region(region, split_dim=1)

        expected = th.tensor(
            [
                [[-2.0, -2.0], [2.0, 0.0]],
                [[-2.0, 0.0], [2.0, 2.0]],
            ],
            dtype=th.float32,
        )
        self.assertTrue(th.allclose(refined.cpu(), expected))


class TestLiRPALyapunovRegionBounds(unittest.TestCase):
    def test_compute_bounds_for_linear_model_is_exact(self) -> None:
        regions = th.tensor(
            [
                [[-1.0, 0.0], [2.0, 3.0]],
                [[0.0, -2.0], [1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        bounder = LiRPALyapunovRegionBounds(
            lyap_model=_LinearLyapunov(),
            state_dim=2,
            batch_size=2,
            default_bound_method="ibp",
        )

        bounds = bounder.compute_bounds_for_regions(regions)

        expected_lower = th.tensor([-5.0, -1.0], dtype=th.float32)
        expected_upper = th.tensor([4.0, 4.0], dtype=th.float32)
        self.assertTrue(th.allclose(bounds.lower.cpu(), expected_lower, atol=1e-5))
        self.assertTrue(th.allclose(bounds.upper.cpu(), expected_upper, atol=1e-5))


class TestAdaptiveCertifier(unittest.TestCase):
    def test_certify_inside_regions_caches_bounds_and_calls_abcrown_only_for_inside_boxes(self) -> None:
        config = AdaptiveCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
            kappa=0.1,
            bins_per_dim=4,
            origin_exclusion=0.0,
            lirpa_bound_method="ibp",
            batch_size=8,
        )

        fake_abcrown = mock.Mock()
        fake_abcrown.setup_backend = mock.Mock()
        fake_abcrown.certify_regions = mock.Mock(
            return_value=th.tensor([True, False], dtype=th.bool)
        )

        with mock.patch(
            "lcil.certification.adaptive.certifier.AdaptiveABCrownRegionCertifier",
            return_value=fake_abcrown,
        ):
            certifier = AdaptiveCertifier(
                policy_model=_ZeroPolicy(),
                lyap_model=_IdentityLyapunov(),
                dyn_model=_IdentityDynamics(),
                config=config,
                device=th.device("cpu"),
            )
            result = certifier.certify_inside_regions(rho=0.5)

        self.assertIsNotNone(certifier.region_bounds)
        self.assertTrue(
            th.allclose(
                certifier.region_bounds.lower.cpu(),
                th.tensor([-2.0, -1.0, 0.0, 1.0], dtype=th.float32),
            )
        )
        self.assertTrue(
            th.allclose(
                certifier.region_bounds.upper.cpu(),
                th.tensor([-1.0, 0.0, 1.0, 2.0], dtype=th.float32),
            )
        )

        fake_abcrown.setup_backend.assert_called_once_with()
        fake_abcrown.certify_regions.assert_called_once()
        forwarded_regions = fake_abcrown.certify_regions.call_args.args[0]
        forwarded_rho = fake_abcrown.certify_regions.call_args.args[1]
        forwarded_early_exit = fake_abcrown.certify_regions.call_args.kwargs["early_exit"]

        self.assertEqual(float(forwarded_rho), 0.5)
        self.assertFalse(forwarded_early_exit)
        self.assertEqual(tuple(forwarded_regions.shape), (2, 2, 1))
        self.assertTrue(
            th.allclose(
                forwarded_regions[:, :, 0].cpu(),
                th.tensor([[-2.0, -1.0], [-1.0, 0.0]], dtype=th.float32),
            )
        )

        self.assertEqual(tuple(result.inside_regions.shape), (2, 2, 1))
        self.assertEqual(tuple(result.boundary_regions.shape), (1, 2, 1))
        self.assertEqual(tuple(result.outside_regions.shape), (1, 2, 1))
        self.assertEqual(tuple(result.certified_inside_regions.shape), (1, 2, 1))
        self.assertEqual(tuple(result.failed_inside_regions.shape), (1, 2, 1))
        self.assertFalse(result.global_success)
        self.assertTrue(result.partial_success)

    def test_certify_inside_regions_reuses_previously_certified_inside_boxes_only_for_larger_rho(self) -> None:
        config = AdaptiveCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
            kappa=0.1,
            bins_per_dim=4,
            origin_exclusion=0.0,
            lirpa_bound_method="ibp",
            batch_size=8,
        )

        fake_abcrown = mock.Mock()
        fake_abcrown.setup_backend = mock.Mock()
        fake_abcrown.certify_regions = mock.Mock(
            side_effect=[
                th.tensor([True, False], dtype=th.bool),
                th.tensor([True, True], dtype=th.bool),
            ]
        )

        with mock.patch(
            "lcil.certification.adaptive.certifier.AdaptiveABCrownRegionCertifier",
            return_value=fake_abcrown,
        ):
            certifier = AdaptiveCertifier(
                policy_model=_ZeroPolicy(),
                lyap_model=_IdentityLyapunov(),
                dyn_model=_IdentityDynamics(),
                config=config,
                device=th.device("cpu"),
            )
            first_result = certifier.certify_inside_regions(rho=0.5)
            second_result = certifier.certify_inside_regions(rho=1.5)

        self.assertEqual(fake_abcrown.certify_regions.call_count, 2)

        first_forwarded_regions = fake_abcrown.certify_regions.call_args_list[0].args[0]
        second_forwarded_regions = fake_abcrown.certify_regions.call_args_list[1].args[0]

        self.assertEqual(tuple(first_forwarded_regions.shape), (2, 2, 1))
        self.assertTrue(
            th.allclose(
                first_forwarded_regions[:, :, 0].cpu(),
                th.tensor([[-2.0, -1.0], [-1.0, 0.0]], dtype=th.float32),
            )
        )
        self.assertEqual(tuple(second_forwarded_regions.shape), (2, 2, 1))
        self.assertTrue(
            th.allclose(
                second_forwarded_regions[:, :, 0].cpu(),
                th.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=th.float32),
            )
        )

        self.assertEqual(tuple(first_result.certified_inside_regions.shape), (1, 2, 1))
        self.assertEqual(tuple(second_result.certified_inside_regions.shape), (3, 2, 1))
        self.assertTrue(
            th.allclose(
                second_result.certified_inside_regions[:, :, 0].cpu(),
                th.tensor([[-2.0, -1.0], [-1.0, 0.0], [0.0, 1.0]], dtype=th.float32),
            )
        )
        self.assertEqual(tuple(second_result.failed_inside_regions.shape), (0, 2, 1))

    def test_certify_inside_regions_does_not_reuse_for_equal_rho(self) -> None:
        config = AdaptiveCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
            kappa=0.1,
            bins_per_dim=4,
            origin_exclusion=0.0,
            lirpa_bound_method="ibp",
            batch_size=8,
        )

        fake_abcrown = mock.Mock()
        fake_abcrown.setup_backend = mock.Mock()
        fake_abcrown.certify_regions = mock.Mock(
            side_effect=[
                th.tensor([True, False], dtype=th.bool),
                th.tensor([True, True], dtype=th.bool),
            ]
        )

        with mock.patch(
            "lcil.certification.adaptive.certifier.AdaptiveABCrownRegionCertifier",
            return_value=fake_abcrown,
        ):
            certifier = AdaptiveCertifier(
                policy_model=_ZeroPolicy(),
                lyap_model=_IdentityLyapunov(),
                dyn_model=_IdentityDynamics(),
                config=config,
                device=th.device("cpu"),
            )
            certifier.certify_inside_regions(rho=0.5)
            second_result = certifier.certify_inside_regions(rho=0.5)

        self.assertEqual(fake_abcrown.certify_regions.call_count, 2)
        second_forwarded_regions = fake_abcrown.certify_regions.call_args_list[1].args[0]
        self.assertEqual(tuple(second_forwarded_regions.shape), (2, 2, 1))
        self.assertTrue(
            th.allclose(
                second_forwarded_regions[:, :, 0].cpu(),
                th.tensor([[-2.0, -1.0], [-1.0, 0.0]], dtype=th.float32),
            )
        )
        self.assertEqual(tuple(second_result.certified_inside_regions.shape), (2, 2, 1))

    def test_pareto_curve_refines_until_unresolved_ratio_meets_tolerance(self) -> None:
        config = AdaptiveCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
            kappa=0.1,
            bins_per_dim=4,
            origin_exclusion=0.0,
            lirpa_bound_method="ibp",
            batch_size=8,
        )
        certifier = AdaptiveCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_IdentityLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            device=th.device("cpu"),
        )

        first_result = _make_result(
            rho=0.5,
            certified_inside_regions=th.tensor([[[-2.0], [-1.0]]], dtype=th.float32),
            boundary_regions=th.tensor([[[-1.0], [0.0]]], dtype=th.float32),
            outside_regions=th.tensor([[[0.0], [2.0]]], dtype=th.float32),
        )
        second_result = _make_result(
            rho=0.5,
            certified_inside_regions=th.tensor(
                [[[-2.0], [-1.0]], [[-1.0], [-0.5]]],
                dtype=th.float32,
            ),
            boundary_regions=th.empty((0, 2, 1), dtype=th.float32),
            outside_regions=th.tensor([[[ -0.5], [2.0]]], dtype=th.float32),
        )

        with mock.patch.object(
            certifier,
            "certify_inside_regions",
            side_effect=[first_result, second_result],
        ) as certify_mock:
            points = certifier.pareto_curve(
                rho_values=[0.5],
                unresolved_tolerance=0.25,
                max_refinement_rounds=1,
            )

        self.assertEqual(certify_mock.call_count, 2)
        self.assertEqual(len(points), 1)
        point = points[0]
        self.assertEqual(point.refinement_rounds, 1)
        self.assertTrue(point.feasible)
        self.assertAlmostEqual(point.volumes.certified_volume, 1.5)
        self.assertAlmostEqual(point.volumes.unresolved_ratio, 0.0)

    def test_certify_selects_largest_feasible_rho_from_full_curve(self) -> None:
        config = AdaptiveCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
            kappa=0.1,
            bins_per_dim=4,
            origin_exclusion=0.0,
            lirpa_bound_method="ibp",
            batch_size=8,
        )
        certifier = AdaptiveCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_IdentityLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            device=th.device("cpu"),
        )

        results = [
            _make_result(
                rho=0.5,
                certified_inside_regions=th.tensor([[[-2.0], [-1.0]]], dtype=th.float32),
                boundary_regions=th.empty((0, 2, 1), dtype=th.float32),
                outside_regions=th.tensor([[[-1.0], [2.0]]], dtype=th.float32),
            ),
            _make_result(
                rho=1.0,
                certified_inside_regions=th.tensor([[[-2.0], [-1.0]]], dtype=th.float32),
                boundary_regions=th.tensor([[[-1.0], [1.0]]], dtype=th.float32),
                outside_regions=th.tensor([[[1.0], [2.0]]], dtype=th.float32),
            ),
            _make_result(
                rho=1.5,
                certified_inside_regions=th.tensor(
                    [[[-2.0], [-1.0]], [[-1.0], [1.0]]],
                    dtype=th.float32,
                ),
                boundary_regions=th.tensor([[[1.0], [2.0]]], dtype=th.float32),
                outside_regions=th.empty((0, 2, 1), dtype=th.float32),
            ),
        ]

        with mock.patch.object(
            certifier,
            "certify_inside_regions",
            side_effect=results,
        ) as certify_mock:
            certify_result = certifier.certify(
                rho_values=[0.5, 1.0, 1.5],
                unresolved_tolerance=0.4,
                max_refinement_rounds=0,
            )

        self.assertEqual(certify_mock.call_count, 3)
        self.assertEqual(len(certify_result.pareto_points), 3)
        self.assertAlmostEqual(certify_result.best_rho, 1.5)
        self.assertIsNotNone(certify_result.best_point)
        self.assertAlmostEqual(certify_result.pareto_points[0].volumes.unresolved_ratio, 0.0)
        self.assertGreater(certify_result.pareto_points[1].volumes.unresolved_ratio, 0.4)
        self.assertLessEqual(certify_result.pareto_points[2].volumes.unresolved_ratio, 0.4)


class TestAdaptiveABCrownRegionCertifier(unittest.TestCase):
    def test_certify_regions_processes_each_region_individually(self) -> None:
        config = AdaptiveCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
            kappa=0.1,
            bins_per_dim=2,
            batch_size=4,
        )
        regions = th.tensor(
            [
                [[-2.0], [-1.0]],
                [[-1.0], [0.0]],
                [[0.0], [1.0]],
            ],
            dtype=th.float32,
        )
        certifier = AdaptiveABCrownRegionCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_IdentityLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            device=th.device("cpu"),
        )

        with mock.patch.object(
            certifier,
            "certify_region",
            side_effect=[True, False, True],
        ) as certify_region_mock:
            is_certified = certifier.certify_regions(regions, rho=0.25, early_exit=False)

        self.assertTrue(
            th.equal(is_certified.cpu(), th.tensor([True, False, True], dtype=th.bool))
        )
        self.assertEqual(certify_region_mock.call_count, 3)
        self.assertTrue(th.allclose(certify_region_mock.call_args_list[0].args[0], regions[0]))
        self.assertTrue(th.allclose(certify_region_mock.call_args_list[1].args[0], regions[1]))
        self.assertTrue(th.allclose(certify_region_mock.call_args_list[2].args[0], regions[2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)