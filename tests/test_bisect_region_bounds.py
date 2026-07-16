import unittest
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn

from shared_utils import (
    _IdentityDynamics,
    _QuadraticLyapunov,
    _ZeroDynamics,
    _ZeroPolicy,
)
from certification_mock_common import (
    BisectCertifier,
    CertificationMockedABCrownTestCase,
    LyapunovRegionBounds,
)


class _StaticRegionBounder:
    def __init__(self, lower: th.Tensor, upper: th.Tensor):
        self.bounds = LyapunovRegionBounds(lower=lower, upper=upper)
        self.calls = 0

    def compute_bounds_for_regions(self, regions: th.Tensor, *, method: str | None = None):
        del regions, method
        self.calls += 1
        return self.bounds


class TestBisectRegionBounds(CertificationMockedABCrownTestCase):
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

    def test_partition_regions_by_sublevel_reuses_cached_root_region_bounds(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=1e-6,
        )
        regions = th.tensor(
            [
                [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        certifier.regions = regions
        bounder = _StaticRegionBounder(
            lower=th.tensor([0.75, 2.0], dtype=th.float32),
            upper=th.tensor([1.25, 3.0], dtype=th.float32),
        )

        with mock.patch.object(
            certifier,
            "_get_region_bounder",
            return_value=bounder,
        ):
            cached_regions, cached_bounds = certifier._ensure_region_bounds(
                certifier.regions,
                make_current=True,
            )
            partition_high = certifier.region_manager.partition_certification_regions(
                cached_regions,
                region_bounds=cached_bounds,
                rho=1.5,
                sublevel_tolerance=certifier.config.sublevel_tolerance,
            )
            partition_low = certifier.region_manager.partition_certification_regions(
                cached_regions,
                region_bounds=cached_bounds,
                rho=0.5,
                sublevel_tolerance=certifier.config.sublevel_tolerance,
            )

        self.assertEqual(bounder.calls, 1)
        self.assertIsNotNone(certifier.region_manager.region_bounds)
        self.assertEqual(partition_high.inside_core_unchecked_regions.shape[0], 1)
        self.assertEqual(partition_high.irrelevant_regions.shape[0], 1)
        self.assertFalse(partition_low.has_relevant_regions)
        self.assertEqual(partition_low.irrelevant_regions.shape[0], 2)

    def test_partition_regions_by_sublevel_reuses_cached_child_region_bounds(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=1e-6,
        )
        children = th.tensor(
            [
                [[-1.0, -1.0, -1.0], [-0.5, -0.5, -0.5]],
                [[-0.5, -0.5, -0.5], [0.0, 0.0, 0.0]],
            ],
            dtype=th.float32,
        )
        bounder = _StaticRegionBounder(
            lower=th.tensor([0.6, 1.6], dtype=th.float32),
            upper=th.tensor([0.8, 2.1], dtype=th.float32),
        )

        with mock.patch.object(
            certifier,
            "_get_region_bounder",
            return_value=bounder,
        ):
            cached_children, cached_bounds = certifier._ensure_region_bounds(
                children,
                make_current=False,
            )
            partition_first = certifier.region_manager.partition_certification_regions(
                cached_children,
                region_bounds=cached_bounds,
                rho=1.0,
                sublevel_tolerance=certifier.config.sublevel_tolerance,
            )
            partition_second = certifier.region_manager.partition_certification_regions(
                children.clone(),
                region_bounds=cached_bounds,
                rho=0.5,
                sublevel_tolerance=certifier.config.sublevel_tolerance,
            )

        self.assertEqual(bounder.calls, 1)
        self.assertEqual(partition_first.inside_core_unchecked_regions.shape[0], 1)
        self.assertEqual(partition_first.irrelevant_regions.shape[0], 1)
        self.assertFalse(partition_second.has_relevant_regions)
        self.assertEqual(partition_second.irrelevant_regions.shape[0], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)