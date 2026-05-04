import unittest
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn

from certification_mock_common import (
    BisectCertifier,
    CertificationMockedABCrownTestCase,
    LyapunovRegionBounds,
    _IdentityDynamics,
    _QuadraticLyapunov,
    _ZeroDynamics,
    _ZeroPolicy,
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

    def test_partition_regions_by_sublevel_reuses_cached_root_region_bounds(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=0.0,
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
            relevant_high, irrelevant_high = certifier._partition_regions_by_sublevel(
                regions,
                rho=1.5,
            )
            relevant_low, irrelevant_low = certifier._partition_regions_by_sublevel(
                regions,
                rho=0.5,
            )

        self.assertEqual(bounder.calls, 1)
        self.assertIsNotNone(certifier.region_manager.region_bounds)
        self.assertEqual(relevant_high.shape[0], 1)
        self.assertEqual(irrelevant_high.shape[0], 1)
        self.assertEqual(relevant_low.shape[0], 0)
        self.assertEqual(irrelevant_low.shape[0], 2)

    def test_partition_regions_by_sublevel_reuses_cached_child_region_bounds(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=0.0,
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
            relevant_first, irrelevant_first = certifier._partition_regions_by_sublevel(
                children,
                rho=1.0,
            )
            relevant_second, irrelevant_second = certifier._partition_regions_by_sublevel(
                children.clone(),
                rho=0.5,
            )

        self.assertEqual(bounder.calls, 1)
        self.assertEqual(relevant_first.shape[0], 1)
        self.assertEqual(irrelevant_first.shape[0], 1)
        self.assertEqual(relevant_second.shape[0], 0)
        self.assertEqual(irrelevant_second.shape[0], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)