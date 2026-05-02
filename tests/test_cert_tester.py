import unittest

import numpy as np
import torch as th
import torch.nn as nn

from lcil.certification.bisect_certifier import RegionCertificationResult
from lcil.certification.cert_tester import CertificationResultTester
from lcil.certification.config import LyapunovCertificationConfig


class _ZeroPolicy(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)


class _IdentityLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x[:, :1]


class _DoubleDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return 2.0 * x


class TestCertificationResultTester(unittest.TestCase):
    @staticmethod
    def _make_config(*, bounds: list[list[float]] | None = None) -> LyapunovCertificationConfig:
        cert_bounds = np.array([[0.0], [10.0]], dtype=np.float32) if bounds is None else np.array(bounds, dtype=np.float32)
        return LyapunovCertificationConfig(
            state_dim=1,
            cert_bounds=cert_bounds,
            kappa=0.0,
            rho_min=1e-6,
            bins_per_dim=1,
            sublevel_tolerance=1e-6,
            condition_tolerance=1e-6,
        )

    @classmethod
    def _make_tester(cls, *, bounds: list[list[float]] | None = None) -> CertificationResultTester:
        return CertificationResultTester(
            policy_model=_ZeroPolicy(),
            lyap_model=_IdentityLyapunov(),
            dyn_model=_DoubleDynamics(),
            config=cls._make_config(bounds=bounds),
            device=th.device("cpu"),
        )

    def test_evaluate_regions_returns_empty_summary_for_missing_regions(self) -> None:
        tester = self._make_tester()

        result_none = tester._evaluate_regions(None, name="demo", tolerance=1e-6, rollout_steps=1)
        result_empty = tester._evaluate_regions(
            np.empty((0, 2, 1), dtype=np.float32),
            name="demo",
            tolerance=1e-6,
            rollout_steps=1,
        )

        for result in (result_none, result_empty):
            self.assertEqual(result.num_regions, 0)
            self.assertEqual(result.violation_rate, 0.0)
            self.assertEqual(result.max_violation, 0.0)
            self.assertIsNone(result.violations_per_step)

    def test_evaluate_regions_uses_box_centers_for_rollout_checks(self) -> None:
        tester = self._make_tester()
        regions = np.array([[[1.0], [3.0]]], dtype=np.float32)

        result = tester._evaluate_regions(
            regions,
            name="centered",
            tolerance=1e-6,
            rollout_steps=1,
        )

        self.assertEqual(result.num_regions, 1)
        self.assertEqual(result.violation_rate, 1.0)
        self.assertAlmostEqual(result.max_violation, 2.0, places=6)
        np.testing.assert_allclose(result.violations_per_step, np.array([[2.0]], dtype=np.float32))

    def test_test_result_does_not_hide_outside_sublevel_hard_violations(self) -> None:
        tester = self._make_tester()
        cert_result = RegionCertificationResult(
            global_success=False,
            partial_success=False,
            rho=1.0,
            certified_regions=np.empty((0, 2, 1), dtype=np.float32),
            failed_regions=np.empty((0, 2, 1), dtype=np.float32),
            outside_sublevel_regions=np.array([[[1.5], [1.7]]], dtype=np.float32),
        )

        result = tester.test_result(cert_result, rollout_steps=1)

        self.assertEqual(result.outside_sublevel.num_regions, 1)
        self.assertEqual(result.outside_sublevel.violation_rate, 1.0)
        self.assertAlmostEqual(result.outside_sublevel.max_violation, 1.6, places=6)
        np.testing.assert_allclose(
            result.outside_sublevel.violations_per_step,
            np.array([[1.6]], dtype=np.float32),
        )

    def test_rollout_steps_must_be_positive(self) -> None:
        tester = self._make_tester()
        regions = np.array([[[1.0], [3.0]]], dtype=np.float32)
        cert_result = RegionCertificationResult(
            global_success=False,
            partial_success=False,
            rho=1.0,
            certified_regions=regions,
            failed_regions=np.empty((0, 2, 1), dtype=np.float32),
            outside_sublevel_regions=np.empty((0, 2, 1), dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "rollout_steps must be positive"):
            tester._evaluate_regions(regions, name="demo", tolerance=1e-6, rollout_steps=0)

        with self.assertRaisesRegex(ValueError, "rollout_steps must be positive"):
            tester.test_result(cert_result, rollout_steps=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)