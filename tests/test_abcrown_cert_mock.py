import unittest
from types import SimpleNamespace
from unittest import mock

import torch as th

from lcil.certification.abcrown_region_certifier import _is_safe_status

from shared_utils import (
    _IdentityDynamics,
    _QuadraticLyapunov,
    _ShiftDynamics
)
from certification_mock_common import (
    CertificationMockedABCrownTestCase,
    _FakeOutputVars,
)


class TestABCrownRegionCertifierMock(CertificationMockedABCrownTestCase):
    def test_is_verified_status_accepts_verified_and_safe_prefixes(self) -> None:
        self.assertTrue(_is_safe_status("verified"))
        self.assertTrue(_is_safe_status(" SAFE "))
        self.assertTrue(_is_safe_status("safe-incomplete"))
        self.assertFalse(_is_safe_status("unsafe"))

    def test_build_safe_output_constraint_respects_tolerances(self) -> None:
        certifier = self.make_abcrown_region_certifier(
            state_dim=1,
            cert_bounds=[[-1.0], [1.0]],
            sublevel_tolerance=0.2,
            condition_tolerance=0.1,
            batch_size=8,
        )

        y = _FakeOutputVars(3)
        constraint = certifier._build_safe_output_constraint(y=y, rho=1.0)
        values = th.tensor(
            [
                [999.0, 1.25, 10.0],
                [-0.05, 0.05, 1.05],
                [0.0, -0.11, 0.0],
                [-0.11, 0.05, 0.0],
                [0.0, 0.05, 1.2],
            ],
            dtype=th.float32,
        )

        safe_mask = constraint.evaluate(values)

        self.assertTrue(
            th.equal(
                safe_mask,
                th.tensor([True, True, False, False, False], dtype=th.bool),
            )
        )

    def test_certify_region_rejects_invalid_shapes(self) -> None:
        certifier = self.make_abcrown_region_certifier(state_dim=1)

        with self.assertRaisesRegex(ValueError, "region must have shape"):
            certifier.verify_region(th.zeros((1, 2, 1), dtype=th.float32), rho=1.0)

    def test_verify_region_returns_verified_for_safe_region(self) -> None:
        certifier = self.make_abcrown_region_certifier(
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            state_dim=1,
            cert_bounds=[[-2.0], [2.0]],
            kappa=0.0,
            batch_size=8,
        )
        region = th.tensor([[-0.5], [0.5]], dtype=th.float32)

        verification = certifier.verify_region(region, rho=1.0)

        self.assertTrue(verification.verified)
        self.assertFalse(verification.counterexample_found)
        self.assertIsNotNone(certifier.verifier)

    def test_verify_region_returns_counterexample_when_successor_leaves_global_bounds(self) -> None:
        certifier = self.make_abcrown_region_certifier(
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_ShiftDynamics(shift=2.0),
            state_dim=1,
            cert_bounds=[[0.0], [2.0]],
            kappa=0.0,
            batch_size=8,
        )
        region = th.tensor([[1.0], [1.5]], dtype=th.float32)

        verification = certifier.verify_region(region, rho=10.0)

        self.assertFalse(verification.verified)
        self.assertTrue(verification.counterexample_found)
        self.assertIsNotNone(certifier.verifier)

    def test_certify_regions_honors_early_exit(self) -> None:
        certifier = self.make_abcrown_region_certifier(state_dim=1)
        regions = th.tensor(
            [
                [[-2.0], [-1.0]],
                [[-1.0], [0.0]],
                [[0.0], [1.0]],
            ],
            dtype=th.float32,
        )

        with mock.patch.object(
            certifier,
            "verify_region",
            side_effect=[
                SimpleNamespace(verified=True, counterexample_found=False),
                SimpleNamespace(verified=False, counterexample_found=True),
                SimpleNamespace(verified=True, counterexample_found=False),
            ],
        ) as verify_region_mock:
            batch_result = certifier.certify_regions(regions, rho=0.25, early_exit=True)

        self.assertTrue(
            th.equal(
                batch_result.verified_mask.cpu(),
                th.tensor([True, False, False], dtype=th.bool),
            )
        )
        self.assertTrue(
            th.equal(
                batch_result.counterexample_mask.cpu(),
                th.tensor([False, True, False], dtype=th.bool),
            )
        )
        self.assertTrue(
            th.equal(
                batch_result.unknown_mask.cpu(),
                th.tensor([False, False, False], dtype=th.bool),
            )
        )
        self.assertEqual(verify_region_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)