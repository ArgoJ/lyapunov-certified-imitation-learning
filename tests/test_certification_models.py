import unittest

import torch as th

from shared_utils import (
    _DescendingLinearLyapunov,
    _ShiftDynamics,
    _ZeroPolicy,
)
from certification_mock_common import (
    LyapunovCoreVerifier,
)


class TestCertificationModels(unittest.TestCase):
    def test_core_verifier_returns_condition_v_and_xnext(self) -> None:
        verifier = LyapunovCoreVerifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_DescendingLinearLyapunov(),
            dyn_model=_ShiftDynamics(shift=2.0),
            kappa=1e-6,
        )

        outputs = verifier(
            th.tensor([[1.0]], dtype=th.float32),
        )

        self.assertEqual(outputs.shape, (1, 3))
        self.assertAlmostEqual(float(outputs[0, 0]), 2.0 - 1e-6, places=5)
        self.assertAlmostEqual(float(outputs[0, 1]), 1.0, places=5)
        self.assertAlmostEqual(float(outputs[0, 2]), 3.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)