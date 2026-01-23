import unittest

import numpy as np

from lyapunov_certified_imitation_learning.data_generation.verification.certification import (
    certify_linear_mpc_grune_no_terminal,
)


class TestGruneNoTerminalCertificate(unittest.TestCase):
    def test_passes_for_stable_system(self) -> None:
        # Stable A, simple quadratic cost; should pass for small horizons.
        A = np.array([[0.9]])
        B = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[1.0]])

        rep = certify_linear_mpc_grune_no_terminal(
            A=A,
            B=B,
            Q=Q,
            R=R,
            N=5,
            x_bounds=None,
            u_bounds=None,
        )

        self.assertTrue(rep.applicability)
        self.assertTrue(rep.is_stable)
        self.assertTrue(np.isfinite(rep.gamma))
        self.assertTrue(rep.gamma > 0.0)

    def test_fails_for_too_short_horizon_on_unstable_system(self) -> None:
        # Unstable A; short horizons should fail the Grüne threshold.
        A = np.array([[1.5]])
        B = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[1.0]])

        rep_short = certify_linear_mpc_grune_no_terminal(
            A=A,
            B=B,
            Q=Q,
            R=R,
            N=2,
            x_bounds=None,
            u_bounds=None,
        )

        self.assertTrue(rep_short.applicability)
        self.assertFalse(rep_short.is_stable)

        rep_long = certify_linear_mpc_grune_no_terminal(
            A=A,
            B=B,
            Q=Q,
            R=R,
            N=50,
            x_bounds=None,
            u_bounds=None,
        )

        self.assertTrue(rep_long.applicability)
        self.assertTrue(rep_long.is_stable)


if __name__ == "__main__":
    unittest.main()
