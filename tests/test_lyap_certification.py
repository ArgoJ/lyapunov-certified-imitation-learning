import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are


# Ensure we can import from ./src without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from lyapunov_certified_imitation_learning.data_generation.verification.certification import (  # noqa: E402
    certify_linear_mpc_terminal_ingredients,
)


def _toy_system():
    """Small controllable discrete-time system with a nontrivial box invariance cone."""
    A = np.array(
        [
            [0.67772127, -0.27625594],
            [-0.55083177, 0.21239573],
        ],
        dtype=float,
    )
    B = np.eye(2)
    Q = np.eye(2)
    R = 0.1 * np.eye(2)
    P = solve_discrete_are(A, B, Q, R)
    return A, B, Q, R, P


class TestMpcCertification(unittest.TestCase):
    def test_small_terminal_box_certifies(self):
        A, B, Q, R, P = _toy_system()

        # Symmetric bounds around (x*, u*) = (0, 0)
        x_star = np.zeros(2)
        u_star = np.zeros(2)

        # Pick a "small" terminal box that satisfies |Acl| h <= h.
        # (Any positive scaling of this vector preserves the condition.)
        h_small = 0.01 * np.array([17.90939672, 44.77363032])

        term_bounds = (-h_small, h_small)

        # Make state/input constraints non-binding for this test.
        x_bounds = (-1e6 * np.ones(2), 1e6 * np.ones(2))
        u_bounds = (-1e6 * np.ones(2), 1e6 * np.ones(2))

        rep = certify_linear_mpc_terminal_ingredients(
            A=A,
            B=B,
            Q=Q,
            R=R,
            P=P,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            terminal_x_bounds=term_bounds,
            x_star=x_star,
            u_star=u_star,
        )

        self.assertTrue(rep.applicability)
        self.assertTrue(rep.is_stable)
        self.assertGreaterEqual(rep.invariance_margin, -1e-10)

    def test_tight_input_bounds_fails(self):
        A, B, Q, R, P = _toy_system()

        x_star = np.zeros(2)
        u_star = np.zeros(2)

        # Use the same terminal box as in the certified case.
        h_small = 0.01 * np.array([17.90939672, 44.77363032])
        term_bounds = (-h_small, h_small)
        x_bounds = (-1e6 * np.ones(2), 1e6 * np.ones(2))
        # Zero-width symmetric input bounds around u*=0 forces u \equiv 0.
        # Since the LQR feedback is nontrivial, there is no rho>0 that can satisfy
        # |u|<=0 on an ellipsoid neighborhood.
        u_bounds = (np.zeros(2), np.zeros(2))

        rep = certify_linear_mpc_terminal_ingredients(
            A=A,
            B=B,
            Q=Q,
            R=R,
            P=P,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            terminal_x_bounds=term_bounds,
            x_star=x_star,
            u_star=u_star,
        )

        self.assertTrue(rep.applicability)
        self.assertFalse(rep.is_stable)
        self.assertIn("rho>0", rep.message)
