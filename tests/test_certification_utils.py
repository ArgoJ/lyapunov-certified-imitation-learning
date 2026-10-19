import unittest

import numpy as np
import torch as th

from lcil.certification import compute_blended_lqr_c0_c1
from lcil.imitation_learning import BlendedLQRPolicy, BoundedPolicy, LQRController
from lcil.utils import MLP


class TestCertificationUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.P_2d = th.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=th.float64)
        self.r_2d = (0.1, 0.2)

    def test_2d_analytical_value(self) -> None:
        # For P = [[2, 0.5], [0.5, 1]], r = (0.1, 0.2)
        # v1 = (0.1, 0.2) -> 2*0.01 + 0.04 + 2*0.5*0.1*0.2 = 0.02 + 0.04 + 0.02 = 0.08
        # v2 = (0.1, -0.2) -> 2*0.01 + 0.04 - 0.02 = 0.04
        # max on box is 0.08. Double box is 4 * 0.08 = 0.32
        c0, c1 = compute_blended_lqr_c0_c1(self.r_2d, self.P_2d)
        self.assertIsInstance(c0, float)
        self.assertIsInstance(c1, float)
        self.assertAlmostEqual(c0, 0.08, places=7)
        self.assertAlmostEqual(c1, 0.32, places=7)
        self.assertAlmostEqual(c1, 4.0 * c0, places=7)

    def test_numpy_array_p(self) -> None:
        p_np = np.array([[3.0, 0.0], [0.0, 2.0]])
        r = (0.5, 0.5)
        c0, c1 = compute_blended_lqr_c0_c1(r, p_np)
        self.assertAlmostEqual(c0, 1.25, places=7)
        self.assertAlmostEqual(c1, 5.0, places=7)

    def test_p_from_lqr_controller(self) -> None:
        dt = 0.1
        A = th.tensor([[1.0, dt], [0.0, 1.0]])
        B = th.tensor([[0.0], [dt]])
        Q = th.tensor([[10.0, 0.0], [0.0, 1.0]])
        R = th.tensor([[0.5]])
        lqr = LQRController(A, B, R, Q=Q)

        c0, c1 = compute_blended_lqr_c0_c1((0.1, 0.1), lqr)
        self.assertGreater(c0, 0.0)
        self.assertAlmostEqual(c1, 4.0 * c0, places=5)

    def test_p_from_blended_lqr_policy(self) -> None:
        dt = 0.1
        A = th.tensor([[1.0, dt], [0.0, 1.0]])
        B = th.tensor([[0.0], [dt]])
        Q = th.tensor([[10.0, 0.0], [0.0, 1.0]])
        R = th.tensor([[0.5]])
        lqr = LQRController(A, B, R, Q=Q)
        nn_policy = BoundedPolicy(
            feature_net=MLP(layer_dims=[2, 8, 1], activations=["relu", "identity"])
        )
        blended = BlendedLQRPolicy(nn_policy, lqr, c0=0.1, c1=0.5)

        c0, c1 = compute_blended_lqr_c0_c1((0.2, 0.1), blended)
        self.assertGreater(c0, 0.0)
        self.assertAlmostEqual(c1, 4.0 * c0, places=5)

    def test_points_in_box_satisfy_bounds(self) -> None:
        c0, c1 = compute_blended_lqr_c0_c1(self.r_2d, self.P_2d)
        torch_gen = th.Generator().manual_seed(42)
        u = th.rand(200, 2, generator=torch_gen, dtype=th.float64) * 2.0 - 1.0
        x_in_box = u * th.tensor(self.r_2d, dtype=th.float64)
        v_box = (x_in_box @ self.P_2d * x_in_box).sum(dim=-1)
        self.assertTrue((v_box <= c0 + 1e-9).all().item())

        x_in_double = u * (2.0 * th.tensor(self.r_2d, dtype=th.float64))
        v_double = (x_in_double @ self.P_2d * x_in_double).sum(dim=-1)
        self.assertTrue((v_double <= c1 + 1e-9).all().item())

    def test_dimension_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_blended_lqr_c0_c1((0.1, 0.2, 0.3), self.P_2d)

    def test_non_square_p_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_blended_lqr_c0_c1((0.1, 0.2), th.randn(2, 3))

    def test_factor_custom(self) -> None:
        c0, c1_default = compute_blended_lqr_c0_c1(self.r_2d, self.P_2d)
        self.assertAlmostEqual(c1_default, 4.0 * c0, places=7)

        # Positional factor
        _, c1_pos = compute_blended_lqr_c0_c1(self.r_2d, self.P_2d, 2.5)
        self.assertAlmostEqual(c1_pos, 2.5 * c0, places=7)

        # Keyword factor
        _, c1_kw = compute_blended_lqr_c0_c1(self.r_2d, self.P_2d, factor=3.0)
        self.assertAlmostEqual(c1_kw, 3.0 * c0, places=7)

    def test_invalid_factor_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_blended_lqr_c0_c1(self.r_2d, self.P_2d, factor=1.0)
        with self.assertRaises(ValueError):
            compute_blended_lqr_c0_c1(self.r_2d, self.P_2d, factor=0.5)


if __name__ == "__main__":
    unittest.main()
