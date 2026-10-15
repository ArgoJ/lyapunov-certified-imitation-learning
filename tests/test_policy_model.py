import tempfile
import unittest
from pathlib import Path

import torch as th
import torch.nn as nn

from lcil.imitation_learning import BlendedLQRPolicy, BoundedPolicy, LQRController
from lcil.utils import MLP


def _build_mlp_feature_net(
    layer_dims: list[int],
    activations: list[str],
    *,
    dropout: float = 0.0,
    normalization: str = "none",
) -> MLP:
    return MLP(
        layer_dims=layer_dims,
        activations=activations,
        dropout=dropout,
        normalization=normalization,
    )


class TestBoundedPolicyBounds(unittest.TestCase):
    def test_vector_bounds_match_output_dim(self) -> None:
        model = BoundedPolicy(
            feature_net=_build_mlp_feature_net([2, 16, 3], ["relu", "identity"]),
            u_min=[-1.0, -2.0, -3.0],
            u_max=[1.0, 2.0, 3.0],
        )

        x = th.randn(8, 2)
        u = model(x)

        self.assertEqual(u.shape[-1], 3)
        self.assertTrue(th.all(u >= th.tensor([-1.0, -2.0, -3.0], dtype=u.dtype)))
        self.assertTrue(th.all(u <= th.tensor([1.0, 2.0, 3.0], dtype=u.dtype)))

    def test_mismatched_vector_bounds_raise(self) -> None:
        with self.assertRaises(ValueError):
            BoundedPolicy(
                feature_net=_build_mlp_feature_net([2, 16, 3], ["relu", "identity"]),
                u_min=[-1.0, -2.0],
                u_max=[1.0, 2.0],
            )

    def test_forward_raw_returns_unclamped_output(self) -> None:
        model = BoundedPolicy(
            feature_net=_build_mlp_feature_net([1, 1], ["identity"]),
            u_min=-1.0,
            u_max=1.0,
        )

        with th.no_grad():
            linear = model.feature_net.net[0]
            assert isinstance(linear, nn.Linear)
            linear.weight.fill_(2.0)
            linear.bias.fill_(0.0)

        x = th.tensor([[2.0]])

        bounded = model(x)
        raw = model.forward_raw(x)

        self.assertTrue(th.allclose(bounded, th.tensor([[1.0]])))
        self.assertTrue(th.allclose(raw, th.tensor([[4.0]])))


class TestBoundedPolicyReferences(unittest.TestCase):
    def test_forward_raw_applies_reference_shift(self) -> None:
        model = BoundedPolicy(
            feature_net=_build_mlp_feature_net([1, 1], ["identity"]),
            u_ref=[0.5],
            x_ref=[1.0],
        )

        with th.no_grad():
            linear = model.feature_net.net[0]
            assert isinstance(linear, nn.Linear)
            linear.weight.fill_(2.0)
            linear.bias.fill_(0.0)

        raw = model.forward_raw(th.tensor([[3.0]]))
        self.assertTrue(th.allclose(raw, th.tensor([[4.5]])))


class TestPolicySerialization(unittest.TestCase):
    def test_save_writes_model_only_checkpoint(self) -> None:
        model = BoundedPolicy(
            feature_net=_build_mlp_feature_net([2, 16, 1], ["relu", "identity"]),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "model.pt"
            model.save(checkpoint_path)

            payload = th.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["policy_type"], "BoundedPolicy")
        self.assertNotIn("train_data_config", payload)

    def test_bounded_policy_round_trip_preserves_outputs(self) -> None:
        model = BoundedPolicy(
            feature_net=_build_mlp_feature_net([2, 16, 1], ["relu", "identity"]),
            u_min=-1.0,
            u_max=1.0,
        )
        sample = th.randn(4, 2)
        expected = model(sample)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "model.pt"
            model.save(checkpoint_path)
            loaded = BoundedPolicy.load(checkpoint_path)

        th.testing.assert_close(loaded(sample), expected)

    def test_bounded_policy_round_trip_preserves_reference_shift(self) -> None:
        model = BoundedPolicy(
            feature_net=_build_mlp_feature_net([2, 16, 1], ["relu", "identity"]),
            u_min=-1.0,
            u_max=1.0,
        )
        sample = th.randn(5, 2)
        expected = model.forward_raw(sample)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "model.pt"
            model.save(checkpoint_path)
            loaded = BoundedPolicy.load(checkpoint_path)

        th.testing.assert_close(loaded.forward_raw(sample), expected)


class TestBoundedPolicyFixedCache(unittest.TestCase):
    def test_prepare_fixed_caches_and_preserves_forward(self) -> None:
        model = BoundedPolicy(
            feature_net=_build_mlp_feature_net([2, 16, 1], ["relu", "identity"]),
            u_min=-1.0,
            u_max=1.0,
            u_ref=[0.5],
            x_ref=[1.0, -0.5],
        )
        sample = th.randn(5, 2)
        expected = model.forward_raw(sample)

        self.assertFalse(model._is_fixed)
        self.assertIsNone(model._cached_u_offset)

        model.prepare_fixed()
        self.assertTrue(model._is_fixed)
        self.assertIsNotNone(model._cached_u_offset)

        fixed_out = model.forward_raw(sample)
        th.testing.assert_close(fixed_out, expected)

        # Train mode resets fixed state
        model.train(True)
        self.assertFalse(model._is_fixed)
        self.assertIsNone(model._cached_u_offset)


class TestLQRController(unittest.TestCase):
    def setUp(self) -> None:
        dt = 0.1
        self.A = th.tensor([[1.0, dt], [0.0, 1.0]])
        self.B = th.tensor([[0.0], [dt]])
        self.Q = th.tensor([[10.0, 0.0], [0.0, 1.0]])
        self.R = th.tensor([[0.5]])

    def test_lqr_dare_solve_and_gain(self) -> None:
        lqr = LQRController(self.A, self.B, self.R, Q=self.Q)
        self.assertEqual(lqr.K.dtype, th.float32)
        self.assertEqual(lqr.P.dtype, th.float32)
        self.assertEqual(lqr.K.shape, (1, 2))
        self.assertEqual(lqr.P.shape, (2, 2))

        # Explicit P matches computed P
        lqr_explicit = LQRController(self.A, self.B, self.R, P=lqr.P)
        self.assertTrue(th.allclose(lqr.K, lqr_explicit.K))

        # Cholesky factor satisfies L @ L.T == P
        self.assertEqual(lqr.L.shape, (2, 2))
        self.assertTrue(th.allclose(lqr.L @ lqr.L.T, lqr.P, atol=1e-5))

    def test_lqr_requires_p_or_q(self) -> None:
        with self.assertRaises(ValueError):
            LQRController(self.A, self.B, self.R)

    def test_lqr_forward_shapes(self) -> None:
        lqr = LQRController(self.A, self.B, self.R, Q=self.Q)
        x_1d = th.randn(2)
        x_2d = th.randn(10, 2)
        self.assertEqual(lqr(x_1d).shape, (1,))
        self.assertEqual(lqr(x_2d).shape, (10, 1))

    def test_lqr_control_bounds(self) -> None:
        lqr = LQRController(self.A, self.B, self.R, Q=self.Q, u_min=-1.0, u_max=1.0)
        x_extreme = th.tensor([[100.0, 100.0], [-100.0, -100.0]])
        u = lqr(x_extreme)
        self.assertTrue(th.allclose(u[0], th.tensor([-1.0])))
        self.assertTrue(th.allclose(u[1], th.tensor([1.0])))


class TestBlendedLQRPolicy(unittest.TestCase):
    def setUp(self) -> None:
        dt = 0.1
        A = th.tensor([[1.0, dt], [0.0, 1.0]])
        B = th.tensor([[0.0], [dt]])
        Q = th.tensor([[10.0, 0.0], [0.0, 1.0]])
        R = th.tensor([[0.5]])
        self.lqr = LQRController(A, B, R, Q=Q)
        self.nn_policy = BoundedPolicy(
            feature_net=_build_mlp_feature_net([2, 16, 1], ["relu", "identity"]),
            u_min=-3.0,
            u_max=3.0,
        )

    def test_blended_validation_c0_c1(self) -> None:
        # Must satisfy 0 < c0 < c1
        with self.assertRaises(ValueError):
            BlendedLQRPolicy(self.nn_policy, self.lqr, c0=0.5, c1=0.2)
        with self.assertRaises(ValueError):
            BlendedLQRPolicy(self.nn_policy, self.lqr, c0=-0.1, c1=0.5)

    def test_blended_alpha_and_output(self) -> None:
        c0, c1 = 0.1, 0.8
        blended = BlendedLQRPolicy(self.nn_policy, self.lqr, c0=c0, c1=c1)

        # Origin: V = 0 <= c0 -> pure LQR
        x_origin = th.zeros(1, 2)
        self.assertEqual(blended.compute_alpha(x_origin).item(), 0.0)
        self.assertTrue(th.allclose(blended(x_origin), self.lqr(x_origin)))

        # Far outside: V >= c1 -> pure NN policy
        x_far = th.tensor([[5.0, 5.0]])
        self.assertEqual(blended.compute_alpha(x_far).item(), 1.0)
        self.assertTrue(th.allclose(blended(x_far), self.nn_policy(x_far)))

        # Batch forward shape
        x_batch = th.randn(20, 2)
        u_batch = blended(x_batch)
        self.assertEqual(u_batch.shape, (20, 1))
        self.assertTrue(th.all(u_batch >= -3.0) and th.all(u_batch <= 3.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
