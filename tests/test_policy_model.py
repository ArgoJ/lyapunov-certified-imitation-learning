import tempfile
import unittest
from pathlib import Path

import torch as th
import torch.nn as nn

from lcil.imitation_learning import BoundedPolicy
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
