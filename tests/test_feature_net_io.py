import tempfile
import unittest

import torch as th

from pathlib import Path

from lcil.utils.base_models import MLP, load_feature_net, save_feature_net


class WrappedFeatureNet(th.nn.Module):
    def __init__(self, feature_net: th.nn.Module) -> None:
        super().__init__()
        self.net = feature_net

    def forward(self, x: th.Tensor) -> th.Tensor:
        cart_pos_vel = x[:, :2]
        theta = x[:, 2:3]
        theta_dot = x[:, 3:]
        features = th.cat(
            [
                cart_pos_vel,
                th.sin(theta),
                th.cos(theta),
                theta_dot,
            ],
            dim=-1,
        )
        return self.net(features)


class SaveableFeatureNet(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = th.nn.Linear(4, 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        th.save({"state_dict": self.state_dict()}, checkpoint_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: th.device | str = "cpu",
    ) -> "SaveableFeatureNet":
        payload = th.load(path, map_location=map_location, weights_only=True)
        model = cls()
        model.load_state_dict(payload["state_dict"])
        return model


class TestFeatureNetIO(unittest.TestCase):
    def test_roundtrip_wrapped_mlp_feature_net(self) -> None:
        feature_net = WrappedFeatureNet(MLP([5, 8, 1], ["tanh", "identity"]))
        x = th.tensor(
            [
                [0.2, -0.1, 0.3, 0.4],
                [-0.4, 0.5, -0.2, 0.1],
            ],
            dtype=th.float32,
        )
        expected = feature_net(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "wrapped_feature_net.pt"
            save_feature_net(feature_net, checkpoint_path)
            loaded = load_feature_net(checkpoint_path)

        self.assertIsInstance(loaded, WrappedFeatureNet)
        self.assertIsInstance(loaded.net, MLP)
        self.assertTrue(th.allclose(loaded(x), expected))

    def test_roundtrip_wrapped_regularized_mlp_feature_net(self) -> None:
        feature_net = WrappedFeatureNet(
            MLP([5, 8, 1], ["tanh", "identity"], dropout=0.2, normalization="layer_norm")
        )
        feature_net.eval()
        x = th.tensor(
            [
                [0.2, -0.1, 0.3, 0.4],
                [-0.4, 0.5, -0.2, 0.1],
            ],
            dtype=th.float32,
        )
        expected = feature_net(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "wrapped_regularized_feature_net.pt"
            save_feature_net(feature_net, checkpoint_path)
            loaded = load_feature_net(checkpoint_path)

        self.assertIsInstance(loaded, WrappedFeatureNet)
        self.assertIsInstance(loaded.net, MLP)
        self.assertAlmostEqual(loaded.net.dropout, 0.2)
        self.assertEqual(loaded.net.normalization, "layer_norm")
        self.assertTrue(th.allclose(loaded(x), expected))

    def test_roundtrip_saveable_feature_net(self) -> None:
        feature_net = SaveableFeatureNet()
        with th.no_grad():
            feature_net.net.weight.copy_(th.tensor([[0.4, -0.2, 0.1, 0.3]], dtype=th.float32))
            feature_net.net.bias.copy_(th.tensor([0.05], dtype=th.float32))

        x = th.tensor(
            [
                [0.1, 0.2, -0.1, 0.3],
                [-0.2, 0.4, 0.5, -0.6],
            ],
            dtype=th.float32,
        )
        expected = feature_net(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "saveable_feature_net.pt"
            save_feature_net(feature_net, checkpoint_path)
            loaded = load_feature_net(checkpoint_path)

        self.assertIsInstance(loaded, SaveableFeatureNet)
        self.assertTrue(th.allclose(loaded.net.weight, feature_net.net.weight))
        self.assertTrue(th.allclose(loaded.net.bias, feature_net.net.bias))
        self.assertTrue(th.allclose(loaded(x), expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)