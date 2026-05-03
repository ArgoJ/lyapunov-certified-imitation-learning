import tempfile
import unittest

import torch as th

from pathlib import Path

from lcil.lyapunov_learning.models import NeuralLyapunovCandidate
from lcil.utils.base_models import MLP


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
        checkpoint = th.load(path, map_location=map_location, weights_only=True)
        model = cls()
        model.load_state_dict(checkpoint["state_dict"])
        return model


class TestNeuralLyapunovCandidateSerialization(unittest.TestCase):
    def test_save_load_roundtrip_with_state_dict_feature_net(self) -> None:
        feature_net = WrappedFeatureNet(
            feature_net=MLP([5, 8, 1], ["tanh", "identity"]),
        )
        model = NeuralLyapunovCandidate(
            feature_net=feature_net,
            state_dim=4,
            eps=5e-3,
            x_star=th.tensor([0.1, -0.2, 0.3, -0.4], dtype=th.float32),
        )

        with th.no_grad():
            model.r_factor.copy_(
                th.tensor(
                    [
                        [1.0, 0.1, 0.0, 0.0],
                        [0.0, 1.1, 0.2, 0.0],
                        [0.0, 0.0, 0.9, 0.3],
                        [0.0, 0.0, 0.0, 1.2],
                    ],
                    dtype=th.float32,
                )
            )

        x = th.tensor(
            [
                [0.2, -0.1, 0.3, 0.4],
                [-0.4, 0.5, -0.2, 0.1],
            ],
            dtype=th.float32,
        )
        expected = model(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "lyapunov_model.pt"
            model.save(checkpoint_path)
            loaded = NeuralLyapunovCandidate.load(
                checkpoint_path,
                feature_net_cls=WrappedFeatureNet,
                feature_net_kwargs={
                    "feature_net": MLP([5, 8, 1], ["tanh", "identity"]),
                },
            )

        self.assertIsInstance(loaded.feature_net, WrappedFeatureNet)
        self.assertAlmostEqual(loaded.eps, model.eps)
        self.assertEqual(loaded.state_dim, model.state_dim)
        self.assertTrue(th.allclose(loaded.x_star, model.x_star))
        self.assertTrue(th.allclose(loaded.r_factor, model.r_factor))
        self.assertTrue(th.allclose(loaded(x), expected))

    def test_save_load_roundtrip_with_saveable_feature_net(self) -> None:
        model = NeuralLyapunovCandidate(
            feature_net=SaveableFeatureNet(),
            state_dim=4,
            eps=1e-2,
            x_star=th.tensor([-0.3, 0.2, 0.1, -0.5], dtype=th.float32),
        )

        with th.no_grad():
            model.feature_net.net.weight.copy_(
                th.tensor([[0.4, -0.2, 0.1, 0.3]], dtype=th.float32)
            )
            model.feature_net.net.bias.copy_(th.tensor([0.05], dtype=th.float32))
            model.r_factor.copy_(
                th.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.1, 0.9, 0.0, 0.0],
                        [0.0, 0.2, 1.1, 0.0],
                        [0.0, 0.0, 0.3, 1.2],
                    ],
                    dtype=th.float32,
                )
            )

        x = th.tensor(
            [
                [0.1, 0.2, -0.1, 0.3],
                [-0.2, 0.4, 0.5, -0.6],
            ],
            dtype=th.float32,
        )
        expected = model(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "lyapunov_model.pt"
            model.save(checkpoint_path)
            loaded = NeuralLyapunovCandidate.load(checkpoint_path)

        self.assertIsInstance(loaded.feature_net, SaveableFeatureNet)
        self.assertAlmostEqual(loaded.eps, model.eps)
        self.assertEqual(loaded.state_dim, model.state_dim)
        self.assertTrue(th.allclose(loaded.x_star, model.x_star))
        self.assertTrue(th.allclose(loaded.r_factor, model.r_factor))
        self.assertTrue(
            th.allclose(loaded.feature_net.net.weight, model.feature_net.net.weight)
        )
        self.assertTrue(th.allclose(loaded.feature_net.net.bias, model.feature_net.net.bias))
        self.assertTrue(th.allclose(loaded(x), expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)