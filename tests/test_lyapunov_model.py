import tempfile
import unittest

import torch as th

from pathlib import Path

from lcil.lyapunov_learning.models import LyapunovCandidate, NeuralLyapunovCandidate
from lcil.lyapunov_learning.utils import calculate_r_factor_from_riccati
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
    def test_r_factor_initializes_pd_matrix(self) -> None:
        feature_net = SaveableFeatureNet()
        p_matrix = th.tensor(
            [
                [2.0, 0.3, 0.0, 0.0],
                [0.3, 1.8, 0.2, 0.0],
                [0.0, 0.2, 1.6, 0.1],
                [0.0, 0.0, 0.1, 1.4],
            ],
            dtype=th.float32,
        )
        eps = 1e-3
        r_factor, _ = calculate_r_factor_from_riccati(p_matrix, eps=eps)
        model = NeuralLyapunovCandidate(
            feature_net=feature_net,
            state_dim=4,
            eps=eps,
            r_factor=r_factor,
        )

        self.assertTrue(th.allclose(model._pd_matrix(), p_matrix, atol=1e-5, rtol=1e-5))

    def test_calculate_r_factor_adjusts_eps_when_lambda_min_less_than_eps(self) -> None:
        p_matrix = th.tensor(
            [
                [2.0, 0.5],
                [0.5, 1.0],
            ],
            dtype=th.float32,
        )
        min_eig = float(th.linalg.eigvalsh(p_matrix)[0].item())
        initial_eps = min_eig + 1.0
        r_factor, adjusted_eps = calculate_r_factor_from_riccati(p_matrix, eps=initial_eps)

        self.assertAlmostEqual(adjusted_eps, 0.5 * min_eig, places=5)
        reconstructed = adjusted_eps * th.eye(2) + r_factor.T @ r_factor
        self.assertTrue(th.allclose(reconstructed, p_matrix, atol=1e-5, rtol=1e-5))

    def test_fixed_r_factor_forces_eps_zero(self) -> None:
        feature_net = SaveableFeatureNet()
        p_matrix = th.tensor(
            [
                [2.0, 0.5],
                [0.5, 1.0],
            ],
            dtype=th.float32,
        )
        r_factor, _ = calculate_r_factor_from_riccati(p_matrix, eps=0.0)
        model = NeuralLyapunovCandidate(
            feature_net=feature_net,
            state_dim=2,
            eps=0.5,
            r_factor=r_factor,
            fixed_r_factor=True,
        )
        self.assertEqual(model.eps, 0.0)
        self.assertTrue(th.allclose(model._pd_matrix(), p_matrix, atol=1e-5, rtol=1e-5))

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


class TestLyapunovCandidateProtocol(unittest.TestCase):
    def test_neural_lyapunov_candidate_is_instance_and_subclass(self) -> None:
        self.assertTrue(issubclass(NeuralLyapunovCandidate, LyapunovCandidate))
        model = NeuralLyapunovCandidate(
            feature_net=SaveableFeatureNet(),
            state_dim=4,
        )
        self.assertIsInstance(model, LyapunovCandidate)

    def test_protocol_methods(self) -> None:
        model = NeuralLyapunovCandidate(
            feature_net=SaveableFeatureNet(),
            state_dim=4,
        )
        x = th.randn(3, 4)
        feature_term = model.get_feature_term(x)
        linear_term = model.get_linear_term(x)
        forward_val = model.forward(x)
        call_val = model(x)

        self.assertEqual(feature_term.shape, (3, 1))
        self.assertEqual(linear_term.shape, (3, 1))
        self.assertEqual(forward_val.shape, (3, 1))
        self.assertTrue(th.allclose(forward_val, feature_term + linear_term))
        self.assertTrue(th.allclose(call_val, forward_val))

    def test_initialized_last_feature_layer(self) -> None:
        feature_net = SaveableFeatureNet()
        r_matrix = th.eye(4, dtype=th.float32)
        model = NeuralLyapunovCandidate(
            feature_net=feature_net,
            state_dim=4,
            r_factor=r_matrix,
        )
        # Verify the last linear layer biases are zero and weights are initialized
        if feature_net.net.bias is not None:
            self.assertTrue(th.all(feature_net.net.bias == 0.0))
        self.assertFalse(th.all(feature_net.net.weight == 0.0))

        # Forward pass feature term at origin x* must be 0
        x_star = th.zeros(1, 4)
        self.assertTrue(th.allclose(model.get_feature_term(x_star), th.zeros(1, 1)))

    def test_init_with_polyhedral_matrix(self) -> None:
        from lcil.lyapunov_learning import (
            calculate_r_factor_from_riccati,
            compute_induced_1norm_gain,
            compute_polyhedral_value_matrix,
        )

        feature_net = SaveableFeatureNet()
        A = th.tensor([[0.8, 0.5, 0.0, 0.0],
                       [0.0, 0.7, 0.0, 0.0],
                       [0.0, 0.0, 0.5, 0.0],
                       [0.0, 0.0, 0.0, 0.5]], dtype=th.float32)
        P_riccati = th.eye(4, dtype=th.float32)
        p_opt = compute_polyhedral_value_matrix(
            a_closed_loop=A,
            p_initial=P_riccati,
            kappa=0.01,
        )
        r_opt, _ = calculate_r_factor_from_riccati(p_opt, eps=0.0)
        model = NeuralLyapunovCandidate(
            feature_net=feature_net,
            state_dim=4,
            r_factor=r_opt,
            fixed_r_factor=True,
        )
        gain = compute_induced_1norm_gain(model._pd_matrix(), A).max().item()
        self.assertLess(gain, 1.0 - 0.01)

    def test_calculate_r_factor_from_riccati(self) -> None:
        from lcil.lyapunov_learning import calculate_r_factor_from_riccati

        p = th.tensor([[2.0, 0.5], [0.5, 3.0]], dtype=th.float32)
        r, eps = calculate_r_factor_from_riccati(p, eps=1e-3)
        reconstructed = eps * th.eye(2) + r.T @ r
        self.assertTrue(th.allclose(p, reconstructed, atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)