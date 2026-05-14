import tempfile
import unittest

from pathlib import Path

import numpy as np
import torch as th

from mpc_datagen import MPCConfig

from lcil.imitation_learning_mlp.transformer import TransformerPolicy


class TestTransformerPolicyBounds(unittest.TestCase):
    def test_vector_bounds_match_output_dim(self) -> None:
        model = TransformerPolicy(
            input_dim=2,
            output_dim=3,
            d_model=12,
            nhead=4,
            num_encoder_layers=1,
            dim_feedforward=24,
            max_seq_len=1,
            u_min=[-1.0, -2.0, -3.0],
            u_max=[1.0, 2.0, 3.0],
        )

        x = th.randn(8, 2)
        u = model(x)

        self.assertEqual(u.shape, (8, 3))
        self.assertTrue(th.all(u >= th.tensor([-1.0, -2.0, -3.0], dtype=u.dtype)))
        self.assertTrue(th.all(u <= th.tensor([1.0, 2.0, 3.0], dtype=u.dtype)))

    def test_forward_raw_returns_unclamped_output(self) -> None:
        model = TransformerPolicy(
            input_dim=1,
            output_dim=1,
            d_model=4,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=8,
            max_seq_len=1,
            dropout=0.0,
            u_min=-1.0,
            u_max=1.0,
        )

        with th.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output_projection.bias.fill_(2.0)

        x = th.tensor([[0.5]])

        bounded = model(x)
        raw = model.forward_raw(x)

        self.assertTrue(th.allclose(raw, th.tensor([[2.0]])))
        self.assertTrue(th.allclose(bounded, th.tensor([[1.0]])))

    def test_forward_without_bounds_returns_raw_output(self) -> None:
        model = TransformerPolicy(
            input_dim=1,
            output_dim=1,
            d_model=4,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=8,
            max_seq_len=1,
            dropout=0.0,
        )

        with th.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output_projection.bias.fill_(1.5)

        x = th.tensor([[0.25]])

        self.assertTrue(th.allclose(model(x), model.forward_raw(x)))


class TestTransformerPolicySequenceIO(unittest.TestCase):
    def test_sequence_input_returns_last_token_output_by_default(self) -> None:
        model = TransformerPolicy(
            input_dim=2,
            output_dim=1,
            d_model=8,
            nhead=2,
            num_encoder_layers=2,
            dim_feedforward=16,
            max_seq_len=5,
        )

        x = th.randn(4, 3, 2)
        u = model.forward_raw(x)

        self.assertEqual(u.shape, (4, 1))

    def test_sequence_input_returns_sequence_output_in_per_step_mode(self) -> None:
        model = TransformerPolicy(
            input_dim=2,
            output_dim=1,
            d_model=8,
            nhead=2,
            num_encoder_layers=2,
            dim_feedforward=16,
            max_seq_len=5,
            output_mode="per_step",
        )

        x = th.randn(4, 3, 2)
        u = model.forward_raw(x)

        self.assertEqual(u.shape, (4, 3, 1))

    def test_sequence_longer_than_max_seq_len_raises(self) -> None:
        model = TransformerPolicy(
            input_dim=2,
            output_dim=1,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=16,
            max_seq_len=2,
        )

        with self.assertRaises(ValueError):
            model.forward_raw(th.randn(1, 3, 2))


class TestTransformerPolicyConfigSerialization(unittest.TestCase):
    def test_save_and_load_round_trip_config_and_outputs(self) -> None:
        model = TransformerPolicy(
            input_dim=2,
            output_dim=1,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=16,
            max_seq_len=4,
            dropout=0.0,
            activation="relu",
            causal=False,
            output_mode="per_step",
            u_min=-1.0,
            u_max=1.0,
        )
        model.eval()

        cfg = MPCConfig(T_sim=30, N=10, nx=2, nu=1, dt=0.15)
        cfg.constraints.lbx = np.array([-2.0, -1.0], dtype=float)
        cfg.constraints.ubx = np.array([2.0, 1.0], dtype=float)

        sample = th.randn(3, 2)
        expected = model(sample)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "transformer_model.pt"
            model.save(checkpoint_path, global_config=cfg)
            loaded = TransformerPolicy.load(checkpoint_path)

        loaded.eval()
        actual = loaded(sample)

        self.assertIsInstance(loaded.global_config, MPCConfig)
        assert loaded.global_config is not None
        self.assertAlmostEqual(loaded.global_config.dt, cfg.dt)
        self.assertEqual(loaded.max_seq_len, 4)
        self.assertFalse(loaded.causal)
        self.assertEqual(loaded.output_mode, "per_step")
        th.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)