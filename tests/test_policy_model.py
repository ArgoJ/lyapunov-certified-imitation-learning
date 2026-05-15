import unittest
import tempfile

import numpy as np

import torch as th
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from mpc_datagen import MPCConfig
from lcil.imitation_learning import ImitationTrainingConfig, MLPPolicy, PolicyTrainer, TransformerPolicy
from examples.double_integrator import load_policy_model


class TestMLPPolicyBounds(unittest.TestCase):
	def test_vector_bounds_match_output_dim(self) -> None:
		model = MLPPolicy(
			layer_sizes=[2, 16, 3],
			activations=["relu", "identity"],
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
			MLPPolicy(
				layer_sizes=[2, 16, 3],
				activations=["relu", "identity"],
				u_min=[-1.0, -2.0],
				u_max=[1.0, 2.0],
			)

	def test_forward_raw_returns_unclamped_output(self) -> None:
		model = MLPPolicy(
			layer_sizes=[1, 1],
			activations=["identity"],
			u_min=-1.0,
			u_max=1.0,
		)

		with th.no_grad():
			linear = model.mlp.net[0]
			assert isinstance(linear, nn.Linear)
			linear.weight.fill_(2.0)
			linear.bias.fill_(0.0)

		x = th.tensor([[2.0]])

		bounded = model(x)
		raw = model.forward_raw(x)

		self.assertTrue(th.allclose(bounded, th.tensor([[1.0]])))
		self.assertTrue(th.allclose(raw, th.tensor([[4.0]])))


class _WrappedRawPolicy(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.weight = nn.Parameter(th.tensor([[2.0]]))

	def forward_raw(self, x: th.Tensor) -> th.Tensor:
		return x @ self.weight

	def forward(self, x: th.Tensor) -> th.Tensor:
		return th.clamp(self.forward_raw(x), min=-1.0, max=1.0)


class TestPolicyTrainerRawPredictions(unittest.TestCase):
	def test_trainer_prefers_forward_raw_for_wrapped_models(self) -> None:
		model = _WrappedRawPolicy()
		dataloader = DataLoader(
			TensorDataset(th.tensor([[1.0]]), th.tensor([[2.0]])),
			batch_size=1,
		)
		trainer = PolicyTrainer(
			model=model,
			dataloader=dataloader,
			training_config=ImitationTrainingConfig(epochs=1),
		)

		trainer.model.eval()
		pred = trainer._predict_actions(th.tensor([[1.0]]))

		self.assertTrue(th.allclose(pred, th.tensor([[2.0]])))

	def test_trainer_applies_weight_decay_from_config(self) -> None:
		model = _WrappedRawPolicy()
		dataloader = DataLoader(
			TensorDataset(th.tensor([[1.0]]), th.tensor([[2.0]])),
			batch_size=1,
		)
		trainer = PolicyTrainer(
			model=model,
			dataloader=dataloader,
			training_config=ImitationTrainingConfig(epochs=1, weight_decay=1e-2),
		)

		self.assertAlmostEqual(trainer.optimizer.param_groups[0]["weight_decay"], 1e-2)


class TestMLPPolicyConfigSerialization(unittest.TestCase):
	def test_save_stores_mpc_config_and_dataset_paths(self) -> None:
		model = MLPPolicy(
			layer_sizes=[2, 16, 1],
			activations=["relu", "identity"],
		)

		cfg = MPCConfig(T_sim=40, N=20, nx=2, nu=1, dt=0.2)
		cfg.constraints.lbx = np.array([-5.0, -3.0], dtype=float)
		cfg.constraints.ubx = np.array([5.0, 3.0], dtype=float)
		cfg.constraints.lbu = np.array([-1.0], dtype=float)
		cfg.constraints.ubu = np.array([1.0], dtype=float)

		with tempfile.TemporaryDirectory() as tmp_dir:
			tmp_path = Path(tmp_dir)
			checkpoint_path = tmp_path / "model.pt"

			model.save(
				checkpoint_path,
				global_config=cfg,
			)

			payload = th.load(checkpoint_path, map_location="cpu", weights_only=True)
		self.assertEqual(payload["train_data_config"], cfg.to_dict())

	def test_load_reconstructs_mpc_config_from_dict(self) -> None:
		model = MLPPolicy(
			layer_sizes=[2, 16, 1],
			activations=["relu", "identity"],
		)

		cfg = MPCConfig(T_sim=30, N=10, nx=2, nu=1, dt=0.15)
		cfg.constraints.lbx = np.array([-2.0, -1.0], dtype=float)
		cfg.constraints.ubx = np.array([2.0, 1.0], dtype=float)

		with tempfile.TemporaryDirectory() as tmp_dir:
			checkpoint_path = Path(tmp_dir) / "model.pt"
			model.save(checkpoint_path, global_config=cfg)
			loaded = MLPPolicy.load(checkpoint_path)

		self.assertIsInstance(loaded.global_config, MPCConfig)
		assert loaded.global_config is not None
		self.assertAlmostEqual(loaded.global_config.dt, cfg.dt)
		np.testing.assert_allclose(loaded.global_config.constraints.lbx, cfg.constraints.lbx)
		np.testing.assert_allclose(loaded.global_config.constraints.ubx, cfg.constraints.ubx)

	def test_load_preserves_regularization_metadata(self) -> None:
		model = MLPPolicy(
			layer_sizes=[2, 16, 1],
			activations=["relu", "identity"],
			dropout=0.25,
			normalization="layer_norm",
		)

		with tempfile.TemporaryDirectory() as tmp_dir:
			checkpoint_path = Path(tmp_dir) / "model.pt"
			model.save(checkpoint_path)
			loaded = MLPPolicy.load(checkpoint_path)

		self.assertAlmostEqual(loaded.dropout, 0.25)
		self.assertEqual(loaded.normalization, "layer_norm")
		self.assertTrue(any(isinstance(module, nn.LayerNorm) for module in loaded.mlp.net))
		self.assertTrue(any(isinstance(module, nn.Dropout) for module in loaded.mlp.net))


class TestDoubleIntegratorPolicyLoader(unittest.TestCase):
	def test_load_policy_model_auto_detects_mlp_checkpoint(self) -> None:
		model = MLPPolicy(
			layer_sizes=[2, 8, 1],
			activations=["relu", "identity"],
		)
		cfg = MPCConfig(T_sim=10, N=5, nx=2, nu=1, dt=0.1)

		with tempfile.TemporaryDirectory() as tmp_dir:
			checkpoint_path = Path(tmp_dir) / "model.pt"
			model.save(checkpoint_path, global_config=cfg)
			loaded = load_policy_model(checkpoint_path, th.device("cpu"))

		self.assertIsInstance(loaded, MLPPolicy)

	def test_load_policy_model_auto_detects_transformer_checkpoint(self) -> None:
		model = TransformerPolicy(
			input_dim=2,
			output_dim=1,
			d_model=8,
			nhead=2,
			num_encoder_layers=1,
			dim_feedforward=16,
			max_seq_len=4,
		)
		cfg = MPCConfig(T_sim=10, N=5, nx=2, nu=1, dt=0.1)

		with tempfile.TemporaryDirectory() as tmp_dir:
			checkpoint_path = Path(tmp_dir) / "model.pt"
			model.save(checkpoint_path, global_config=cfg)
			loaded = load_policy_model(checkpoint_path, th.device("cpu"))

		self.assertIsInstance(loaded, TransformerPolicy)


if __name__ == "__main__":
	unittest.main(verbosity=2)
