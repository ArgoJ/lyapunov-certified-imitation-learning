import unittest
import tempfile

import numpy as np

import torch as th

from pathlib import Path

from mpc_datagen import MPCConfig
from lcil.imitation_learning_mlp.models import MLPPolicy


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


if __name__ == "__main__":
	unittest.main(verbosity=2)
