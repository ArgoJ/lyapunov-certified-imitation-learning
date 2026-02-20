import unittest

import torch as th

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


if __name__ == "__main__":
	unittest.main(verbosity=2)
