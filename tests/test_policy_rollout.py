import unittest

import numpy as np

from lcil.imitation_learning_mlp.policy_rollout import (
    PolicyRolloutConfig,
    RandomBoundsSampler,
)


class TestPolicyRolloutConfig(unittest.TestCase):
    def test_config_bounds_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            PolicyRolloutConfig(
                T_sim=10,
                dt=0.1,
                nx=2,
                nu=1,
                state_bounds=np.array([[-1.0, -1.0]]),
            )

    def test_random_bounds_sampler_samples_in_box(self) -> None:
        bounds = np.array([[-1.0, -2.0], [1.0, 2.0]], dtype=float)
        sampler = RandomBoundsSampler(bounds=bounds, seed=7)

        x0 = sampler.sample_x0()

        self.assertEqual(x0.shape, (2,))
        self.assertTrue(np.all(x0 >= bounds[0]))
        self.assertTrue(np.all(x0 <= bounds[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
