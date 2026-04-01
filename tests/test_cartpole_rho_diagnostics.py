import math
import unittest

import numpy as np
import torch as th

from shared_cartpole import (
    PendulumOnCartConfig,
    RiccatiPolicy,
    RiccatiQuadraticLyapunov,
    NonlinearInvertedPendulumOnCartDynamics,
    riccati_gain_and_value_matrix,
)


class TestCartpoleRhoDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sys_cfg = PendulumOnCartConfig()
        k_gain, p_matrix = riccati_gain_and_value_matrix(cls.sys_cfg)

        cls.policy = RiccatiPolicy(k_gain)
        cls.lyap = RiccatiQuadraticLyapunov(p_matrix)
        cls.dyn = NonlinearInvertedPendulumOnCartDynamics(cls.sys_cfg)

        cls.p_matrix = p_matrix
        cls.state_dim = int(p_matrix.shape[0])

        theta_bound = np.pi * 1.4999
        cls.lb = np.array([-2.0, -10.0, -theta_bound, -10.0], dtype=np.float32)
        cls.ub = np.array([2.0, 10.0, theta_bound, 10.0], dtype=np.float32)

        cls.rho = 31.5
        cls.kappa = 1e-3
        cls.invariance_weight = 1.0

    @staticmethod
    def _unit_ball_volume(state_dim: int) -> float:
        return float(np.pi ** (0.5 * state_dim) / math.gamma(0.5 * state_dim + 1.0))

    @classmethod
    def _sample_inside_sublevel(cls, rho: float, count: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)

        z = rng.normal(size=(count, cls.state_dim)).astype(np.float64)
        z_norm = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / np.maximum(z_norm, 1e-12)

        radii = rng.random(size=(count, 1)) ** (1.0 / cls.state_dim)
        y = z * radii

        chol = np.linalg.cholesky(cls.p_matrix)
        x = np.sqrt(rho) * np.linalg.solve(chol.T, y.T).T
        return x.astype(np.float32)

    def test_sublevel_volume_ratio_is_tiny(self) -> None:
        det_p = float(np.linalg.det(self.p_matrix))
        ellipsoid_volume = (
            self._unit_ball_volume(self.state_dim)
            * (self.rho ** (0.5 * self.state_dim))
            / math.sqrt(det_p)
        )
        box_volume = float(np.prod(self.ub - self.lb))
        ratio = ellipsoid_volume / box_volume

        self.assertGreater(ratio, 0.0)
        self.assertLess(
            ratio,
            1e-4,
            msg=(
                "Sublevel set is expected to be tiny compared to cert bounds; "
                f"got volume ratio={ratio:.3e}."
            ),
        )

    def test_uniform_box_sampling_rarely_hits_sublevel(self) -> None:
        rng = np.random.default_rng(0)
        x_box = rng.uniform(self.lb, self.ub, size=(120_000, self.state_dim)).astype(np.float32)

        with th.no_grad():
            v = self.lyap(th.tensor(x_box, dtype=th.float32))[:, 0].cpu().numpy()

        inside_fraction = float(np.mean(v <= self.rho))
        self.assertLess(
            inside_fraction,
            1e-3,
            msg=(
                "Uniform sampling over cert_bounds should almost never hit V<=rho for this setup; "
                f"got inside_fraction={inside_fraction:.6f}."
            ),
        )

    def test_true_residual_inside_sublevel_has_margin(self) -> None:
        x_inside = self._sample_inside_sublevel(rho=self.rho, count=120_000, seed=1)
        x = th.tensor(x_inside, dtype=th.float32)
        lb_t = th.tensor(self.lb, dtype=th.float32).reshape(1, -1)
        ub_t = th.tensor(self.ub, dtype=th.float32).reshape(1, -1)

        with th.no_grad():
            v_curr = self.lyap(x)
            u = self.policy(x)
            x_next = self.dyn(x, u)
            v_next = self.lyap(x_next)

            f_term = (v_next - (1.0 - self.kappa) * v_curr)[:, 0]
            h_term = (th.relu(x_next - ub_t) + th.relu(lb_t - x_next)).sum(dim=1)
            g_term = th.relu(f_term) + self.invariance_weight * h_term

        self.assertLessEqual(
            float(v_curr.max()),
            self.rho + 1e-4,
            msg="Sublevel sampler produced points outside V<=rho.",
        )
        self.assertLess(
            float(f_term.max()),
            -1e-4,
            msg=(
                "Expected strict local decrease margin inside sampled V<=rho set; "
                f"got f_max={float(f_term.max()):.6f}."
            ),
        )
        self.assertLessEqual(
            float(h_term.max()),
            1e-7,
            msg=(
                "Expected invariance penalty to stay zero in sampled V<=rho set; "
                f"got h_max={float(h_term.max()):.6e}."
            ),
        )
        self.assertLessEqual(
            float(g_term.max()),
            1e-7,
            msg=(
                "Expected relaxed residual to stay zero in sampled V<=rho set; "
                f"got g_max={float(g_term.max()):.6e}."
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
