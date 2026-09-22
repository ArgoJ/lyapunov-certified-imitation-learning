import unittest
from unittest.mock import patch

import numpy as np
import torch as th
from shared_utils import (
    _FirstCoordinateValue,
    _QuadraticLyapunov,
)

from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.buffer import BoundaryStateBuffer, CEGISBuffer
from lcil.lyapunov_learning.sublevel import (
    BoundaryRhoEstimate,
    BoundaryTermDiagnostics,
    BoundaryRhoEvaluation,
    RhoEstimationConfig,
    estimate_rho,
    estimate_rho_from_boundary,
    _boundary_term_diagnostics,
)
from lcil.lyapunov_learning.utils import ThresholdMonitor


class TestLyapunovSublevel(unittest.TestCase):
    def test_rho_threshold_monitor_triggers_after_consecutive_low_values(self) -> None:
        monitor = ThresholdMonitor(threshold=1.0, patience=3)

        outputs = [monitor.update(value) for value in (0.8, 0.9, 1.2, 0.7, 0.6, 0.5)]

        self.assertEqual(outputs, [False, False, False, False, False, True])
        self.assertEqual(monitor.value_history, [0.8, 0.9, 1.2, 0.7, 0.6, 0.5])
        self.assertEqual(monitor.consecutive_low, 3)

    def test_rho_estimation_config_from_training_config(self) -> None:
        bounds = np.array([[-2.0, -1.0], [2.0, 1.0]], dtype=np.float32)
        train_cfg = LyapunovTrainingConfig(
            state_dim=2,
            state_bounds=bounds,
            rho_estimation_samples=512,
            rho_step_size=0.02,
            rho_descent_steps=5,
            rho_estimate_quantile=0.1,
            rho_min=1e-3,
            rho_growth_gamma=1.2,
            enable_diagnosis=True,
        )
        rho_cfg = RhoEstimationConfig.from_training_config(train_cfg, samples=256)

        self.assertEqual(rho_cfg.samples, 256)
        self.assertEqual(rho_cfg.step_size, 0.02)
        self.assertEqual(rho_cfg.descent_steps, 5)
        self.assertEqual(rho_cfg.estimate_quantile, 0.1)
        self.assertEqual(rho_cfg.rho_min, 1e-3)
        self.assertEqual(rho_cfg.rho_growth_gamma, 1.2)
        self.assertTrue(rho_cfg.enable_diagnosis)
        np.testing.assert_allclose(rho_cfg.train_bounds, bounds)

    def test_estimate_rho_uses_configured_quantile(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=2,
            state_bounds=np.array([[-4.0, -1.0], [4.0, 1.0]], dtype=np.float32),
            rho_estimation_samples=4,
            rho_descent_steps=0,
            rho_growth_gamma=1.5,
            rho_estimate_quantile=0.5,
        )
        boundary_points = th.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=th.float32,
        )
        face_dims = th.zeros(4, dtype=th.long)
        is_ub = th.ones(4, dtype=th.bool)

        with patch(
            "lcil.lyapunov_learning.sublevel.sample_boundary_points",
            return_value=(boundary_points, face_dims, is_ub),
        ):
            rho_config = RhoEstimationConfig.from_training_config(config)
            rho_eval, _ = estimate_rho_from_boundary(_FirstCoordinateValue(), rho_config)
            rho = rho_eval.rho.rho

        expected = 1.5 * th.quantile(boundary_points[:, 0], q=0.5).item()
        self.assertAlmostEqual(rho, expected, places=6)

    def test_estimate_rho_reuses_boundary_buffer_low_values(self) -> None:
        first_boundary = th.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=th.float32)
        second_boundary = th.tensor([[9.0, 0.0], [10.0, 0.0]], dtype=th.float32)
        boundary_buffer = BoundaryStateBuffer(state_dim=2, max_size=2, device="cpu")

        value_fn = _FirstCoordinateValue()
        boundary_buffer.update(first_boundary, value_fn=value_fn)
        first_max_val = float(value_fn(boundary_buffer.states).max().item())

        boundary_buffer.update(second_boundary, value_fn=value_fn)
        second_max_val = float(value_fn(boundary_buffer.states).max().item())

        self.assertAlmostEqual(first_max_val, 2.0, places=6)
        self.assertAlmostEqual(second_max_val, 2.0, places=6)

    def test_estimate_rho_caps_with_violating_buffer_states(self) -> None:
        lyap_model = _QuadraticLyapunov()
        bounds = np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32)
        rho_config = RhoEstimationConfig(
            train_bounds=bounds,
            samples=8,
            descent_steps=0,
            rho_growth_gamma=1.0,
            estimate_quantile=0.05,
            rho_min=1e-4,
            cex_quantile=0.1,
        )

        buffer = CEGISBuffer(
            initial_states=th.tensor([[0.5, 0.5]], dtype=th.float32),
            state_buffer_limit=16,
            cex_buffer_limit=8,
            lb=th.tensor([-2.0, -2.0]),
            ub=th.tensor([2.0, 2.0]),
            device=th.device("cpu"),
        )
        buffer.register_cex(
            th.tensor([[0.3, 0.3]], dtype=th.float32),
            objective=lambda x: -x.sum(dim=-1),
        )

        def condition_evaluator(x: th.Tensor) -> th.Tensor:
            viol = th.zeros(x.shape[0], dtype=th.float32)
            viol[(x[:, 0] > 0.25) & (x[:, 1] > 0.25)] = 1.0
            return viol

        eval_res, _ = estimate_rho(
            lyap_model=lyap_model,
            config=rho_config,
            condition_evaluator=condition_evaluator,
            state_buffer=buffer,
            device="cpu",
        )

        self.assertIsNotNone(eval_res.rho.cex_cap)
        self.assertLess(eval_res.rho.rho, 1.0)

    def test_estimate_rho_respects_origin_exclusion(self) -> None:
        lyap_model = _QuadraticLyapunov()
        bounds = np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32)
        rho_config = RhoEstimationConfig(
            train_bounds=bounds,
            samples=8,
            descent_steps=0,
            rho_growth_gamma=1.0,
            estimate_quantile=0.05,
            rho_min=1e-4,
            origin_exclusion=0.5,
        )

        buffer = CEGISBuffer(
            initial_states=th.tensor([[0.2, 0.2]], dtype=th.float32),
            state_buffer_limit=16,
            cex_buffer_limit=8,
            lb=th.tensor([-2.0, -2.0]),
            ub=th.tensor([2.0, 2.0]),
            device=th.device("cpu"),
        )
        buffer.register_cex(
            th.tensor([[0.2, 0.2]], dtype=th.float32),
            objective=lambda x: -x.sum(dim=-1),
        )

        def condition_evaluator(x: th.Tensor) -> th.Tensor:
            return th.ones(x.shape[0], dtype=th.float32)

        eval_res, _ = estimate_rho(
            lyap_model=lyap_model,
            config=rho_config,
            condition_evaluator=condition_evaluator,
            state_buffer=buffer,
            device="cpu",
        )

        self.assertIsNone(eval_res.rho.cex_cap)

    def test_boundary_term_diagnostics_nan_fallback(self) -> None:
        lyap_model = _QuadraticLyapunov()
        boundary_x = th.tensor([[1.0, 1.0]], dtype=th.float32)
        diag = _boundary_term_diagnostics(lyap_model, boundary_x, quantile=0.5)

        self.assertTrue(np.isnan(diag.feature_term_quantile))
        self.assertTrue(np.isnan(diag.linear_term_quantile))
        self.assertTrue(np.isnan(diag.feature_term_mean))
        self.assertTrue(np.isnan(diag.linear_term_mean))


if __name__ == "__main__":
    unittest.main(verbosity=2)
