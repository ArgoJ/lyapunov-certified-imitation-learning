import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
from shared_utils import (
    _FirstCoordinateValue,
    _IdentityDynamics,
    _LinearValue,
    _QuadraticLyapunov,
    _TrainableQuadraticLyapunov,
    _ZeroPolicy,
)

from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.buffer import BoundaryStateBuffer, DynamicStateBuffer
from lcil.lyapunov_learning.counterexample import (
    BoundaryRhoDiagnostics,
    estimate_rho_from_boundary_diagnostics,
    find_counter_examples,
)
from lcil.lyapunov_learning.loss import FormalPositivityLoss, LyapunovTrainingLoss
from lcil.lyapunov_learning.trainer import LyapunovTrainer, LyapunovTrainingResult
from lcil.lyapunov_learning.utils import ThresholdMonitor


class TestLyapunovCounterexamples(unittest.TestCase):
    def test_rho_threshold_monitor_triggers_after_consecutive_low_values(self) -> None:
        monitor = ThresholdMonitor(threshold=1.0, patience=3)

        outputs = [monitor.update(value) for value in (0.8, 0.9, 1.2, 0.7, 0.6, 0.5)]

        self.assertEqual(outputs, [False, False, False, False, False, True])
        self.assertEqual(monitor.value_history, [0.8, 0.9, 1.2, 0.7, 0.6, 0.5])
        self.assertEqual(monitor.consecutive_low, 3)

    def test_build_scaled_state_bounds_supports_scalar_and_vector_stages(self) -> None:
        base_bounds = np.array([[-2.0, -4.0], [2.0, 4.0]], dtype=np.float32)

        scaled = LyapunovTrainer._build_scaled_state_bounds(
            base_bounds=base_bounds,
            bound_scales=[0.5, [0.25, 0.75]],
        )

        first_bounds, first_scale = scaled[0]
        second_bounds, second_scale = scaled[1]

        np.testing.assert_allclose(first_scale, np.array([0.5, 0.5], dtype=np.float32))
        np.testing.assert_allclose(first_bounds, np.array([[-1.0, -2.0], [1.0, 2.0]], dtype=np.float32))
        np.testing.assert_allclose(second_scale, np.array([0.25, 0.75], dtype=np.float32))
        np.testing.assert_allclose(second_bounds, np.array([[-0.5, -3.0], [0.5, 3.0]], dtype=np.float32))

    def test_estimate_rho_uses_configured_quantile(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=2,
            state_bounds=np.array([[-4.0, -1.0], [4.0, 1.0]], dtype=np.float32),
            rho_boundary_samples=4,
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
            "lcil.lyapunov_learning.counterexample.sample_boundary_points",
            return_value=(boundary_points, face_dims, is_ub),
        ):
            rho = estimate_rho_from_boundary_diagnostics(_FirstCoordinateValue(), config).rho

        expected = 1.5 * th.quantile(boundary_points[:, 0], q=0.5).item()
        self.assertAlmostEqual(rho, expected, places=6)

    def test_estimate_rho_reuses_boundary_buffer_low_values(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=2,
            state_bounds=np.array([[-12.0, -1.0], [12.0, 1.0]], dtype=np.float32),
            rho_boundary_samples=2,
            rho_boundary_buffer_size=2,
            rho_descent_steps=0,
            rho_growth_gamma=1.0,
            rho_estimate_quantile=1.0,
        )
        first_boundary = th.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=th.float32)
        second_boundary = th.tensor([[9.0, 0.0], [10.0, 0.0]], dtype=th.float32)
        face_dims = th.zeros(2, dtype=th.long)
        is_ub = th.ones(2, dtype=th.bool)
        boundary_buffer = BoundaryStateBuffer(state_dim=2, max_size=2, device="cpu")

        with patch(
            "lcil.lyapunov_learning.counterexample.sample_boundary_points",
            side_effect=[
                (first_boundary, face_dims, is_ub),
                (second_boundary, face_dims, is_ub),
            ],
        ):
            first_rho = estimate_rho_from_boundary_diagnostics(
                _FirstCoordinateValue(),
                config,
                boundary_buffer=boundary_buffer,
            ).rho
            second_rho = estimate_rho_from_boundary_diagnostics(
                _FirstCoordinateValue(),
                config,
                boundary_buffer=boundary_buffer,
            ).rho

        self.assertAlmostEqual(first_rho, 2.0, places=6)
        self.assertAlmostEqual(second_rho, 2.0, places=6)

    def test_counterexample_mining_respects_current_rho_gate(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            train_policy_model=False,
            adversarial_samples=256,
            counterexample_steps=0,
            adversarial_step_size=0.0,
            condition_tolerance=1e-6,
        )
        loss_module = LyapunovTrainingLoss(
            policy_model=_ZeroPolicy(),
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            device="cpu",
        )

        th.manual_seed(0)
        rho_estimate = 0.1
        gated_cex = find_counter_examples(
            objective=lambda x: loss_module.mining_objective(x_batch=x, rho_estimate=rho_estimate),
            config=config,
        )
        gated_values = loss_module.lyap_model(gated_cex).flatten()

        self.assertGreater(gated_cex.shape[0], 0)
        self.assertTrue(th.all(gated_values <= rho_estimate + 1e-6).item())

    def test_dynamic_state_buffer_keeps_most_violating_counterexamples(self) -> None:
        initial_states = th.zeros((4, 1), dtype=th.float32)
        state_buffer = DynamicStateBuffer(
            initial_states=initial_states,
            max_size=16,
            cex_buffer_size=3,
            device=th.device("cpu"),
        )

        state_buffer.register_cex(
            th.tensor([[0.2], [0.4]], dtype=th.float32),
            objective=lambda x: -x,
        )
        state_buffer.register_cex(
            th.tensor([[0.1], [0.9]], dtype=th.float32),
            objective=lambda x: -x,
        )

        retained = state_buffer.cexs.flatten()
        expected = th.tensor([0.9, 0.4, 0.2], dtype=th.float32)

        self.assertEqual(state_buffer.state_count, 4)
        self.assertEqual(state_buffer.cex_count, 3)
        self.assertEqual(len(state_buffer), 7)
        self.assertTrue(th.allclose(retained, expected))

    def test_dynamic_state_buffer_sample_returns_requested_batch_size(self) -> None:
        state_buffer = DynamicStateBuffer(
            initial_states=th.tensor([[1.0], [2.0]], dtype=th.float32),
            max_size=4,
            device=th.device("cpu"),
        )

        batch = state_buffer.sample(batch_size=5)

        self.assertEqual(batch.shape, (5, 1))
        self.assertTrue(th.all((batch == 1.0) | (batch == 2.0)).item())

    def test_dynamic_state_buffer_sample_uses_regular_and_cex_pools_separately(self) -> None:
        state_buffer = DynamicStateBuffer(
            initial_states=th.tensor([[1.0], [2.0]], dtype=th.float32),
            max_size=8,
            device=th.device("cpu"),
        )
        state_buffer.register_cex(th.tensor([[10.0]], dtype=th.float32))

        batch = state_buffer.sample(batch_size=4, cex_fraction=0.5)
        batch_values = batch.flatten()

        self.assertEqual(batch.shape, (4, 1))
        self.assertEqual(int((batch_values == 10.0).sum().item()), 1)
        self.assertTrue(th.all((batch_values != 10.0) <= ((batch_values == 1.0) | (batch_values == 2.0))).item())

    def test_dynamic_state_buffer_rejects_empty_initial_states(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial_states cannot be empty"):
            DynamicStateBuffer(
                initial_states=th.empty((0, 1), dtype=th.float32),
                max_size=4,
                device=th.device("cpu"),
            )

    def test_dynamic_state_buffer_sample_clamps_out_of_range_cex_fraction(self) -> None:
        state_buffer = DynamicStateBuffer(
            initial_states=th.tensor([[1.0], [2.0]], dtype=th.float32),
            max_size=4,
            device=th.device("cpu"),
        )
        state_buffer.register_cex(th.tensor([[10.0]], dtype=th.float32))

        batch = state_buffer.sample(batch_size=4, cex_fraction=2.0)
        batch_values = batch.flatten()

        self.assertEqual(batch.shape, (4, 1))
        self.assertEqual(int((batch_values == 10.0).sum().item()), 1)
        self.assertTrue(th.all((batch_values == 10.0) | (batch_values == 1.0) | (batch_values == 2.0)).item())

    def test_trainer_mining_uses_current_rho_estimate(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            train_policy_model=False,
            adversarial_samples=256,
            counterexample_steps=0,
            adversarial_step_size=0.0,
            condition_tolerance=1e-6,
        )
        trainer = LyapunovTrainer(
            policy_model=_ZeroPolicy(),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
        )

        th.manual_seed(0)
        rho_estimate = 0.1
        mined_cex = trainer._mine_new_counterexamples(rho_estimate=rho_estimate)
        mined_values = trainer.lyap_model(mined_cex).flatten()

        self.assertGreater(mined_cex.shape[0], 0)
        self.assertTrue(th.all(mined_values <= rho_estimate + 1e-6).item())

    def test_trainer_returns_aborted_result_after_sustained_low_rho(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            initial_sample_size=4,
            batch_size=2,
            outer_epochs=3,
            steps_per_epoch=1,
            counterexample_every=100,
            train_policy_model=False,
        )
        trainer = LyapunovTrainer(
            policy_model=_ZeroPolicy(),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            rho_monitor=ThresholdMonitor(threshold=1.0, patience=2),
        )
        rho_diagnostics = BoundaryRhoDiagnostics(
            rho=0.5,
            boundary_quantile=0.5,
            boundary_mean=0.5,
            feature_term_quantile=0.0,
            linear_term_quantile=0.5,
            feature_term_mean=0.0,
            linear_term_mean=0.5,
            feature_term_mean_share=0.0,
            linear_term_mean_share=1.0,
            r_factor_fro_norm=0.0,
        )

        with patch(
            "lcil.lyapunov_learning.trainer.estimate_rho_from_boundary_diagnostics",
            return_value=rho_diagnostics,
        ):
            train_result = trainer.train()

        self.assertTrue(train_result.aborted)
        self.assertIs(trainer.results, train_result)
        self.assertEqual(
            train_result.abort_reason,
            "Lyapunov training aborted after 2 consecutive rho estimates below 1.000.",
        )
        self.assertIsNotNone(trainer.metrics)
        assert trainer.metrics is not None
        self.assertEqual(trainer.metrics.outer_iterations_completed, 1)

    def test_train_with_scaled_bounds_returns_completed_stages_before_abort(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
            train_policy_model=False,
        )
        trainer = LyapunovTrainer(
            policy_model=_ZeroPolicy(),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            rho_monitor=ThresholdMonitor(threshold=1.0, patience=2),
        )

        def _fake_train(stage_self: LyapunovTrainer) -> LyapunovTrainingResult:
            stage_upper = float(stage_self.config.state_bounds[1, 0])
            if stage_upper > 1.0:
                stage_self.results = LyapunovTrainingResult(
                    rho_estimate=stage_upper,
                    num_mined_counterexamples=0,
                    train_time=0.0,
                    aborted=True,
                    abort_reason="rho monitor triggered",
                )
                return stage_self.results
            stage_self.results = LyapunovTrainingResult(
                rho_estimate=stage_upper,
                num_mined_counterexamples=0,
                train_time=0.0,
            )
            return stage_self.results

        with patch.object(LyapunovTrainer, "train", autospec=True, side_effect=_fake_train):
            curriculum_result = trainer.train_with_scaled_bounds([0.5, 1.0])

        self.assertTrue(curriculum_result.aborted)
        self.assertEqual(curriculum_result.abort_reason, "rho monitor triggered")
        self.assertEqual(curriculum_result.aborted_stage_index, 1)
        self.assertEqual(len(curriculum_result.stages), 1)
        self.assertIsNotNone(curriculum_result.final_result)
        assert curriculum_result.final_result is not None
        self.assertTrue(curriculum_result.final_result.aborted)
        self.assertIsNotNone(curriculum_result.last_completed_result)
        assert curriculum_result.last_completed_result is not None
        np.testing.assert_allclose(
            curriculum_result.stages[0].state_bounds,
            np.array([[-1.0], [1.0]], dtype=np.float32),
        )
        self.assertAlmostEqual(curriculum_result.last_completed_result.rho_estimate, 1.0, places=6)

    def test_train_with_scaled_bounds_returns_aborted_result_when_first_stage_aborts(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
            train_policy_model=False,
        )
        trainer = LyapunovTrainer(
            policy_model=_ZeroPolicy(),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            rho_monitor=ThresholdMonitor(threshold=1.0, patience=2),
        )

        def _always_abort(stage_self: LyapunovTrainer) -> LyapunovTrainingResult:
            stage_self.results = LyapunovTrainingResult(
                rho_estimate=0.5,
                num_mined_counterexamples=0,
                train_time=0.0,
                aborted=True,
                abort_reason="rho monitor triggered",
            )
            return stage_self.results

        with patch.object(LyapunovTrainer, "train", autospec=True, side_effect=_always_abort):
            curriculum_result = trainer.train_with_scaled_bounds([0.5, 1.0])

        self.assertTrue(curriculum_result.aborted)
        self.assertEqual(curriculum_result.abort_reason, "rho monitor triggered")
        self.assertEqual(curriculum_result.aborted_stage_index, 0)
        self.assertEqual(len(curriculum_result.stages), 0)
        self.assertIsNone(curriculum_result.last_completed_result)
        self.assertIsNotNone(curriculum_result.final_result)
        assert curriculum_result.final_result is not None
        self.assertTrue(curriculum_result.final_result.aborted)

    def test_formal_positivity_backward_returns_expected_lower_bound(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            train_policy_model=False,
        )
        loss_module = LyapunovTrainingLoss(
            policy_model=_ZeroPolicy(),
            lyap_model=_LinearValue(),
            dyn_model=_IdentityDynamics(),
            config=config,
            device="cpu",
        )
        lower = loss_module.compute_lyapunov_lower_bound(method="backward")

        self.assertAlmostEqual(float(lower.item()), -1.0, places=6)

    def test_trainer_reuses_cached_bounded_model_for_formal_positivity(self) -> None:
        lyap_model = _LinearValue()
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            train_policy_model=False,
        )
        with patch.object(
            FormalPositivityLoss,
            "_build_lyapunov_bounded_model",
            autospec=True,
            side_effect=FormalPositivityLoss._build_lyapunov_bounded_model,
        ) as build_bounded_model:
            trainer = LyapunovTrainer(
                policy_model=_ZeroPolicy(),
                lyap_model=lyap_model,
                dyn_model=_IdentityDynamics(),
                config=config,
            )
            first_loss = trainer.loss_module.formal_positivity_loss()
            with th.no_grad():
                lyap_model.linear.weight.fill_(2.0)
            second_loss = trainer.loss_module.formal_positivity_loss()

        self.assertEqual(build_bounded_model.call_count, 1)
        self.assertAlmostEqual(float(first_loss.item()), 1.0, places=6)
        self.assertAlmostEqual(float(second_loss.item()), 2.0, places=6)

    def test_train_with_scaled_bounds_runs_stages_and_updates_trainer(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
            train_policy_model=False,
        )
        trainer = LyapunovTrainer(
            policy_model=_ZeroPolicy(),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
        )

        stage_bounds_seen: list[np.ndarray] = []

        def _fake_train(stage_self: LyapunovTrainer) -> LyapunovTrainingResult:
            stage_bounds_seen.append(np.asarray(stage_self.config.state_bounds, dtype=np.float32).copy())
            stage_self.results = LyapunovTrainingResult(
                rho_estimate=float(stage_self.config.state_bounds[1, 0]),
                num_mined_counterexamples=stage_self.config.state_dim,
                train_time=0.0,
            )
            stage_self.metrics = None
            return stage_self.results

        with patch.object(LyapunovTrainer, "train", autospec=True, side_effect=_fake_train):
            curriculum_result = trainer.train_with_scaled_bounds([0.5, 1.0])

        expected_stage_bounds = [
            np.array([[-1.0], [1.0]], dtype=np.float32),
            np.array([[-2.0], [2.0]], dtype=np.float32),
        ]
        self.assertEqual(len(curriculum_result.stages), 2)
        np.testing.assert_allclose(stage_bounds_seen[0], expected_stage_bounds[0])
        np.testing.assert_allclose(stage_bounds_seen[1], expected_stage_bounds[1])
        np.testing.assert_allclose(trainer.config.state_bounds, expected_stage_bounds[1])
        self.assertAlmostEqual(curriculum_result.final_result.rho_estimate, 2.0, places=6)

    def test_trainer_save_writes_training_result_json(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            train_policy_model=False,
        )
        trainer = LyapunovTrainer(
            policy_model=_ZeroPolicy(),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
        )
        trainer.results = LyapunovTrainingResult(
            rho_estimate=0.75,
            num_mined_counterexamples=3,
            train_time=1.25,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            trainer.save(out_dir)
            loaded_result = LyapunovTrainingResult.load(out_dir)

        self.assertAlmostEqual(loaded_result.rho_estimate, 0.75, places=6)
        self.assertEqual(loaded_result.num_mined_counterexamples, 3)
        self.assertEqual(loaded_result.train_time, 1.25)
        self.assertEqual(loaded_result.lyap_model_path, out_dir / "lyapunov_model.pt")
        self.assertEqual(loaded_result.policy_model_path, out_dir / "policy_model.pt")


if __name__ == "__main__":
    unittest.main(verbosity=2)