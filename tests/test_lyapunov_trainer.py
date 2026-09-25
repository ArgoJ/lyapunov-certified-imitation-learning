import unittest
import tempfile
from typing import Any
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
from shared_utils import (
    _IdentityDynamics,
    _LinearValue,
    _TrainableQuadraticLyapunov,
    _ZeroPolicy,
)

from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.loss import FormalPositivityLoss
from lcil.lyapunov_learning.sublevel import (
    BoundaryRhoEstimate,
    BoundaryTermDiagnostics,
    BoundaryRhoEvaluation,
)
from lcil.lyapunov_learning.trainer import LyapunovTrainer, LyapunovTrainingResult
from lcil.lyapunov_learning.utils import ThresholdMonitor


class _SingleWeightPolicy(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        with th.no_grad():
            self.linear.weight.fill_(weight)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(x)


class _ControlAffineDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return x + u


class TestLyapunovTrainer(unittest.TestCase):
    def test_build_scaled_state_bounds_supports_scalar_and_vector_stages(self) -> None:
        base_bounds = np.array([[-2.0, -4.0], [2.0, 4.0]], dtype=np.float32)

        scaled = LyapunovTrainer._build_scaled_train_bounds(
            base_bounds=base_bounds,
            bound_scales=[0.5, [0.25, 0.75]],
        )

        first_bounds, first_scale = scaled[0]
        second_bounds, second_scale = scaled[1]

        np.testing.assert_allclose(first_scale, np.array([0.5, 0.5], dtype=np.float32))
        np.testing.assert_allclose(first_bounds, np.array([[-1.0, -2.0], [1.0, 2.0]], dtype=np.float32))
        np.testing.assert_allclose(second_scale, np.array([0.25, 0.75], dtype=np.float32))
        np.testing.assert_allclose(second_bounds, np.array([[-0.5, -3.0], [0.5, 3.0]], dtype=np.float32))

    def test_build_scaled_state_bounds_validations(self) -> None:
        base_bounds = np.array([[-2.0, -4.0], [2.0, 4.0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "must have shape"):
            LyapunovTrainer._build_scaled_train_bounds(base_bounds[0], bound_scales=[1.0])

        with self.assertRaisesRegex(ValueError, "at least one stage"):
            LyapunovTrainer._build_scaled_train_bounds(base_bounds, bound_scales=[])

        with self.assertRaisesRegex(ValueError, "Each bound scale must be"):
            LyapunovTrainer._build_scaled_train_bounds(base_bounds, bound_scales=[[1.0, 2.0, 3.0]])

        with self.assertRaisesRegex(ValueError, "All bound scales must be positive"):
            LyapunovTrainer._build_scaled_train_bounds(base_bounds, bound_scales=[-0.5])

    def test_enable_policy_training_rebuilds_optimizer_and_preserves_state(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            state_buffer_limit=4,
            batch_size=2,
            outer_epochs=2,
            steps_per_epoch=1,
            cex_every=100,
            policy_epochs=1,
        )
        trainer = LyapunovTrainer(
            policy_model=_SingleWeightPolicy(weight=1.0),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_ControlAffineDynamics(),
            config=config,
        )

        policy_param = next(trainer.policy_model.parameters())
        lyap_param = next(trainer.lyap_model.parameters())

        self.assertFalse(trainer._curr_policy_train_status)
        self.assertFalse(trainer.policy_model.training)

        trainer.optimizer.zero_grad()
        lyap_param.sum().backward()
        trainer.optimizer.step()

        old_state = trainer.optimizer.state[lyap_param]
        self.assertIn("exp_avg", old_state)

        trainer._enable_policy_training()

        self.assertTrue(trainer._curr_policy_train_status)
        self.assertTrue(trainer.policy_model.training)
        self.assertTrue(any(param is policy_param for param in trainer.optimizer.param_groups[-1]["params"]))
        self.assertIn(lyap_param, trainer.optimizer.state)
        self.assertTrue(th.allclose(trainer.optimizer.state[lyap_param]["exp_avg"], old_state["exp_avg"]))

    def test_trainer_returns_aborted_result_after_sustained_low_rho(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            state_buffer_limit=4,
            batch_size=2,
            outer_epochs=3,
            steps_per_epoch=1,
            cex_every=100,
        )
        trainer = LyapunovTrainer(
            policy_model=_ZeroPolicy(),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            rho_monitor=ThresholdMonitor(threshold=1.0, patience=2),
        )
        rho_eval = BoundaryRhoEvaluation(
            rho=BoundaryRhoEstimate(
                rho=0.5,
                boundary_quantile=0.5,
                boundary_mean=0.5,
            ),
            terms=BoundaryTermDiagnostics(
                feature_term_quantile=0.0,
                linear_term_quantile=0.5,
                feature_term_mean=0.0,
                linear_term_mean=0.5,
                feature_term_mean_share=0.0,
                linear_term_mean_share=1.0,
            ),
        )

        with patch(
            "lcil.lyapunov_learning.trainer.estimate_rho_from_boundary",
            return_value=(rho_eval, th.zeros((1, 1))),
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
        )
        trainer = LyapunovTrainer(
            policy_model=_ZeroPolicy(),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            rho_monitor=ThresholdMonitor(threshold=1.0, patience=2),
        )

        def _fake_train(stage_self: LyapunovTrainer, *args: Any, **kwargs: Any) -> LyapunovTrainingResult:
            stage_upper = float(stage_self.config.train_bounds[1, 0])
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
            curriculum_result.stages[0].train_bounds,
            np.array([[-1.0], [1.0]], dtype=np.float32),
        )
        self.assertAlmostEqual(curriculum_result.last_completed_result.rho_estimate, 1.0, places=6)

    def test_train_with_scaled_bounds_returns_aborted_result_when_first_stage_aborts(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
        )
        trainer = LyapunovTrainer(
            policy_model=_ZeroPolicy(),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            rho_monitor=ThresholdMonitor(threshold=1.0, patience=2),
        )

        def _always_abort(stage_self: LyapunovTrainer, *args: Any, **kwargs: Any) -> LyapunovTrainingResult:
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

    def test_trainer_reuses_cached_bounded_model_for_formal_positivity(self) -> None:
        lyap_model = _LinearValue()
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            formal_positivity_weight=1.0,
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
            first_loss = trainer.loss_module.positivity_loss()
            with th.no_grad():
                lyap_model.linear.weight.fill_(2.0)
            second_loss = trainer.loss_module.positivity_loss()

        self.assertEqual(build_bounded_model.call_count, 1)
        self.assertAlmostEqual(float(first_loss.item()), 1.0, places=6)
        self.assertAlmostEqual(float(second_loss.item()), 2.0, places=6)

    def test_train_with_scaled_bounds_runs_stages_and_updates_trainer(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
        )
        trainer = LyapunovTrainer(
            policy_model=_ZeroPolicy(),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
        )

        stage_bounds_seen: list[np.ndarray] = []

        def _fake_train(stage_self: LyapunovTrainer, *args: Any, **kwargs: Any) -> LyapunovTrainingResult:
            stage_bounds_seen.append(np.asarray(stage_self.config.train_bounds, dtype=np.float32).copy())
            stage_self.results = LyapunovTrainingResult(
                rho_estimate=float(stage_self.config.train_bounds[1, 0]),
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
        np.testing.assert_allclose(trainer.config.train_bounds, expected_stage_bounds[1])
        self.assertAlmostEqual(curriculum_result.final_result.rho_estimate, 2.0, places=6)

    def test_train_with_scaled_bounds_regularizes_to_true_initial_policy(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-2.0], [2.0]], dtype=np.float32),
            policy_regularization_weight=1.0,
        )
        policy = _SingleWeightPolicy(weight=1.0)
        trainer = LyapunovTrainer(
            policy_model=policy,
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
        )

        stage_init_weights: list[float] = []

        def _fake_train(stage_self: LyapunovTrainer, *args: Any, **kwargs: Any) -> LyapunovTrainingResult:
            reg_loss = stage_self.loss_module.policy_regularization_loss
            self.assertIsNotNone(reg_loss)
            init_w = float(reg_loss.init_policy.linear.weight.item())
            stage_init_weights.append(init_w)

            # Mutate policy to simulate training drift in this stage
            with th.no_grad():
                stage_self.policy_model.linear.weight.add_(10.0)

            stage_self.results = LyapunovTrainingResult(
                rho_estimate=1.0,
                num_mined_counterexamples=0,
                train_time=0.0,
            )
            stage_self.metrics = None
            return stage_self.results

        with patch.object(LyapunovTrainer, "train", autospec=True, side_effect=_fake_train):
            curriculum_result = trainer.train_with_scaled_bounds([0.5, 1.0])

        self.assertEqual(len(curriculum_result.stages), 2)
        # Both stage 0 and stage 1 must regularize against the original initial policy (weight 1.0),
        # even though stage 0 mutated policy_model to weight 11.0!
        self.assertEqual(stage_init_weights, [1.0, 1.0])
        self.assertAlmostEqual(float(trainer.init_policy_model.linear.weight.item()), 1.0, places=6)

    def test_trainer_save_writes_training_result_json(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
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
