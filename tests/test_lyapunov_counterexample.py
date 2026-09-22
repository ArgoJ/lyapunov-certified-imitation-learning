import unittest
import numpy as np
import torch as th
from shared_utils import (
    _IdentityDynamics,
    _QuadraticLyapunov,
    _TrainableQuadraticLyapunov,
    _ZeroPolicy,
)

from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.counterexample import (
    CounterexampleMiningConfig,
    find_counter_examples,
)
from lcil.lyapunov_learning.loss import LyapunovTrainingLoss
from lcil.lyapunov_learning.trainer import LyapunovTrainer


class TestLyapunovCounterexamples(unittest.TestCase):
    def test_counterexample_mining_config_from_training_config(self) -> None:
        bounds = np.array([[-2.0], [2.0]], dtype=np.float32)
        train_cfg = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=bounds,
            cex_step_size=0.02,
            cex_descent_steps=15,
            origin_exclusion=0.1,
        )
        cex_cfg = CounterexampleMiningConfig.from_training_config(train_cfg, descent_steps=8)

        self.assertEqual(cex_cfg.step_size, 0.02)
        self.assertEqual(cex_cfg.descent_steps, 8)
        self.assertEqual(cex_cfg.origin_exclusion, (0.1,))
        np.testing.assert_allclose(cex_cfg.train_bounds, bounds)

    def test_counterexample_mining_respects_current_rho_gate(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            state_buffer_limit=256,
            cex_descent_steps=1,
            cex_step_size=0.01,
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
        initial_states = th.linspace(-1.0, 1.0, 256).unsqueeze(-1)
        cex_config = CounterexampleMiningConfig.from_training_config(config)
        gated_cex, _ = find_counter_examples(
            objective=lambda x: loss_module.mining_objective(x_batch=x, rho_estimate=rho_estimate),
            condition_evaluator=lambda x: loss_module.get_counterexample_mask(x, rho_estimate),
            config=cex_config,
            initial_states=initial_states,
        )
        gated_values = loss_module.lyap_model(gated_cex).flatten()

        self.assertGreater(gated_cex.shape[0], 0)
        self.assertTrue(th.all(gated_values <= rho_estimate + 1e-6).item())

    def test_find_counter_examples_respects_origin_exclusion(self) -> None:
        cex_config = CounterexampleMiningConfig(
            train_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            descent_steps=1,
            origin_exclusion=0.2,
        )
        initial_states = th.tensor([[0.05], [0.1], [0.5]], dtype=th.float32)

        cex_states, _ = find_counter_examples(
            objective=lambda x: x.sum(dim=-1),
            condition_evaluator=lambda x: (th.ones(x.shape[0]), th.ones(x.shape[0], dtype=th.bool)),
            config=cex_config,
            initial_states=initial_states,
            device="cpu",
        )

        self.assertTrue(th.all(th.abs(cex_states) > 0.2).item())

    def test_trainer_mining_uses_current_rho_estimate(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            state_buffer_limit=256,
            cex_descent_steps=1,
            cex_step_size=0.01,
        )
        trainer = LyapunovTrainer(
            policy_model=_ZeroPolicy(),
            lyap_model=_TrainableQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
        )

        th.manual_seed(0)
        rho_estimate = 0.1
        initial_states = th.linspace(-1.0, 1.0, 256).unsqueeze(-1)
        mined_cex, _ = trainer.mine_counterexamples(
            rho_estimate=rho_estimate,
            initial_states=initial_states,
        )
        mined_values = trainer.lyap_model(mined_cex).flatten()

        self.assertGreater(mined_cex.shape[0], 0)
        self.assertTrue(th.all(mined_values <= rho_estimate + 1e-6).item())


if __name__ == "__main__":
    unittest.main(verbosity=2)