import unittest
from unittest.mock import patch

import numpy as np
import torch as th
import torch.nn as nn

from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.counterexample import (
    estimate_rho_from_boundary,
    find_counter_examples,
)
from lcil.lyapunov_learning.loss import FormalPositivityLoss, LyapunovTrainingLoss
from lcil.lyapunov_learning.trainer import LyapunovTrainer


class _FirstCoordinateValue(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x[:, :1]


class _ZeroPolicy(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)


class _IdentityDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return x


class _QuadraticLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x.pow(2).sum(dim=1, keepdim=True)


class _TrainableQuadraticLyapunov(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(th.ones(1, dtype=th.float32))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.scale * x.pow(2).sum(dim=1, keepdim=True)


class _LinearValue(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        with th.no_grad():
            self.linear.weight.fill_(1.0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(x)


class TestLyapunovCounterexamples(unittest.TestCase):
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
            rho = estimate_rho_from_boundary(_FirstCoordinateValue(), config)

        expected = 1.5 * th.quantile(boundary_points[:, 0], q=0.5).item()
        self.assertAlmostEqual(rho, expected, places=6)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)