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
from lcil.lyapunov_learning.models import ClosedLoopLyapunovTrainingVerifier
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

    def test_global_counterexample_mining_ignores_current_rho_gate(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            adversarial_samples=256,
            counterexample_steps=0,
            adversarial_step_size=0.0,
            condition_tolerance=1e-6,
        )
        verifier = ClosedLoopLyapunovTrainingVerifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            lbx=th.tensor([[-1.0]], dtype=th.float32),
            ubx=th.tensor([[1.0]], dtype=th.float32),
            invariance_weight=1.0,
            kappa=0.1,
        )

        th.manual_seed(0)
        global_cex = find_counter_examples(verifier, config)
        global_values = verifier.lyap(global_cex).flatten()

        self.assertTrue(th.any(global_values > 0.1 + 1e-6).item())

    def test_formal_positivity_backward_returns_expected_lower_bound(self) -> None:
        lyap_model = _LinearValue()
        bounded_model = LyapunovTrainer._build_lyapunov_bounded_model(
            lyap_model=lyap_model,
            input_size=1,
            device=th.device("cpu"),
        )
        lower = LyapunovTrainer._compute_lyapunov_lower_bound(
            lbx=th.tensor([0.0], dtype=th.float32),
            ubx=th.tensor([1.0], dtype=th.float32),
            device=th.device("cpu"),
            bounded_model=bounded_model,
            method="backward",
        )

        self.assertAlmostEqual(float(lower.item()), 0.0, places=6)

    def test_trainer_reuses_cached_bounded_model_for_formal_positivity(self) -> None:
        lyap_model = _LinearValue()
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            train_policy_model=False,
        )
        with patch.object(
            LyapunovTrainer,
            "build_lyapunov_bounded_model",
            wraps=LyapunovTrainer._build_lyapunov_bounded_model,
        ) as build_bounded_model:
            trainer = LyapunovTrainer(
                policy_model=_ZeroPolicy(),
                lyap_model=lyap_model,
                dyn_model=_IdentityDynamics(),
                config=config,
            )
            first_loss = trainer._formal_positivity_loss()
            with th.no_grad():
                lyap_model.linear.weight.fill_(2.0)
            second_loss = trainer._formal_positivity_loss()

        self.assertEqual(build_bounded_model.call_count, 1)
        self.assertAlmostEqual(float(first_loss.item()), 1.0, places=6)
        self.assertAlmostEqual(float(second_loss.item()), 2.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)