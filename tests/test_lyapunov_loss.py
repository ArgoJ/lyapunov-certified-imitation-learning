import unittest
import numpy as np
import torch as th
import torch.nn as nn

from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.loss import (
    BoundedStateSamplingModule,
    ConditionLirpaLoss,
    FormalPositivityLoss,
    InvarianceViolation,
    ParameterL1Loss,
    PolicyRegularizationLoss,
    RFactorFrobeniusLoss,
)
from lcil.lyapunov_learning.models import NeuralLyapunovCandidate
from lcil.utils.base_models import MLP


class _MockRFactorModel(nn.Module):
    def __init__(self, n: int = 2) -> None:
        super().__init__()
        self.r_factor = nn.Parameter(th.eye(n))


class _PlainPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.fc(x)


class TestLyapunovLossComponents(unittest.TestCase):
    def test_invariance_violation(self) -> None:
        bounds = th.tensor([[-1.0, -2.0], [1.0, 2.0]])
        module = InvarianceViolation(state_bounds=bounds)

        # In bounds -> 0 violation
        x_inside = th.tensor([[0.0, 0.0], [0.5, -1.0]])
        viol_inside = module(x_inside)
        th.testing.assert_close(viol_inside, th.zeros((2, 1)))

        # Out of bounds -> positive violation
        x_outside = th.tensor([[2.0, 0.0], [0.0, -3.0]])
        viol_outside = module(x_outside)
        self.assertGreater(viol_outside[0, 0].item(), 0.0)
        self.assertGreater(viol_outside[1, 0].item(), 0.0)

    def test_bounded_state_sampling_needs_resample(self) -> None:
        bounds = th.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        module = BoundedStateSamplingModule(state_bounds=bounds, num_samples=10, resample_interval=5)

        # Training is True by default
        self.assertTrue(module.training)

        # Eval mode -> should not resample
        module.eval()
        self.assertFalse(module._needs_resample())

        # Back to training, interval <= 0 -> should not resample
        module.train()
        module.resample_interval = 0
        self.assertFalse(module._needs_resample())

        # Step counter exceeding max interval -> forces resample
        module.resample_interval = 5
        module._step_counter = module._max_resample_interval + 1
        self.assertTrue(module._needs_resample())

        # step_sampling() executes resample and resets counter
        resampled = module.step_sampling()
        self.assertTrue(resampled)
        self.assertEqual(module._step_counter, 0)

    def test_r_factor_frobenius_loss(self) -> None:
        model = _MockRFactorModel(n=2)
        loss_fn = RFactorFrobeniusLoss(lyap_model=model)

        # Initially zero
        self.assertAlmostEqual(loss_fn().item(), 0.0)

        # After modifying r_factor, loss is positive
        with th.no_grad():
            model.r_factor.add_(th.ones(2, 2))
        self.assertGreater(loss_fn().item(), 0.0)

    def test_parameter_l1_loss(self) -> None:
        param1 = nn.Parameter(th.tensor([1.0, -2.0]))
        param2 = nn.Parameter(th.tensor([[0.5]]))
        loss_fn = ParameterL1Loss(params=[param1, param2])

        loss_val = loss_fn()
        self.assertGreater(loss_val.item(), 0.0)

        # Setting empty params sets dynamic weight to 0.0
        loss_fn.set_train_params([])
        self.assertEqual(loss_fn._dyn_weight, 0.0)

    def test_policy_regularization_loss_fallback_and_sampling(self) -> None:
        policy = _PlainPolicy()
        bounds = th.tensor([[-1.0, 1.0], [-1.0, 1.0]])

        loss_fn = PolicyRegularizationLoss(
            policy=policy,
            state_bounds=bounds,
            num_samples=8,
            resample_interval=10,
        )

        # Fallback to forward when no forward_raw
        loss = loss_fn()
        self.assertGreaterEqual(loss.item(), 0.0)

        # Calling step_sampling directly
        res = loss_fn.step_sampling()
        self.assertIsInstance(res, bool)

    def test_lyapunov_positivity_lower_bound_validation(self) -> None:
        bounds = th.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "Lyapunov model must be provided"):
            FormalPositivityLoss(lyap_model=None, train_bounds=bounds)

    def test_condition_lirpa_loss_empty_regions(self) -> None:
        policy = _PlainPolicy()
        feature_net = MLP(layer_dims=[2, 4, 1], activations=["relu", "identity"])
        lyap = NeuralLyapunovCandidate(feature_net=feature_net, state_dim=2)
        class _AddDyn(nn.Module):
            def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
                return x + u

        cfg = LyapunovTrainingConfig(
            state_dim=2,
            state_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]]),
            use_relative_decrease=False,
        )
        loss_fn = ConditionLirpaLoss(
            policy_model=policy,
            lyap_model=lyap,
            dyn_model=_AddDyn(),
            config=cfg,
        )

        empty_regions = th.empty((0, 2, 2))
        out = loss_fn(empty_regions)
        self.assertEqual(out.item(), 0.0)
        self.assertTrue(out.requires_grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
