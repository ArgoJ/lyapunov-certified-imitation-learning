import unittest
import numpy as np
import torch as th
import torch.nn as nn

from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.loss import (
    BoundedStateSamplingModule,
    ConditionLirpaLoss,
    EquilibriumLoss,
    FormalPositivityLoss,
    InvarianceViolation,
    LyapunovTrainingLoss,
    ParameterL1Loss,
    PolicyRegularizationLoss,
    RFactorRegularizationLoss,
    RoaSurrogateLoss,
)
from lcil.lyapunov_learning.models import NeuralLyapunovCandidate
from lcil.utils.base_models import MLP
from shared_utils import (
    _IdentityDynamics,
    _LinearValue,
    _ZeroPolicy,
)


class _SingleWeightPolicy(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        with th.no_grad():
            self.linear.weight.fill_(weight)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(x)


class _MockRFactorModel(nn.Module):
    def __init__(self, n: int = 2) -> None:
        super().__init__()
        self.r_factor = nn.Parameter(th.eye(n))

    def _pd_matrix(self) -> th.Tensor:
        return self.r_factor.transpose(0, 1) @ self.r_factor


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

        # Out of bounds -> positive violation and gradient flow
        x_outside = th.tensor([[2.0, 0.0], [0.0, -3.0]], requires_grad=True)
        viol_outside = module(x_outside)
        self.assertGreater(viol_outside[0, 0].item(), 0.0)
        self.assertGreater(viol_outside[1, 0].item(), 0.0)
        viol_outside.sum().backward()
        self.assertIsNotNone(x_outside.grad)
        self.assertGreater(x_outside.grad.abs().sum().item(), 0.0)

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

    def test_r_factor_regularization_loss(self) -> None:
        model = _MockRFactorModel(n=2)
        loss_fn = RFactorRegularizationLoss(lyap_model=model, min_eig=0.02, max_cond=25.0)

        # 1. Initially zero (P = eye(2), eigenvalues are [1.0, 1.0])
        self.assertAlmostEqual(loss_fn().item(), 0.0)

        # 2. After shrinking r_factor below min_eig, floor penalty triggers and gradients flow
        with th.no_grad():
            model.r_factor.copy_(0.01 * th.eye(2))
        floor_loss = loss_fn()
        self.assertGreater(floor_loss.item(), 0.0)
        floor_loss.backward()
        self.assertIsNotNone(model.r_factor.grad)
        self.assertGreater(model.r_factor.grad.abs().sum().item(), 0.0)

        # 3. After making r_factor ill-conditioned (floor fulfilled: lam_min=0.09 > 0.02, cond=100 > 25),
        # condition penalty triggers and gradients flow
        model.r_factor.grad = None
        with th.no_grad():
            model.r_factor.copy_(th.diag(th.tensor([3.0, 0.3])))
        cond_loss = loss_fn()
        self.assertGreater(cond_loss.item(), 0.0)
        cond_loss.backward()
        self.assertIsNotNone(model.r_factor.grad)
        self.assertGreater(model.r_factor.grad.abs().sum().item(), 0.0)

    def test_parameter_l1_loss(self) -> None:
        param1 = nn.Parameter(th.tensor([1.0, -2.0]))
        param2 = nn.Parameter(th.tensor([[0.5]]))
        loss_fn = ParameterL1Loss(params=[param1, param2])

        loss_val = loss_fn()
        self.assertGreater(loss_val.item(), 0.0)
        loss_val.backward()
        self.assertIsNotNone(param1.grad)
        self.assertIsNotNone(param2.grad)
        self.assertGreater(param1.grad.abs().sum().item(), 0.0)
        self.assertGreater(param2.grad.abs().sum().item(), 0.0)

        # Setting empty params sets dynamic weight to 0.0
        loss_fn.set_train_params([])
        self.assertEqual(loss_fn._dyn_weight, 0.0)

    def test_policy_regularization_loss_fallback_and_sampling(self) -> None:
        policy = _PlainPolicy()
        bounds = th.tensor([[-1.0, -1.0], [1.0, 1.0]])

        loss_fn = PolicyRegularizationLoss(
            policy=policy,
            state_bounds=bounds,
            num_samples=8,
            resample_interval=10,
        )

        # Initially zero when policy matches init_policy
        self.assertAlmostEqual(loss_fn().item(), 0.0)

        # Perturb policy bias to induce non-zero loss and verify gradient flow
        with th.no_grad():
            policy.fc.bias.add_(1.0)
        loss = loss_fn()
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(policy.fc.bias.grad)
        self.assertGreater(policy.fc.bias.grad.abs().sum().item(), 0.0)

        # Calling step_sampling directly
        res = loss_fn.step_sampling()
        self.assertIsInstance(res, bool)

    def test_lyapunov_positivity_lower_bound_validation(self) -> None:
        bounds = th.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "Lyapunov model must be provided"):
            FormalPositivityLoss(lyap_model=None, train_bounds=bounds)

        feature_net = MLP(layer_dims=[2, 4, 1], activations=["relu", "identity"])
        lyap = NeuralLyapunovCandidate(feature_net=feature_net, state_dim=2)
        pos_loss_fn = FormalPositivityLoss(lyap_model=lyap, train_bounds=bounds)
        lb = pos_loss_fn.compute_lyapunov_lower_bound()
        lb.sum().backward()
        self.assertIsNotNone(lyap.r_factor.grad)
        self.assertGreater(lyap.r_factor.grad.abs().sum().item(), 0.0)

    def test_condition_lirpa_loss(self) -> None:
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

        # Empty regions: zero loss but requires_grad True
        empty_regions = th.empty((0, 2, 2))
        out = loss_fn(empty_regions)
        self.assertEqual(out.item(), 0.0)
        self.assertTrue(out.requires_grad)

        # Non-empty region: non-zero loss and gradient flow to both lyap and policy
        regions = th.tensor([[[-0.5, -0.5], [0.5, 0.5]]])
        out_non_empty = loss_fn(regions)
        self.assertGreater(out_non_empty.item(), 0.0)
        out_non_empty.backward()
        self.assertIsNotNone(lyap.r_factor.grad)
        self.assertGreater(lyap.r_factor.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(policy.fc.weight.grad)
        self.assertGreater(policy.fc.weight.grad.abs().sum().item(), 0.0)

    def test_equilibrium_loss(self) -> None:
        feature_net = MLP(layer_dims=[2, 4, 1], activations=["relu", "identity"])
        lyap = NeuralLyapunovCandidate(feature_net=feature_net, state_dim=2)
        lyap.set_x_star(th.tensor([0.5, -0.5]))
        loss_fn = EquilibriumLoss(model=lyap, state_dim=2)
        loss = loss_fn()
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(lyap.r_factor.grad)
        self.assertGreater(lyap.r_factor.grad.abs().sum().item(), 0.0)

    def test_roa_surrogate_loss(self) -> None:
        cfg = LyapunovTrainingConfig(state_dim=2, state_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]]))
        loss_fn = RoaSurrogateLoss(config=cfg)
        v_candidates = th.tensor([0.05, 0.10], requires_grad=True)
        loss = loss_fn(v_candidates, rho_estimate=0.01)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(v_candidates.grad)
        self.assertGreater(v_candidates.grad.abs().sum().item(), 0.0)

    def test_condition_violation_with_and_without_margin(self) -> None:
        policy = _PlainPolicy()
        feature_net = MLP(layer_dims=[2, 4, 1], activations=["relu", "identity"])
        lyap = NeuralLyapunovCandidate(feature_net=feature_net, state_dim=2)

        # Dynamic model that contracts slightly: x_next = 0.99 * x
        class _ContractingDyn(nn.Module):
            def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
                return 0.99 * x

        # Set kappa=0.0 (meaning V(x_next) <= V(x_curr) is sufficient for stability)
        # and condition_margin = 0.5
        cfg = LyapunovTrainingConfig(
            state_dim=2,
            state_bounds=np.array([[-2.0, -2.0], [2.0, 2.0]]),
            kappa=0.0,
            condition_margin=0.5,
            use_relative_decrease=False,
        )
        from lcil.lyapunov_learning.loss import LyapunovTrainingLoss
        loss_mod = LyapunovTrainingLoss(
            policy_model=policy,
            lyap_model=lyap,
            dyn_model=_ContractingDyn(),
            config=cfg,
        )

        # Non-zero state where V decreases: V(x_next) < V(x_curr)
        x = th.tensor([[1.0, 1.0]], dtype=th.float32)

        # 1. With margin: decrease violation is relu(V(next) - V(curr) + margin) > 0
        viol_with_margin = loss_mod.condition_violation(x, with_margin=True)
        self.assertGreater(viol_with_margin.item(), 0.0)

        # 2. Without margin: true violation is relu(V(next) - V(curr)) == 0.0
        viol_without_margin = loss_mod.condition_violation(x, with_margin=False)
        self.assertEqual(viol_without_margin.item(), 0.0)

    def test_policy_regularization_tracks_initial_policy_outputs(self) -> None:
        policy = _SingleWeightPolicy(weight=1.0)
        regularization_loss = PolicyRegularizationLoss(policy, state_bounds=th.tensor([[-1.0], [1.0]]), device="cpu")

        self.assertAlmostEqual(float(regularization_loss().item()), 0.0, places=6)

        with th.no_grad():
            policy.linear.weight.fill_(3.0)

        self.assertGreater(float(regularization_loss().item()), 0.0)

    def test_formal_positivity_backward_returns_expected_lower_bound(self) -> None:
        config = LyapunovTrainingConfig(
            state_dim=1,
            state_bounds=np.array([[-1.0], [1.0]], dtype=np.float32),
            formal_positivity_weight=1.0,
        )
        loss_module = LyapunovTrainingLoss(
            policy_model=_ZeroPolicy(),
            lyap_model=_LinearValue(),
            dyn_model=_IdentityDynamics(),
            config=config,
            device="cpu",
        )
        lower = loss_module.positivity_loss.compute_lyapunov_lower_bound(method="backward")

        self.assertAlmostEqual(float(lower.item()), -1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

