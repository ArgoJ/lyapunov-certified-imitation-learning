import unittest
import torch as th
import torch.nn as nn

from lcil.imitation_learning.loss import (
    ActionWeightedMSELoss,
    BaselineDynamicsAwareLoss,
    DynamicsAwareLoss,
    ImitationLearningLossParts,
    ScaledMSELoss,
    StateWeightedMSELoss,
    _as_row_tensor,
    per_sample_mse,
    reference_weights,
    scaled_error,
)


class _MockDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return x + u


class TestImitationLoss(unittest.TestCase):
    """Thorough unit tests for imitation learning loss functions."""

    def test_as_row_tensor_and_helpers(self) -> None:
        t = _as_row_tensor([1.0, 2.0, 3.0], "test_var")
        self.assertEqual(t.shape, (1, 3))

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            _as_row_tensor([], "empty_var")

        # scaled_error and per_sample_mse
        pos = th.tensor([[2.0, 4.0]])
        neg = th.tensor([[1.0, 2.0]])
        scale = th.tensor([[1.0, 2.0]])
        err = scaled_error(pos, neg, scale)
        th.testing.assert_close(err, th.tensor([[1.0, 1.0]]))

        mse = per_sample_mse(pos, neg, scale)
        th.testing.assert_close(mse, th.tensor([[1.0]]))

        # reference_weights
        weights = reference_weights(
            value=pos,
            reference=neg,
            scale=scale,
            center_alpha=1.0,
            min_weight=0.2,
        )
        self.assertTrue(th.all(weights >= 0.2))
        self.assertTrue(th.all(weights <= 1.0))

    def test_scaled_mse_loss(self) -> None:
        loss_fn = ScaledMSELoss(scale=[2.0, 1.0])
        pred = th.tensor([[2.0, 1.0], [4.0, 2.0]])
        target = th.tensor([[0.0, 1.0], [2.0, 1.0]])
        val = loss_fn(pred, target)
        self.assertGreater(val.item(), 0.0)

        with self.assertRaisesRegex(ValueError, "All scale entries must be positive"):
            ScaledMSELoss(scale=[1.0, -0.5])

    def test_action_weighted_mse_loss_validation_and_forward(self) -> None:
        # Invalid params
        with self.assertRaisesRegex(ValueError, "All action_scale entries must be positive"):
            ActionWeightedMSELoss(action_scale=[0.0], action_reference=[0.0])

        with self.assertRaisesRegex(ValueError, "center_alpha must be positive"):
            ActionWeightedMSELoss(action_scale=[1.0], action_reference=[0.0], center_alpha=-1.0)

        with self.assertRaisesRegex(ValueError, "min_weight must be in"):
            ActionWeightedMSELoss(action_scale=[1.0], action_reference=[0.0], min_weight=0.0)

        with self.assertRaisesRegex(ValueError, "min_weight must be in"):
            ActionWeightedMSELoss(action_scale=[1.0], action_reference=[0.0], min_weight=1.5)

        with self.assertRaisesRegex(ValueError, "same shape"):
            ActionWeightedMSELoss(action_scale=[1.0, 2.0], action_reference=[0.0])

        # Valid forward
        loss_fn = ActionWeightedMSELoss(
            action_scale=[1.0, 2.0],
            action_reference=[0.0, 0.0],
            center_alpha=1.5,
            min_weight=0.1,
        )
        pred = th.tensor([[0.5, 1.0]])
        target = th.tensor([[0.0, 0.0]])
        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0.0)

    def test_state_weighted_mse_loss_validation_and_forward(self) -> None:
        # Invalid params
        with self.assertRaisesRegex(ValueError, "All x_scale entries must be positive"):
            StateWeightedMSELoss(x_reference=[0.0], x_scale=[-1.0])

        with self.assertRaisesRegex(ValueError, "center_alpha must be positive"):
            StateWeightedMSELoss(x_reference=[0.0], x_scale=[1.0], center_alpha=0.0)

        with self.assertRaisesRegex(ValueError, "min_weight must be in"):
            StateWeightedMSELoss(x_reference=[0.0], x_scale=[1.0], min_weight=-0.1)

        with self.assertRaisesRegex(ValueError, "same shape"):
            StateWeightedMSELoss(x_reference=[0.0, 1.0], x_scale=[1.0])

        with self.assertRaisesRegex(ValueError, "All action_scale entries must be positive"):
            StateWeightedMSELoss(x_reference=[0.0], x_scale=[1.0], action_scale=[0.0])

        # Valid with custom action_scale
        loss_fn = StateWeightedMSELoss(
            x_reference=[0.0, 0.0],
            x_scale=[1.0, 1.0],
            action_scale=[2.0],
            center_alpha=1.0,
            min_weight=0.25,
        )
        pred = th.tensor([[1.0], [2.0]])
        target = th.tensor([[0.0], [1.0]])
        states = th.tensor([[0.0, 0.0], [1.0, 1.0]])

        # States required
        with self.assertRaisesRegex(ValueError, "requires states"):
            loss_fn(pred, target)

        loss = loss_fn(pred, target, states)
        self.assertGreater(loss.item(), 0.0)

        # Default action_scale expands when action dim > 1
        loss_fn_default_a = StateWeightedMSELoss(
            x_reference=[0.0, 0.0],
            x_scale=[1.0, 1.0],
        )
        pred_2d = th.tensor([[1.0, 2.0]])
        target_2d = th.tensor([[0.0, 0.0]])
        loss_2d = loss_fn_default_a(pred_2d, target_2d, states[:1])
        self.assertGreater(loss_2d.item(), 0.0)

    def test_dynamics_aware_loss(self) -> None:
        dyn = _MockDynamics()

        # No bounds -> returns 0
        loss_no_bounds = DynamicsAwareLoss(dynamics=dyn)
        states = th.tensor([[1.0, 2.0]])
        actions = th.tensor([[0.5, 0.5]])
        self.assertEqual(loss_no_bounds(actions, states).item(), 0.0)

        # With upper and lower bounds
        loss_bounds = DynamicsAwareLoss(
            dynamics=dyn,
            x_min=[-1.0, -1.0],
            x_max=[1.0, 1.0],
        )

        # In-bounds next state [0.5, 0.5] -> 0 penalty
        zero_loss = loss_bounds(th.tensor([[0.5, 0.5]]), th.tensor([[0.0, 0.0]]))
        self.assertEqual(zero_loss.item(), 0.0)

        # Exceeding upper bound [2.0, 2.0] + [0.5, 0.5] = [2.5, 2.5] > [1.0, 1.0]
        pos_loss = loss_bounds(th.tensor([[0.5, 0.5]]), th.tensor([[2.0, 2.0]]))
        self.assertGreater(pos_loss.item(), 0.0)

        # Exceeding lower bound [-2.0, -2.0] + [-0.5, -0.5] = [-2.5, -2.5] < [-1.0, -1.0]
        neg_loss = loss_bounds(th.tensor([[-0.5, -0.5]]), th.tensor([[-2.0, -2.0]]))
        self.assertGreater(neg_loss.item(), 0.0)

        # Only upper bound
        loss_upper_only = DynamicsAwareLoss(dynamics=dyn, x_max=[1.0, 1.0])
        self.assertGreater(loss_upper_only(th.tensor([[0.5, 0.5]]), th.tensor([[2.0, 2.0]])).item(), 0.0)
        self.assertEqual(loss_upper_only(th.tensor([[-0.5, -0.5]]), th.tensor([[-2.0, -2.0]])).item(), 0.0)

        # Only lower bound
        loss_lower_only = DynamicsAwareLoss(dynamics=dyn, x_min=[-1.0, -1.0])
        self.assertGreater(loss_lower_only(th.tensor([[-0.5, -0.5]]), th.tensor([[-2.0, -2.0]])).item(), 0.0)
        self.assertEqual(loss_lower_only(th.tensor([[0.5, 0.5]]), th.tensor([[2.0, 2.0]])).item(), 0.0)

        # Sequence states (3D) aligned with 2D actions (uses last state)
        seq_states = th.tensor([[[0.0, 0.0], [2.0, 2.0]]])  # shape (1, 2, 2)
        seq_loss = loss_bounds(actions, seq_states)
        self.assertGreater(seq_loss.item(), 0.0)

        # 3D states with 3D actions (reshaped to 2D)
        seq_actions = th.tensor([[[0.5, 0.5], [0.5, 0.5]]])  # shape (1, 2, 2)
        seq_both_loss = loss_bounds(seq_actions, seq_states)
        self.assertGreater(seq_both_loss.item(), 0.0)

        # Rank mismatch raises ValueError
        with self.assertRaisesRegex(ValueError, "rank mismatch"):
            loss_bounds(actions, th.zeros(1, 2, 3, 2))

    def test_baseline_dynamics_aware_loss(self) -> None:
        dyn = _MockDynamics()
        dyn_loss = DynamicsAwareLoss(dyn, x_min=[-1.0, -1.0], x_max=[1.0, 1.0])

        # Base loss without states parameter
        base_scaled = ScaledMSELoss(scale=[1.0, 1.0])
        combined = BaselineDynamicsAwareLoss(
            base_loss=base_scaled,
            dynamics_loss=dyn_loss,
            base_weight=2.0,
            dynamics_weight=0.5,
        )

        states = th.tensor([[2.0, 2.0]])
        pred = th.tensor([[0.5, 0.5]])
        target = th.tensor([[0.0, 0.0]])

        total = combined(pred, target, states)
        parts = combined.last_loss_parts
        self.assertIsNotNone(parts)
        self.assertAlmostEqual(parts.base_weight, 2.0)
        self.assertAlmostEqual(parts.dynamics_weight, 0.5)
        th.testing.assert_close(total, parts.total)

        # Base loss with states parameter
        base_state = StateWeightedMSELoss(x_reference=[0.0, 0.0], x_scale=[1.0, 1.0])
        combined_state = BaselineDynamicsAwareLoss(
            base_loss=base_state,
            dynamics_loss=dyn_loss,
        )
        total_state = combined_state(pred, target, states)
        self.assertGreater(total_state.item(), 0.0)

        # ImitationLearningLossParts properties
        p = ImitationLearningLossParts(
            base_raw=th.tensor(3.0),
            dynamics_raw=th.tensor(4.0),
            base_weight=2.0,
            dynamics_weight=3.0,
        )
        self.assertEqual(p.base.item(), 6.0)
        self.assertEqual(p.dynamics.item(), 12.0)
        self.assertEqual(p.total.item(), 18.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
