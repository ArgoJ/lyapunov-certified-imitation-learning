from __future__ import annotations

from collections.abc import Sequence

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ReferenceWeightedMSELoss(nn.Module):
    """Weighted MSE that emphasizes targets close to a reference value.

    The per-output weights are computed from the absolute distance between
    target actions and a reference action:

    ``w = |target - reference| + eps)^alpha``

    Weights are normalized to mean 1 per batch for stable loss scale.
    """

    def __init__(
        self,
        reference: Sequence[float] | th.Tensor,
        alpha: float = 1.0,
        min_weight: float = 1e-3,
        max_weight: float | None = None,
    ) -> None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if min_weight <= 0.0:
            raise ValueError("min_weight must be positive.")
        if max_weight is not None and max_weight <= 0.0:
            raise ValueError("max_weight must be positive when provided.")

        ref_tensor = th.as_tensor(reference, dtype=th.float32).view(-1)

        self.register_buffer("reference", ref_tensor)
        self.alpha = float(alpha)
        self.min_weight = float(min_weight)
        self.max_weight = max_weight

    def reference_weights(self, target_actions: th.Tensor) -> th.Tensor:
        centered_abs = (target_actions - self.reference).abs()
        weights = centered_abs.pow(self.alpha) + self.min_weight
        if self.max_weight is not None:
            weights = th.clamp(weights, max=self.max_weight)

        weights_mean = weights.mean(dim=0, keepdim=True)
        normalized_weights = weights / (weights_mean + 1e-8)
        return normalized_weights

    def forward(self, pred_actions: th.Tensor, target_actions: th.Tensor) -> th.Tensor:
        ref_weights = self.reference_weights(target_actions)
        return F.mse_loss(pred_actions, target_actions, reduction="mean", weight=ref_weights)



class DynamicsAwareLoss(nn.Module):
    """Loss that penalizes predicted actions that lead to next states outside given bounds."""
    
    def __init__(
        self, 
        dynamics: th.nn.Module, 
        x_min: Sequence[float] | th.Tensor | None = None, 
        x_max: Sequence[float] | th.Tensor | None = None, 
    ):
        super().__init__()
        self.dynamics = dynamics

        self._has_lower_bound = x_min is not None and len(x_min) > 0
        self._has_upper_bound = x_max is not None and len(x_max) > 0

        if self._has_lower_bound:
            self.register_buffer("x_min", th.as_tensor(x_min, dtype=th.float32).view(1, -1))
        if self._has_upper_bound:
            self.register_buffer("x_max", th.as_tensor(x_max, dtype=th.float32).view(1, -1))

    def forward(self, pred_actions: th.Tensor, states: th.Tensor) -> th.Tensor:
        if not self._has_lower_bound and not self._has_upper_bound:
            return th.tensor(0.0, device=states.device)

        next_states_pred = self.dynamics(states, pred_actions)
        constraint_loss = th.tensor(0.0, device=states.device)

        if self._has_upper_bound:
            constraint_loss += F.relu(next_states_pred - self.x_max).mean()
        if self._has_lower_bound:
            constraint_loss += F.relu(self.x_min - next_states_pred).mean()

        return constraint_loss



class ReferenceWeightedDynamicsAwareLoss(nn.Module):
    """Combined loss that adds a dynamics-aware penalty to the reference-weighted MSE loss."""
    
    def __init__(
        self, 
        reference_loss: ReferenceWeightedMSELoss, 
        dynamics_loss: DynamicsAwareLoss, 
        lambda_dyn: float = 1.0,
    ):
        super().__init__()
        self.reference_loss = reference_loss
        self.dynamics_loss = dynamics_loss
        self.lambda_dyn = lambda_dyn

    def forward(self, pred_actions: th.Tensor, target_actions: th.Tensor, states: th.Tensor) -> th.Tensor:
        ref_loss = self.reference_loss(pred_actions, target_actions)
        dyn_loss = self.dynamics_loss(pred_actions, states)
        return ref_loss + self.lambda_dyn * dyn_loss