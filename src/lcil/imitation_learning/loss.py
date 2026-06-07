from __future__ import annotations

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from collections.abc import Sequence

from lcil.lyapunov_learning import loss


@dataclass
class ImitationLearningLossParts:
    scaled_raw: th.Tensor
    dynamics_raw: th.Tensor

    scaled_weight: float
    dynamics_weight: float

    @property
    def scaled(self) -> th.Tensor:
        return self.scaled_raw * self.scaled_weight

    @property
    def dynamics(self) -> th.Tensor:
        return self.dynamics_raw * self.dynamics_weight

    @property
    def total(self) -> th.Tensor:
        return self.scaled + self.dynamics


class ScaledMSELoss(nn.Module):
    """Mean squared error loss with per-dimension scaling."""

    def __init__(
        self,
        scale: Sequence[float] | th.Tensor,
        device: th.device | str = "cpu"
    ) -> None:
        """
        Parameters
        ----------
        scale : Sequence[float] | th.Tensor
            Scale values for normalizing actions.
        """
        super().__init__()
        self.action_scale = th.as_tensor(scale, dtype=th.float32, device=device).view(1, -1)

    def forward(self, pred_actions: th.Tensor, target_actions: th.Tensor) -> th.Tensor:
        return (((pred_actions - target_actions) / self.action_scale) ** 2).mean()


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

        aligned_states = states
        if states.ndim == pred_actions.ndim + 1:
            aligned_states = states[:, -1, :]
        elif states.ndim == pred_actions.ndim and states.ndim == 3:
            aligned_states = states.reshape(-1, states.shape[-1])
            pred_actions = pred_actions.reshape(-1, pred_actions.shape[-1])
        elif states.ndim != pred_actions.ndim:
            raise ValueError(
                "states/pred_actions rank mismatch: "
                f"got states ndim {states.ndim} and pred_actions ndim {pred_actions.ndim}."
            )

        next_states_pred = self.dynamics(aligned_states, pred_actions)
        constraint_loss = th.tensor(0.0, device=states.device)

        if self._has_upper_bound:
            constraint_loss += F.relu(next_states_pred - self.x_max).mean()
        if self._has_lower_bound:
            constraint_loss += F.relu(self.x_min - next_states_pred).mean()

        return constraint_loss



class ScaledDynamicsAwareLoss(nn.Module):
    """Combined loss that adds a dynamics-aware penalty to the reference-weighted MSE loss."""
    
    def __init__(
        self, 
        scaled_loss: ScaledMSELoss, 
        dynamics_loss: DynamicsAwareLoss, 
        scaled_weight: float = 1.0,
        dynamics_weight: float = 1.0,
    ):
        super().__init__()
        self.scaled_loss = scaled_loss
        self.dynamics_loss = dynamics_loss
        self.scaled_weight = scaled_weight
        self.dynamics_weight = dynamics_weight
        self.last_loss_parts: ImitationLearningLossParts | None = None

    def compute_loss_parts(self, pred_actions: th.Tensor, target_actions: th.Tensor, states: th.Tensor) -> ImitationLearningLossParts:
        ref_loss_raw = self.scaled_loss(pred_actions, target_actions) 
        dyn_loss_raw = (self.dynamics_loss(pred_actions, states)
                        if self.dynamics_loss is not None else th.tensor(0.0, device=pred_actions.device))

        parts = ImitationLearningLossParts(
            scaled_raw=ref_loss_raw,
            dynamics_raw=dyn_loss_raw,
            scaled_weight=self.scaled_weight,
            dynamics_weight=self.dynamics_weight,
        )
        self.last_loss_parts = parts
        return parts

    def forward(self, pred_actions: th.Tensor, target_actions: th.Tensor, states: th.Tensor) -> th.Tensor:
        self.last_loss_parts = self.compute_loss_parts(pred_actions, target_actions, states)
        return self.last_loss_parts.total