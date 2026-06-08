from __future__ import annotations

import logging
import inspect
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from collections.abc import Sequence

__logger__ = logging.getLogger(__name__)

@dataclass
class ImitationLearningLossParts:
    base_raw: th.Tensor
    dynamics_raw: th.Tensor

    base_weight: float
    dynamics_weight: float

    @property
    def base(self) -> th.Tensor:
        return self.base_raw * self.base_weight

    @property
    def dynamics(self) -> th.Tensor:
        return self.dynamics_raw * self.dynamics_weight

    @property
    def total(self) -> th.Tensor:
        return self.base + self.dynamics

def _as_row_tensor(x: Sequence[float] | th.Tensor, name: str) -> th.Tensor:
    t = th.as_tensor(x, dtype=th.float32).view(1, -1)
    if t.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    return t

def scaled_error(positive: th.Tensor, negative: th.Tensor, scale: th.Tensor) -> th.Tensor:
    return (positive - negative) / scale

def per_sample_mse(pred: th.Tensor, target: th.Tensor, scale: th.Tensor) -> th.Tensor:
    return (scaled_error(pred, target, scale) ** 2).mean(dim=-1, keepdim=True)

def reference_weights(
    value: th.Tensor,
    reference: th.Tensor,
    scale: th.Tensor,
    center_alpha: float,
    min_weight: float
) -> th.Tensor:
    d = scaled_error(value, reference, scale).abs().mean(dim=-1, keepdim=True)
    score = (1.0 / (1.0 + d)).pow(center_alpha)
    return min_weight + (1.0 - min_weight) * score


class ScaledMSELoss(nn.Module):
    """Mean squared error loss with per-dimension scaling."""

    def __init__(
        self,
        scale: Sequence[float] | th.Tensor,
    ) -> None:
        """
        Parameters
        ----------
        scale : Sequence[float] | th.Tensor
            Scale values for normalizing actions.
        """
        super().__init__()
        scale_tensor = th.as_tensor(scale, dtype=th.float32).view(1, -1)
        if not th.all(scale_tensor > 0):
            raise ValueError("All scale entries must be positive.")
        self.register_buffer("scale", scale_tensor)
        __logger__.info("Initializing ScaledMSELoss with scale=%s", scale_tensor)

    def forward(self, pred: th.Tensor, target: th.Tensor) -> th.Tensor:
        return per_sample_mse(pred, target, self.scale).mean()


class ActionWeightedMSELoss(nn.Module):
    """Action MSE with per-sample weights based on distance to an action reference."""

    def __init__(
        self,
        action_scale: Sequence[float] | th.Tensor,
        action_reference: Sequence[float] | th.Tensor,
        center_alpha: float = 1.0,
        min_weight: float = 0.25,
    ) -> None:
        super().__init__()

        action_scale = _as_row_tensor(action_scale, "action_scale")
        action_reference = _as_row_tensor(action_reference, "action_reference")

        if not th.all(action_scale > 0):
            raise ValueError("All action_scale entries must be positive.")
        if center_alpha <= 0:
            raise ValueError("center_alpha must be positive.")
        if not (0 < min_weight <= 1):
            raise ValueError("min_weight must be in (0, 1].")
        if action_scale.shape != action_reference.shape:
            raise ValueError("action_scale and action_reference must have the same shape.")

        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_reference", action_reference)
        self.center_alpha = float(center_alpha)
        self.min_weight = float(min_weight)
        __logger__.info("Initializing ActionWeightedMSELoss with reference=%s, scale=%s", 
                        action_reference, action_scale)

    def sample_weights(self, target: th.Tensor) -> th.Tensor:
        return reference_weights(
            value=target,
            reference=self.action_reference,
            scale=self.action_scale,
            center_alpha=self.center_alpha,
            min_weight=self.min_weight,
        )

    def forward(
        self,
        pred: th.Tensor,
        target: th.Tensor,
    ) -> th.Tensor:
        per_sample = per_sample_mse(pred, target, self.action_scale)
        weights = self.sample_weights(target)
        return (weights * per_sample).mean()


class StateWeightedMSELoss(nn.Module):
    """State MSE with per-sample weights based on distance to a state reference."""

    def __init__(
        self,
        x_reference: Sequence[float] | th.Tensor,
        x_scale: Sequence[float] | th.Tensor,
        action_scale: Sequence[float] | th.Tensor | None = None,
        center_alpha: float = 1.0,
        min_weight: float = 0.25,
    ) -> None:
        super().__init__()

        x_reference = _as_row_tensor(x_reference, "x_reference")
        x_scale = _as_row_tensor(x_scale, "x_scale")

        if not th.all(x_scale > 0):
            raise ValueError("All x_scale entries must be positive.")
        if center_alpha <= 0:
            raise ValueError("center_alpha must be positive.")
        if not (0 < min_weight <= 1):
            raise ValueError("min_weight must be in (0, 1].")
        if x_reference.shape != x_scale.shape:
            raise ValueError("x_reference and x_scale must have the same shape.")

        self.register_buffer("x_reference", x_reference)
        self.register_buffer("x_scale", x_scale)
        self.center_alpha = float(center_alpha)
        self.min_weight = float(min_weight)

        if action_scale is None:
            action_scale_tensor = th.ones(1, 1, dtype=th.float32)
            self._scalar_action_scale = True
        else:
            action_scale_tensor = _as_row_tensor(action_scale, "action_scale")
            if not th.all(action_scale_tensor > 0):
                raise ValueError("All action_scale entries must be positive.")
            self._scalar_action_scale = False

        self.register_buffer("action_scale", action_scale_tensor)
        __logger__.info("Initializing StateWeightedMSELoss with x_reference=%s, x_scale=%s, a_scale=%s", 
                        x_reference, x_scale, action_scale_tensor)

    def sample_weights(self, states: th.Tensor) -> th.Tensor:
        return reference_weights(
            value=states,
            reference=self.x_reference,
            scale=self.x_scale,
            center_alpha=self.center_alpha,
            min_weight=self.min_weight,
        )

    def forward(
        self,
        pred: th.Tensor,
        target: th.Tensor,
        states: th.Tensor | None = None,
    ) -> th.Tensor:
        if states is None:
            raise ValueError("StateWeightedMSELoss requires states.")

        action_scale = self.action_scale
        if self._scalar_action_scale and pred.shape[-1] != 1:
            action_scale = th.ones(1, pred.shape[-1], device=pred.device, dtype=pred.dtype)

        per_sample = per_sample_mse(pred, target, action_scale)
        weights = self.sample_weights(states)
        return (weights * per_sample).mean()


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
            return states.new_zeros(())

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
        constraint_loss = states.new_zeros(())

        if self._has_upper_bound:
            constraint_loss += F.relu(next_states_pred - self.x_max).mean()
        if self._has_lower_bound:
            constraint_loss += F.relu(self.x_min - next_states_pred).mean()

        return constraint_loss



class BaselineDynamicsAwareLoss(nn.Module):
    """Combined loss that adds a dynamics-aware penalty to the reference-weighted MSE loss."""
    
    def __init__(
        self, 
        base_loss: nn.Module, 
        dynamics_loss: DynamicsAwareLoss, 
        base_weight: float = 1.0,
        dynamics_weight: float = 1.0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.dynamics_loss = dynamics_loss
        self.base_weight = base_weight
        self.dynamics_weight = dynamics_weight
        self.last_loss_parts: ImitationLearningLossParts | None = None

        loss_signature = inspect.signature(self.base_loss.forward)
        self._base_requires_states = "states" in loss_signature.parameters

    def compute_loss_parts(self, pred_actions: th.Tensor, target_actions: th.Tensor, states: th.Tensor) -> ImitationLearningLossParts:
        if self._base_requires_states:
            scaled_loss_raw = self.base_loss(pred_actions, target_actions, states)
        else:
            scaled_loss_raw = self.base_loss(pred_actions, target_actions) 

        dyn_loss_raw = (self.dynamics_loss(pred_actions, states)
                        if self.dynamics_loss is not None else pred_actions.new_zeros(()))

        parts = ImitationLearningLossParts(
            base_raw=scaled_loss_raw,
            dynamics_raw=dyn_loss_raw,
            base_weight=self.base_weight,
            dynamics_weight=self.dynamics_weight,
        )
        self.last_loss_parts = parts
        return parts

    def forward(self, pred_actions: th.Tensor, target_actions: th.Tensor, states: th.Tensor) -> th.Tensor:
        self.last_loss_parts = self.compute_loss_parts(pred_actions, target_actions, states)
        return self.last_loss_parts.total