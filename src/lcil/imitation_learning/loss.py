from __future__ import annotations

from collections.abc import Sequence

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from collections.abc import Sequence


@dataclass
class ImitationLearningLossParts:
    reference_raw: th.Tensor
    dynamics_raw: th.Tensor

    reference_weight: float
    dynamics_weight: float

    @property
    def reference(self) -> th.Tensor:
        return self.reference_raw * self.reference_weight

    @property
    def dynamics(self) -> th.Tensor:
        return self.dynamics_raw * self.dynamics_weight

    @property
    def total(self) -> th.Tensor:
        return self.reference + self.dynamics


class BoundedReferenceWeightedMSELoss(nn.Module):
    """
    Weighted MSE that normalizes actions via bounds before calculating the 
    distance to a specific reference value.
    """

    def __init__(
        self,
        reference: Sequence[float] | th.Tensor,
        x_min: Sequence[float] | th.Tensor,
        x_max: Sequence[float] | th.Tensor,
        alpha: float = 1.0,
        min_weight: float = 1e-3,
        max_weight: float | None = None,
        emphasize_close: bool = False, 
    ) -> None:
        """
        Parameters
        ----------
        reference : Sequence[float] | th.Tensor
            Reference action values to compare against.
        x_min : Sequence[float] | th.Tensor
            Minimum action values for normalization.
        x_max : Sequence[float] | th.Tensor
            Maximum action values for normalization.
        alpha : float, optional
            Exponent for weighting, by default 1.0
        min_weight : float, optional
            Minimum weight, by default 1e-3
        max_weight : float | None, optional
            Maximum weight, by default None
        emphasize_close : bool, optional
            If False (default): Emphasizes actions FAR from the reference.
            If True: Emphasizes actions CLOSE to the reference.
        """
        super().__init__()
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if min_weight <= 0.0:
            raise ValueError("min_weight must be positive.")

        ref_tensor = th.as_tensor(reference, dtype=th.float32).view(-1)
        x_min_tensor = th.as_tensor(x_min, dtype=th.float32).view(-1)
        x_max_tensor = th.as_tensor(x_max, dtype=th.float32).view(-1)
        
        assert th.all(x_max_tensor > x_min_tensor), "x_max must be strictly greater than x_min"

        self.register_buffer("reference", ref_tensor)
        self.register_buffer("x_min", x_min_tensor)
        self.register_buffer("x_max", x_max_tensor)
        
        self.emphasize_close = emphasize_close
        self.alpha = float(alpha)
        self.min_weight = float(min_weight)
        self.max_weight = max_weight

    def _compute_weights(self, target_actions: th.Tensor) -> th.Tensor:
        # 1. Normalize targets and reference to a scale of 0.0 to 1.0
        range_tensor = self.x_max - self.x_min

        target_norm = (target_actions - self.x_min) / range_tensor
        target_norm = th.clamp(target_norm, 0.0, 1.0)

        ref_norm = (self.reference - self.x_min) / range_tensor
        ref_norm = th.clamp(ref_norm, 0.0, 1.0)

        # 2. Calculate distance in the normalized space (values between 0.0 and 1.0)
        distance = (target_norm - ref_norm).abs()

        # 3. Invert logic if emphasizing closeness to the reference
        if self.emphasize_close:
            distance = 1.0 - distance

        # 4. calculate weights
        weights = distance.pow(self.alpha) + self.min_weight

        if self.max_weight is not None:
            weights = th.clamp(weights, max=self.max_weight)

        # 5. normalize weights
        weights_mean = weights.mean(dim=0, keepdim=True)
        normalized_weights = weights / (weights_mean + 1e-8)
        
        return normalized_weights

    def forward(self, pred_actions: th.Tensor, target_actions: th.Tensor) -> th.Tensor:
        weights = self._compute_weights(target_actions)
        return F.mse_loss(pred_actions, target_actions, reduction="mean", weight=weights)



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



class ReferenceWeightedDynamicsAwareLoss(nn.Module):
    """Combined loss that adds a dynamics-aware penalty to the reference-weighted MSE loss."""
    
    def __init__(
        self, 
        reference_loss: BoundedReferenceWeightedMSELoss, 
        dynamics_loss: DynamicsAwareLoss, 
        dynamics_weight: float = 1.0,
    ):
        super().__init__()
        self.reference_loss = reference_loss
        self.dynamics_loss = dynamics_loss
        self.dynamics_weight = dynamics_weight
        self.last_loss_parts: ImitationLearningLossParts | None = None

    def compute_loss_parts(self, pred_actions: th.Tensor, target_actions: th.Tensor, states: th.Tensor) -> ImitationLearningLossParts:
        ref_loss_raw = self.reference_loss(pred_actions, target_actions)
        dyn_loss_raw = self.dynamics_loss(pred_actions, states)

        parts = ImitationLearningLossParts(
            reference_raw=ref_loss_raw,
            dynamics_raw=dyn_loss_raw,
            reference_weight=1.0,
            dynamics_weight=self.dynamics_weight,
        )
        self.last_loss_parts = parts
        return parts

    def forward(self, pred_actions: th.Tensor, target_actions: th.Tensor, states: th.Tensor) -> th.Tensor:
        self.last_loss_parts = self.compute_loss_parts(pred_actions, target_actions, states)
        return self.last_loss_parts.total