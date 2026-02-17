from __future__ import annotations

from collections.abc import Sequence

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ReferenceWeightedMSELoss(nn.Module):
    """Weighted MSE that emphasizes targets close to a reference value.

    The per-output weights are computed from the absolute distance between
    target actions and a reference action:

    ``w = 1 / (|target - reference| + eps)^alpha``

    Weights are normalized to mean 1 per batch for stable loss scale.
    """

    def __init__(
        self,
        reference: Sequence[float] | th.Tensor,
        alpha: float = 1.0,
        eps: float = 1e-3,
        max_weight: float | None = None,
    ) -> None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        if max_weight is not None and max_weight <= 0.0:
            raise ValueError("max_weight must be positive when provided.")

        ref_tensor = th.as_tensor(reference, dtype=th.float32)
        if ref_tensor.ndim == 0:
            ref_tensor = ref_tensor.unsqueeze(0)

        self.register_buffer("reference", ref_tensor)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.max_weight = max_weight

    def forward(self, pred_actions: th.Tensor, target_actions: th.Tensor) -> th.Tensor:
        if pred_actions.shape != target_actions.shape:
            raise ValueError(
                "pred_actions and target_actions must have identical shape, "
                f"got {tuple(pred_actions.shape)} vs {tuple(target_actions.shape)}."
            )

        if self.reference.numel() != pred_actions.shape[-1]:
            raise ValueError(
                "reference dimension must match action dimension, "
                f"got reference dim {self.reference.numel()} and action dim {pred_actions.shape[-1]}."
            )

        centered_abs = (target_actions - self.reference).abs()
        weights = 1.0 / (centered_abs + self.eps).pow(self.alpha)
        if self.max_weight is not None:
            weights = th.clamp(weights, max=self.max_weight)

        weights = weights / weights.mean().clamp_min(1e-12)
        return F.mse_loss(pred_actions, target_actions, reduction="mean", weight=weights)