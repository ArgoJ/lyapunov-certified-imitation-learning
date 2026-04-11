from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any, Literal
from collections.abc import Mapping

from ..utils.config_io import JsonConfigMixin


@dataclass(frozen=True)
class ImitationTrainingConfig(JsonConfigMixin):
    """Configuration for imitation policy training.

    Parameters
    ----------
    epochs : int, optional
        Number of optimization epochs. Must be positive.
    restore_best_model : bool, optional
        If ``True``, restore the best checkpoint tracked by early stopping
        after training.
    tb_log_dir : str or os.PathLike[str] or None, optional
        TensorBoard logging directory. If ``None``, TensorBoard logging is
        disabled.
    learning_rate : float, optional
        Adam optimizer learning rate. Must be positive.
    scheduler_type : {"none", "step", "cosine", "plateau"}, optional
        Learning-rate scheduler variant.
    scheduler_kwargs : Mapping[str, Any] or None, optional
        Optional scheduler parameters.

        - ``"step"``: ``{"step_size": int, "gamma": float}``
        - ``"cosine"``: ``{"eta_min": float}``
        - ``"plateau"``: ``{"mode": str, "factor": float, "patience": int, "min_lr": float}``

    Raises
    ------
    ValueError
        If ``epochs <= 0``, ``learning_rate <= 0``, or ``scheduler_type`` is
        not one of ``{"none", "step", "cosine", "plateau"}``.
    """

    epochs: int = 10
    restore_best_model: bool = True
    tb_log_dir: str | os.PathLike[str] | None = None
    learning_rate: float = 1e-3
    scheduler_type: Literal["none", "step", "cosine", "plateau"] = "none"
    scheduler_kwargs: Mapping[str, Any] | None = None
    NP_ARRAY_FIELDS = ()

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.scheduler_type not in {"none", "step", "cosine", "plateau"}:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")