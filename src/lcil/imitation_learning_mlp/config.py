from __future__ import annotations

import os

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Literal
from collections.abc import Mapping

from ..utils.base_config import ArgumentParserConfig, JsonDataclass, config_field


@dataclass(frozen=True)
class ImitationTrainingConfig(JsonDataclass, ArgumentParserConfig):
    """Configuration for imitation policy training.

    Parameters
    ----------
    epochs : int, optional
        Number of optimization epochs. Must be positive.
    restore_best_model : bool, optional
        If ``True``, restore the best checkpoint tracked by early stopping
        after training.
    tb_log_dir : str or os.PathLike or None, optional
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

    epochs: int = config_field(default=10, help="Number of optimization epochs.")
    restore_best_model: bool = config_field(default=True, help="Whether to restore the best checkpoint tracked by early stopping.")
    tb_log_dir: str | os.PathLike | None = config_field(default=None, help="TensorBoard logging directory.")
    learning_rate: float = config_field(default=1e-3, help="Adam optimizer learning rate.")
    scheduler_type: Literal["none", "step", "cosine", "plateau"] = config_field(default="none", help="Learning-rate scheduler variant.")
    scheduler_kwargs: Mapping[str, Any] | None = config_field(default=None, cli=False)

    train_dataset_path: str | os.PathLike | None = field(init=False, default=None, metadata={"cli": False})
    val_dataset_path: str | os.PathLike | None = field(init=False, default=None, metadata={"cli": False})
    
    NP_ARRAY_FIELDS = ()

    def __post_init__(self):
        if self.tb_log_dir is not None and not isinstance(self.tb_log_dir, (str, os.PathLike)):
            raise ValueError("tb_log_dir must be a string, os.PathLike, or None.")
        if self.tb_log_dir is not None:
            object.__setattr__(self, "tb_log_dir", Path(self.tb_log_dir).resolve())
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.scheduler_type not in {"none", "step", "cosine", "plateau"}:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")
        
    def register_datasets(self, train_path: str | os.PathLike, val_path: str | os.PathLike) -> None:
        """Register training and validation dataset paths in the config for checkpoint serialization.

        Parameters
        ----------
        train_path : str | os.PathLike
            training dataset path
        val_path : str | os.PathLike
            validation dataset path
        """
        object.__setattr__(self, "train_dataset_path", Path(train_path).resolve())
        object.__setattr__(self, "val_dataset_path", Path(val_path).resolve())