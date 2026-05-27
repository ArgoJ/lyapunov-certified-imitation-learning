from __future__ import annotations

import os

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Literal
from collections.abc import Mapping

from ..utils.base_config import (
    ArgumentParserConfig,
    JsonDataclass,
    config_field,
    positive_validator,
    non_negative_validator,
    fraction_validator,
    run_field_validators,
)
from ..utils.constants import *


@dataclass(frozen=True)
class ImitationTrainingConfig(JsonDataclass, ArgumentParserConfig):
    """Configuration for imitation policy training.

    Parameters
    ----------
    dataset_path : str or os.PathLike or None, optional
        Path to the source dataset (HDF5). If ``None``, dataset loading is disabled and must be handled manually by the training script.
    sequence_length : int, optional
        Length of state sequences fed to the policy model. Must be positive.
    stride : int, optional
        Stride between the start indices of consecutive state sequences. Must be positive.
    target_mode : {"last", "all"}, optional
        Whether to predict only the last action in the sequence or all actions.
    val_fraction : float, optional
        Fraction of the dataset to use for validation. Must be in the range [0, 1].
    split_seed : int, optional
        Random seed for dataset splitting.
    split_strategy : {"random", "trajectory"}, optional
        Strategy for splitting the dataset.
    use_references : bool, optional
        Whether to use reference trajectories.
    near_duplicate_radius : float, optional
        Radius for considering near-duplicate states. Must be positive.
    batch_size : int, optional
        Training batch size. Must be positive.
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
    weight_decay : float, optional
        Adam optimizer weight decay coefficient. Must be non-negative.
    dropout : float, optional
        Dropout probability for the policy model. Must be in the range [0, 1].
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
    # Dataset
    dataset_path: str | os.PathLike | None = config_field(
        default=None,
        help="Path to the source dataset (HDF5)."
    )
    sequence_length: int = config_field(
        default=1,
        help="Length of state sequences fed to the policy model.",
        validators=(positive_validator,)
    )
    stride: int = config_field(
        default=1,
        help="Stride between the start indices of consecutive state sequences.",
        validators=(positive_validator,)
    )
    target_mode: Literal["last", "all"] = config_field(
        default="last",
        help="Whether to predict only the last action in the sequence or all actions."
    )
    val_fraction: float = config_field(
        default=0.2,
        help="Fraction of the dataset to use for validation.",
        validators=(fraction_validator,)
    )
    seed: int | None = config_field(
        default=None,
        help="Random seed for dataset splitting."
    )
    split_strategy: Literal["random", "trajectory"] = config_field(
        default="random",
        help="Strategy for splitting the dataset."
    )
    use_references: bool = config_field(
        default=False,
        help="Whether to use reference trajectories."
    )
    near_duplicate_radius: float = config_field(
        default=1e-4,
        help="Radius for considering near-duplicate states.",
        validators=(positive_validator,)
    )

    # Training
    batch_size: int = config_field(
        default=256,
        help="Training batch size.",
        validators=(positive_validator,)
    )
    epochs: int = config_field(
        default=10, 
        help="Number of optimization epochs.",
        validators=(positive_validator,)
    )
    restore_best_model: bool = config_field(
        default=True,
        help="Whether to restore the best checkpoint tracked by early stopping."
    )
    learning_rate: float = config_field(
        default=1e-3,
        help="Adam optimizer learning rate.",
        display_alias="lr",
        validators=(positive_validator,)
    )
    weight_decay: float = config_field(
        default=0.0,
        help="Adam optimizer weight decay coefficient.",
        validators=(non_negative_validator,)
    )
    dropout: float = config_field(
        default=0.0,
        help="Dropout probability for the policy model.",
        validators=(fraction_validator,)
    )
    scheduler_type: Literal["none", "step", "cosine", "plateau"] = config_field(
        default="none",
        help="Learning-rate scheduler variant.",
        display_alias="scheduler",
    )
    scheduler_kwargs: Mapping[str, Any] | None = config_field(
        default=None,
        cli=False
    )
    
    # Others
    tb_log_dir: str | os.PathLike | None = config_field(
        default=None,
        help="TensorBoard logging directory."
    )

    train_dataset_path: str | os.PathLike | None = field(init=False, default=None, metadata={"cli": False})
    val_dataset_path: str | os.PathLike | None = field(init=False, default=None, metadata={"cli": False})
    
    NP_ARRAY_FIELDS = ()
    DEFAULT_FILE_NAME = TRAINING_CONFIG_FILENAME

    def __post_init__(self):
        run_field_validators(self)
        if self.target_mode not in {"last", "all"}:
            raise ValueError(f"Invalid target_mode: {self.target_mode}")
        if self.split_strategy not in {"random", "trajectory"}:
            raise ValueError(f"Invalid split_strategy: {self.split_strategy}")
        if self.scheduler_type not in {"none", "step", "cosine", "plateau"}:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")
        if self.tb_log_dir is not None and not isinstance(self.tb_log_dir, (str, os.PathLike)):
            raise ValueError("tb_log_dir must be a string, os.PathLike, or None.")
        if self.tb_log_dir is not None:
            object.__setattr__(self, "tb_log_dir", Path(self.tb_log_dir).resolve())
        
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