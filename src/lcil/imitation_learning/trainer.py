from __future__ import annotations

import logging
import os
import inspect
import numpy as np
import torch as th
import torch.nn as nn

from dataclasses import dataclass
from pathlib import Path
from numpy.typing import NDArray
from typing import Any
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
    MofNCompleteColumn,
)

from .config import ImitationTrainingConfig
from .dataset import save_state_action_dataset_subset
from .loss import ImitationLearningLossParts
from ..utils import (
    EarlyStopping,
    GracefulInterruptHandler,
    none_to_float,
)
from ..utils.constants import *

__logger__ = logging.getLogger(__name__)


@dataclass
class PolicyEpochLossSummary:
    total: float
    base_raw: float = np.nan
    dynamics_raw: float = np.nan
    base: float = np.nan
    dynamics: float = np.nan


@dataclass
class _PolicyEpochLossAccumulator:
    total_sum: float = 0.0
    base_raw_sum: float = 0.0
    dynamics_raw_sum: float = 0.0
    base_sum: float = 0.0
    dynamics_sum: float = 0.0
    num_datapoints: int = 0
    tracked_datapoints: int = 0

    def update(
        self,
        batch_size: int,
        loss: th.Tensor,
        loss_parts: ImitationLearningLossParts | None,
    ) -> None:
        self.total_sum += float(loss.item()) * batch_size
        self.num_datapoints += batch_size

        if loss_parts is None:
            return

        self.base_raw_sum += float(loss_parts.base_raw.item()) * batch_size
        self.dynamics_raw_sum += float(loss_parts.dynamics_raw.item()) * batch_size
        self.base_sum += float(loss_parts.base.item()) * batch_size
        self.dynamics_sum += float(loss_parts.dynamics.item()) * batch_size
        self.tracked_datapoints += batch_size

    def finalize(self) -> PolicyEpochLossSummary:
        total = self.total_sum / max(self.num_datapoints, 1)

        if self.tracked_datapoints == 0:
            return PolicyEpochLossSummary(total=total)

        n = self.tracked_datapoints
        return PolicyEpochLossSummary(
            total=total,
            base_raw=self.base_raw_sum / n,
            dynamics_raw=self.dynamics_raw_sum / n,
            base=self.base_sum / n,
            dynamics=self.dynamics_sum / n,
        )


def _nan_array(num_epochs: int) -> NDArray:
    return np.full((num_epochs,), np.nan, dtype=np.float64)


def _write_summary(
    loss_array: NDArray,
    base_raw_array: NDArray,
    dynamics_raw_array: NDArray,
    base_array: NDArray,
    dynamics_array: NDArray,
    epoch: int,
    summary: PolicyEpochLossSummary | None,
) -> None:
    if summary is None:
        loss_array[epoch] = np.nan
        base_raw_array[epoch] = np.nan
        dynamics_raw_array[epoch] = np.nan
        base_array[epoch] = np.nan
        dynamics_array[epoch] = np.nan
        return

    loss_array[epoch] = float(summary.total)
    base_raw_array[epoch] = float(summary.base_raw)
    dynamics_raw_array[epoch] = float(summary.dynamics_raw)
    base_array[epoch] = float(summary.base)
    dynamics_array[epoch] = float(summary.dynamics)


@dataclass
class PolicyEpochMetrics:
    loss: NDArray
    base_raw: NDArray
    dynamics_raw: NDArray
    base: NDArray
    dynamics: NDArray
    epochs_completed: int = 0

    @classmethod
    def from_num_epochs(cls, num_epochs: int) -> "PolicyEpochMetrics":
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")

        return cls(
            loss=_nan_array(num_epochs),
            base_raw=_nan_array(num_epochs),
            dynamics_raw=_nan_array(num_epochs),
            base=_nan_array(num_epochs),
            dynamics=_nan_array(num_epochs),
        )

    def update(
        self,
        epoch: int,
        summary: PolicyEpochLossSummary | None,
    ) -> None:
        if summary is None:
            return

        if not (0 <= epoch < len(self.loss)):
            raise IndexError(f"Epoch index {epoch} is out of bounds.")

        _write_summary(
            self.loss,
            self.base_raw,
            self.dynamics_raw,
            self.base,
            self.dynamics,
            epoch,
            summary,
        )
        self.epochs_completed = max(self.epochs_completed, epoch + 1)

    def save(self, path: os.PathLike) -> None:
        metrics_path = Path(path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            metrics_path,
            loss=self.loss,
            base_raw=self.base_raw,
            dynamics_raw=self.dynamics_raw,
            base=self.base,
            dynamics=self.dynamics,
            epochs_completed=np.asarray(self.epochs_completed, dtype=np.int64),
        )


def _tb_writer_add_scalar_if_finite(
    tb_writer: SummaryWriter,
    tag: str,
    value: float,
    step: int,
) -> None:
    if np.isfinite(value):
        tb_writer.add_scalar(tag, value, step)


def _tb_writer_add_summary(
    tb_writer: SummaryWriter,
    summary: PolicyEpochLossSummary | None,
    prefix: str,
    step: int,
) -> None:
    if summary is None:
        return

    _tb_writer_add_scalar_if_finite(tb_writer, f"RawLoss/{prefix}Base", summary.base_raw, step)
    _tb_writer_add_scalar_if_finite(tb_writer, f"RawLoss/{prefix}Dynamics", summary.dynamics_raw, step)
    _tb_writer_add_scalar_if_finite(tb_writer, f"WeightedLoss/{prefix}Total", summary.total, step)
    _tb_writer_add_scalar_if_finite(tb_writer, f"WeightedLoss/{prefix}Base", summary.base, step)
    _tb_writer_add_scalar_if_finite(tb_writer, f"WeightedLoss/{prefix}Dynamics", summary.dynamics, step)

def _tb_writer_close(tb_writer: SummaryWriter | None) -> None:
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

def _tb_writer_build(log_dir: os.PathLike | None) -> SummaryWriter | None:
    if log_dir is not None:
        return SummaryWriter(log_dir=log_dir)
    return None



class PolicyTrainer:
    """Trainer class for MLP policy imitation learning. Encapsulates training loop, metrics tracking, and checkpoint saving."""

    def __init__(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        training_config: ImitationTrainingConfig,
        val_dataloader: DataLoader | None = None, 
        early_stopper: EarlyStopping | None = None,
        loss_fn: nn.Module | None = None,
        device: th.device | str = "cpu"
    ) -> None:
        """
        
        Parameters
        ----------
        model : nn.Module
            The model to be trained. Should take state tensors as input and output action tensors.
        dataloader : torch.utils.data.DataLoader
            The imitation learning dataloader providing ``(state, action)`` pairs for training.
        training_config : ImitationTrainingConfig
            Immutable training configuration controlling epochs, optimizer learning
            rate, scheduler settings, TensorBoard logging, and whether to restore
            the best early-stopping checkpoint.
        val_dataloader : torch.utils.data.DataLoader, optional
            Optional validation dataloader providing ``(state, action)`` pairs.
            If provided, validation loss is computed each epoch and used for
            early stopping / ``"plateau"`` scheduler monitoring.
            If ``None``, training loss is used as the monitored metric.
        early_stopper : EarlyStopping, optional
            Optional externally configured early-stopping utility.
            If provided, it is used directly and internal
            ``early_stopping_*`` arguments are ignored.
        loss_fn : torch.nn.Module, optional
            Optional loss function module. If ``None``, defaults to mean squared error (MSE
        device : torch.device or str, optional
            Device to run training on (e.g., "cpu" or "cuda"). Default is "cpu".
        """
        self.model : nn.Module = model
        self.dataloader: DataLoader = dataloader
        self.training_config = training_config
        self.val_dataloader: DataLoader | None = val_dataloader
        self.early_stopper: EarlyStopping | None = early_stopper
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        self.device = th.device(device)
        
        loss_signature = inspect.signature(self.loss_fn.forward)
        self._loss_requires_states = "states" in loss_signature.parameters
        self._use_refs = hasattr(dataloader.dataset, "refs") and dataloader.dataset._refs is not None
        
        self.optimizer: th.optim.Optimizer = th.optim.Adam(
            self.model.parameters(),
            lr=float(self.training_config.learning_rate),
            weight_decay=float(self.training_config.weight_decay),
        )
        self.scheduler = self._configure_scheduler()
        self.metrics: PolicyEpochMetrics | None = None

    def _configure_scheduler(self) -> (
            th.optim.lr_scheduler.LRScheduler | 
            th.optim.lr_scheduler.ReduceLROnPlateau | 
            None
        ):
        """Utility to configure the learning rate scheduler based on the training configuration.

        Returns
        -------
        torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau, None
            Configured learning rate scheduler instance, or ``None`` if no scheduler is used.
        """
        st = self.training_config.scheduler_type
        params = dict(self.training_config.scheduler_kwargs) if self.training_config.scheduler_kwargs is not None else {}

        match st:
            case "none":
                return None
            case "step":
                step_size = int(params.get("step_size", 10))
                gamma = float(params.get("gamma", 0.5))
                return th.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            case "cosine":
                t_max = max(int(self.training_config.epochs), 1)
                eta_min = float(params.get("eta_min", 0.0))
                return th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
            case "plateau":
                factor = float(params.get("factor", 0.5))
                patience = int(params.get("patience", 5))
                min_lr = float(params.get("min_lr", 0.0))
                mode = params.get("mode", "min")
                return th.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode=mode,
                    factor=factor,
                    patience=patience,
                    min_lr=min_lr,
                )
            case _:
                __logger__.warning(f"Unsupported scheduler type: {st}. Using no scheduler.")
                return None

    def _extract_batch(self, batch: tuple[th.Tensor, ...]) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Utility to extract model inputs and action targets from a dataloader batch, handling optional references."""
        if self._use_refs:
            states, actions, refs = batch
            states = states.to(device=self.device, non_blocking=True)
            refs = refs.to(device=self.device, non_blocking=True)
            actions = actions.to(device=self.device, non_blocking=True)
            nn_inputs = states - refs 
        else:
            states, actions = batch
            states = states.to(device=self.device, non_blocking=True)
            actions = actions.to(device=self.device, non_blocking=True)
            nn_inputs = states
        return nn_inputs, states, actions

    def _get_bar_step(self) -> float:
        """Utility to compute progress bar step size based on dataloader sizes."""
        train_datapoints = max(len(self.dataloader.dataset), 1)
        val_datapoints = max(len(self.val_dataloader.dataset), 1) if self.val_dataloader is not None else 0
        total_datapoints = train_datapoints + val_datapoints
        return 1.0 / total_datapoints if total_datapoints > 0 else 1.0

    def _scheduller_step(self, monitored_metric: float) -> None:
        """Utility to step the learning rate scheduler based on the monitored metric."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, th.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(monitored_metric)
            else:
                self.scheduler.step()

    def _early_stopping_break(self, monitored_metric: float) -> bool:
        """Utility to check early stopping conditions based on the monitored metric."""
        if self.early_stopper is not None:
            self.early_stopper(monitored_metric, self.model)
            return self.early_stopper.early_stop
        return False

    def _predict_actions(self, nn_inputs: th.Tensor) -> th.Tensor:
        """Use unclamped policy outputs when the model exposes them."""
        raw_forward = getattr(self.model, "forward_raw", None)
        if callable(raw_forward):
            return raw_forward(nn_inputs)
        return self.model(nn_inputs)

    def _get_last_loss_parts(self) -> ImitationLearningLossParts | None:
        loss_parts = getattr(self.loss_fn, "last_loss_parts", None)
        if isinstance(loss_parts, ImitationLearningLossParts):
            return loss_parts
        return None

    def _train_epoch(self, progress: Progress, task: TaskID, bar_step: float) -> PolicyEpochLossSummary:
        """Executes a single training epoch."""
        self.model.train()
        epoch_loss = _PolicyEpochLossAccumulator()

        for batch in self.dataloader:
            nn_inputs, states, actions = self._extract_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)
            pred_actions = self._predict_actions(nn_inputs)
            
            if self._loss_requires_states:
                loss = self.loss_fn(pred_actions, actions, states=states)
            else:
                loss = self.loss_fn(pred_actions, actions)

            loss_parts = self._get_last_loss_parts()

            loss.backward()
            self.optimizer.step()

            batch_size = nn_inputs.size(0)
            epoch_loss.update(batch_size=batch_size, loss=loss, loss_parts=loss_parts)
            progress.update(task, advance=bar_step * batch_size)

        return epoch_loss.finalize()
    
    def _validate_epoch(
        self,
        progress: Progress,
        task: TaskID,
        bar_step: float,
    ) -> PolicyEpochLossSummary | None:
        """Executes a single validation epoch."""
        if self.val_dataloader is None:
            return None

        self.model.eval()
        epoch_loss = _PolicyEpochLossAccumulator()

        with th.no_grad():
            for batch in self.val_dataloader:
                nn_inputs, states, actions = self._extract_batch(batch)
                pred_actions = self._predict_actions(nn_inputs)

                if self._loss_requires_states:
                    loss = self.loss_fn(pred_actions, actions, states=states)
                else:
                    loss = self.loss_fn(pred_actions, actions)

                loss_parts = self._get_last_loss_parts()

                batch_size = nn_inputs.size(0)
                epoch_loss.update(batch_size=batch_size, loss=loss, loss_parts=loss_parts)
                progress.update(task, advance=bar_step * batch_size)

            return epoch_loss.finalize()
    
    def train(self) -> tuple[PolicyEpochMetrics, PolicyEpochMetrics | None]:
        """
        Train a simple MLP policy model on the provided imitation-learning dataset.

        Returns
        -------
        tuple[PolicyEpochMetrics, PolicyEpochMetrics | None]
            Container with per-epoch training and validation loss, learning rates, and total epochs completed.
        """
        epochs = int(self.training_config.epochs)
        
        self.loss_fn.to(self.device)
        self.model.to(self.device)

        tb_writer = _tb_writer_build(log_dir=self.training_config.tb_log_dir)
        train_metrics = PolicyEpochMetrics.from_num_epochs(epochs)
        val_metrics = PolicyEpochMetrics.from_num_epochs(epochs) if self.val_dataloader is not None else None
        bar_step = self._get_bar_step()

        with GracefulInterruptHandler(logger=__logger__) as interrupt_handler:
            with Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("train: {task.fields[train_loss]:.3f}"),
                TextColumn("val: {task.fields[val_loss]:.3f}"),
                TextColumn("lr: {task.fields[lr]:.1e}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    "Train Policy",
                    total=float(epochs),
                    train_loss=float("nan"),
                    val_loss=float("nan"),
                    lr=float("nan"),
                )
                for epoch in range(epochs):
                    train_summary = self._train_epoch(progress, task, bar_step)
                    val_summary = self._validate_epoch(progress, task, bar_step)

                    # Update metrics
                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    train_metrics.update(epoch, train_summary, current_lr)
                    val_metrics.update(epoch, val_summary, current_lr)

                    # Write TensorBoard scalars
                    _tb_writer_add_summary(tb_writer, train_summary, prefix="Train", step=epoch)
                    _tb_writer_add_summary(tb_writer, val_summary, prefix="Validation", step=epoch)

                    # Early stopping and scheduler stepping
                    monitored_metric = train_summary.total if val_summary is None else val_summary.total
                    self._scheduller_step(monitored_metric)
                    if self._early_stopping_break(monitored_metric):
                        __logger__.info(
                            f"Early stopping at epoch {epoch + 1:d} (loss {monitored_metric:.6f}).",
                        )
                        break

                    # Update bar
                    progress.update(
                        task,
                        train_loss=none_to_float(train_summary.total),
                        val_loss=none_to_float(None if val_summary is None else val_summary.total),
                        lr=none_to_float(current_lr),
                    )
        
        if interrupt_handler.aborted:
            progress.stop()
                
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        _tb_writer_close(tb_writer)
        
        if (
            self.early_stopper is not None
            and self.training_config.restore_best_model
            and self.early_stopper.best_model_state is not None
        ):
            self.early_stopper.load_best_model(self.model)

        return train_metrics, val_metrics

    def save(
        self,
        save_folder: os.PathLike,
        global_config: Any = None,
    ) -> None:
        """Utility function to save training results after training completes.

        Parameters
        ----------
        save_folder : PathLike[str]
            Folder where results should be saved. The model checkpoint and training metrics will be saved
            with filenames derived from the folder name.
        global_config : Any, optional
            Optional metadata stored in the model checkpoint config file.
            ``MPCConfig`` objects are serialized via ``to_dict()``.
        """
        save_folder = Path(save_folder).resolve()
        save_folder.mkdir(parents=True, exist_ok=True)

        # Dataset saving
        train_dataset_path = save_folder / "train_dataset.pt"
        train_saved = save_state_action_dataset_subset(self.dataloader.dataset, train_dataset_path)

        val_saved = False
        if self.val_dataloader is not None:
            val_dataset_path = save_folder / "val_dataset.pt"
            val_saved = save_state_action_dataset_subset(self.val_dataloader.dataset, val_dataset_path)
        else:
            val_dataset_path = None
        
        # Config saving
        config_path = save_folder / TRAINING_CONFIG_FILENAME
        self.training_config.register_datasets(
            train_path=train_dataset_path if train_saved else None,
            val_path=val_dataset_path if val_saved else None,
        )
        self.training_config.save(config_path)

        # Policy model saving
        model_path = save_folder / POLICY_MODEL_FILENAME
        self.model.save(
            model_path,
            global_config=global_config,
        )

        # Metrics
        metrics_path = save_folder / TRAINING_METRICS_FILENAME
        self.train_metrics.save(metrics_path)
        if self.val_metrics is not None:
            val_metrics_path = save_folder / "val_" + TRAINING_METRICS_FILENAME
            self.val_metrics.save(val_metrics_path)
        
        __logger__.info(f"Saved training results to {metrics_path.parent}")
