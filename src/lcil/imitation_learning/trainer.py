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
    """Epoch-averaged total loss and optional constituent parts."""

    total: float
    base_raw: float = np.nan
    dynamics_raw: float = np.nan
    base: float = np.nan
    dynamics: float = np.nan


@dataclass
class _PolicyEpochLossAccumulator:
    """Accumulate batch losses into epoch-averaged summaries."""

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
        num_datapoints = max(self.num_datapoints, 1)
        total = self.total_sum / num_datapoints

        if self.tracked_datapoints == 0:
            return PolicyEpochLossSummary(total=total)

        tracked_datapoints = self.tracked_datapoints
        return PolicyEpochLossSummary(
            total=total,
            base_raw=self.base_raw_sum / tracked_datapoints,
            dynamics_raw=self.dynamics_raw_sum / tracked_datapoints,
            base=self.base_sum / tracked_datapoints,
            dynamics=self.dynamics_sum / tracked_datapoints,
        )


@dataclass
class PolicyTrainingMetrics:
    """Per-epoch training metrics for policy optimization."""

    train_loss: NDArray
    val_loss: NDArray
    train_base_raw: NDArray
    train_dynamics_raw: NDArray
    val_base_raw: NDArray
    val_dynamics_raw: NDArray
    train_base: NDArray
    train_dynamics: NDArray
    val_base: NDArray
    val_dynamics: NDArray
    learning_rate: NDArray
    epochs_completed: int = 0

    @classmethod
    def from_num_epochs(cls, num_epochs: int) -> PolicyTrainingMetrics:
        """Create NaN-initialized metric arrays for a fixed epoch budget.

        Parameters
        ----------
        num_epochs : int
            Number of training epochs to allocate storage for.

        Returns
        -------
        PolicyTrainingMetrics
            Metrics container with ``float64`` arrays of shape ``(num_epochs,)``
            initialized to ``NaN``.
        """
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")

        nan_array = np.full((num_epochs,), np.nan, dtype=np.float64)
        return cls(
            train_loss=nan_array.copy(),
            val_loss=nan_array.copy(),
            train_base_raw=nan_array.copy(),
            train_dynamics_raw=nan_array.copy(),
            val_base_raw=nan_array.copy(),
            val_dynamics_raw=nan_array.copy(),
            train_base=nan_array.copy(),
            train_dynamics=nan_array.copy(),
            val_base=nan_array.copy(),
            val_dynamics=nan_array.copy(),
            learning_rate=nan_array.copy(),
            epochs_completed=0,
        )

    def update(
        self,
        epoch: int,
        train_summary: PolicyEpochLossSummary,
        val_summary: PolicyEpochLossSummary | None,
        learning_rate: float,
    ) -> None:
        """Update metric arrays with new values for a completed epoch.

        Parameters
        ----------
        epoch : int
            Index of the completed epoch (0-based).
        train_summary : PolicyEpochLossSummary
            Average training loss summary for the completed epoch.
        val_summary : PolicyEpochLossSummary | None
            Average validation loss summary for the completed epoch, or ``None`` if not applicable.
        learning_rate : float
            Learning rate used during the completed epoch.
        """
        if not (0 <= epoch < len(self.train_loss)):
            raise IndexError(f"Epoch index {epoch} is out of bounds for metric arrays of length {len(self.train_loss)}.")

        self.train_loss[epoch] = none_to_float(train_summary.total)
        self.train_base_raw[epoch] = none_to_float(train_summary.base_raw)
        self.train_dynamics_raw[epoch] = none_to_float(train_summary.dynamics_raw)
        self.train_base[epoch] = none_to_float(train_summary.base)
        self.train_dynamics[epoch] = none_to_float(train_summary.dynamics)

        val_total = None if val_summary is None else val_summary.total
        val_base_raw = None if val_summary is None else val_summary.base_raw
        val_dynamics_raw = None if val_summary is None else val_summary.dynamics_raw
        val_base = None if val_summary is None else val_summary.base
        val_dynamics = None if val_summary is None else val_summary.dynamics

        self.learning_rate[epoch] = none_to_float(learning_rate)
        self.val_loss[epoch] = none_to_float(val_total)
        self.val_base_raw[epoch] = none_to_float(val_base_raw)
        self.val_dynamics_raw[epoch] = none_to_float(val_dynamics_raw)
        self.val_base[epoch] = none_to_float(val_base)
        self.val_dynamics[epoch] = none_to_float(val_dynamics)
        self.epochs_completed = max(self.epochs_completed, epoch + 1)

    def save(self, path: os.PathLike) -> None:
        metrics_path = Path(path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            metrics_path,
            train_loss=self.train_loss,
            val_loss=self.val_loss,
            train_base_raw=self.train_base_raw,
            train_dynamics_raw=self.train_dynamics_raw,
            val_base_raw=self.val_base_raw,
            val_dynamics_raw=self.val_dynamics_raw,
            train_base=self.train_base,
            train_dynamics=self.train_dynamics,
            val_base=self.val_base,
            val_dynamics=self.val_dynamics,
            learning_rate=self.learning_rate,
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


def _tb_writer_add_metrics(tb_writer: SummaryWriter | None, metrics: PolicyTrainingMetrics) -> None:
    if tb_writer is None:
        return

    epoch = metrics.epochs_completed - 1
    _tb_writer_add_scalar_if_finite(tb_writer, "Loss/Train", metrics.train_loss[epoch], epoch)
    _tb_writer_add_scalar_if_finite(tb_writer, "Loss/Validation", metrics.val_loss[epoch], epoch)
    _tb_writer_add_scalar_if_finite(tb_writer, "RawLoss/TrainBase", metrics.train_base_raw[epoch], epoch)
    _tb_writer_add_scalar_if_finite(tb_writer, "RawLoss/TrainDynamics", metrics.train_dynamics_raw[epoch], epoch)
    _tb_writer_add_scalar_if_finite(tb_writer, "RawLoss/ValidationBase", metrics.val_base_raw[epoch], epoch)
    _tb_writer_add_scalar_if_finite(tb_writer, "RawLoss/ValidationDynamics", metrics.val_dynamics_raw[epoch], epoch)
    _tb_writer_add_scalar_if_finite(tb_writer, "WeightedLoss/TrainBase", metrics.train_base[epoch], epoch)
    _tb_writer_add_scalar_if_finite(tb_writer, "WeightedLoss/TrainDynamics", metrics.train_dynamics[epoch], epoch)
    _tb_writer_add_scalar_if_finite(tb_writer, "WeightedLoss/ValidationBase", metrics.val_base[epoch], epoch)
    _tb_writer_add_scalar_if_finite(tb_writer, "WeightedLoss/ValidationDynamics", metrics.val_dynamics[epoch], epoch)
    _tb_writer_add_scalar_if_finite(tb_writer, "LearningRate", metrics.learning_rate[epoch], epoch)

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
        self.metrics: PolicyTrainingMetrics | None = None

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
    
    def train(
        self,
    ) -> PolicyTrainingMetrics:
        """
        Train a simple MLP policy model on the provided imitation-learning dataset.

        Returns
        -------
        PolicyTrainingMetrics
            Container with per-epoch training and validation loss, learning rates, and total epochs completed.
        """
        epochs = int(self.training_config.epochs)
        
        self.loss_fn.to(self.device)
        self.model.to(self.device)

        tb_writer = _tb_writer_build(log_dir=self.training_config.tb_log_dir)
        metrics = PolicyTrainingMetrics.from_num_epochs(epochs)
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
                    metrics.update(epoch, train_summary, val_summary, current_lr)
                    monitored_metric = val_summary.total if val_summary is not None else train_summary.total
                    self._scheduller_step(monitored_metric)
                    _tb_writer_add_metrics(tb_writer, metrics)

                    # Early stopping
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
                
        self.metrics = metrics
        _tb_writer_close(tb_writer)
        
        if (
            self.early_stopper is not None
            and self.training_config.restore_best_model
            and self.early_stopper.best_model_state is not None
        ):
            self.early_stopper.load_best_model(self.model)

        return metrics

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
        self.metrics.save(metrics_path)
        
        __logger__.info(f"Saved training results to {metrics_path.parent}")
