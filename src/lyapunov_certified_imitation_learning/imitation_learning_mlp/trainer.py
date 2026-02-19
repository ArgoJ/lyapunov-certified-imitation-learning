from __future__ import annotations


import os
import numpy as np
import torch as th
import torch.nn as nn

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from torch.utils.data import DataLoader

from .dataset import save_state_action_dataset_subset
from ..utils.early_stopping import EarlyStopping
from ..utils.package_logger import get_package_logger

__logger__ = get_package_logger(__name__)


@dataclass
class PolicyTrainingMetrics:
    """Per-epoch training metrics for policy optimization."""

    train_loss: np.ndarray
    val_loss: np.ndarray
    learning_rate: np.ndarray
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
            learning_rate=nan_array.copy(),
            epochs_completed=0,
        )

    def save(self, path: os.PathLike[str]) -> None:
        metrics_path = Path(path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            metrics_path,
            train_loss=self.train_loss,
            val_loss=self.val_loss,
            learning_rate=self.learning_rate,
            epochs_completed=np.asarray(self.epochs_completed, dtype=np.int64),
        )


def train_mlp_policy(
    policy_model: nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    loss_fn: nn.Module | None = None,
    scheduler: Literal["none", "step", "cosine", "plateau"] = "none",
    early_stopper: EarlyStopping | None = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    restore_best_model: bool = True,
    device: th.device | str = "cpu",
    save_folder: os.PathLike[str] | None = None,
) -> PolicyTrainingMetrics:
    """
    Train a simple MLP policy model on the provided imitation-learning dataset.
    
    Parameters
    ----------
    policy_model : nn.Module
        The MLP policy model to be trained. Should take state tensors as input and output action tensors.
    dataloader : torch.utils.data.DataLoader
        The imitation learning dataloader providing ``(state, action)`` pairs for training.
    val_dataloader : torch.utils.data.DataLoader or None, optional
        Optional validation dataloader providing ``(state, action)`` pairs.
        If provided, validation loss is computed each epoch and used for
        early stopping / ``"plateau"`` scheduler monitoring.
        If ``None``, training loss is used as the monitored metric.
    scheduler : {"none", "step", "cosine", "plateau"}, optional
        Learning-rate scheduler specifier. ``"step"`` uses a step decay,
        ``"cosine"`` uses cosine annealing, and ``"plateau"`` uses
        validation/train metric plateau reduction.
    early_stopper : EarlyStopping or None, optional
        Optional externally configured early-stopping utility.
        If provided, it is used directly and internal
        ``early_stopping_*`` arguments are ignored.
    num_epochs : int, optional
        Number of training epochs. Default is 10.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 1e-3.
    restore_best_model : bool, optional
        If ``True`` and early stopping is enabled, restore best weights before returning.
    device : torch.device or str, optional
        Device to run training on (e.g., "cpu" or "cuda"). Default is "cpu".
    save_folder : PathLike[str], optional
        If provided, save the trained model state dict and training metrics to this folder after training.

    Returns
    -------
    PolicyTrainingMetrics
        Per-epoch training metrics. Arrays are allocated to ``num_epochs`` and
        unexecuted epochs remain ``NaN`` (e.g., when early stopping triggers).
    """
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    device = th.device(device)
    policy_model.to(device)
    policy_model.train()

    optimizer = th.optim.Adam(policy_model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
    loss_fn = loss_fn.to(device)

    if scheduler == "none":
        scheduler_obj: th.optim.lr_scheduler.LRScheduler | th.optim.lr_scheduler.ReduceLROnPlateau | None = None
    elif scheduler == "step":
        scheduler_obj = th.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler == "cosine":
        scheduler_obj = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1), eta_min=0.0)
    elif scheduler == "plateau":
        scheduler_obj = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=0.0,
        )
    else:
        raise ValueError(
            f"Unsupported scheduler '{scheduler}'. Expected one of: 'none', 'step', 'cosine', 'plateau'."
        )

    train_datapoints = max(len(dataloader.dataset), 1)
    use_validation = val_dataloader is not None
    val_datapoints = max(len(val_dataloader.dataset), 1) if use_validation else 0
    monitored_name = "Val" if use_validation else "Train"
    metrics = PolicyTrainingMetrics.from_num_epochs(num_epochs)
    bar_step = 1.0 / (train_datapoints + val_datapoints)

    with __logger__.tqdm(
        total=float(num_epochs),
        desc="Training MLP Policy",
        unit="epoch",
        bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as pbar:
        for epoch in range(num_epochs):
            train_epoch_loss = 0.0
            policy_model.train()

            # Training loop
            for states, actions in dataloader:
                states = states.to(device=device, non_blocking=True)
                actions = actions.to(device=device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred_actions = policy_model(states)
                loss = loss_fn(pred_actions, actions)
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item() * states.size(0)
                pbar.update(bar_step * states.size(0))

            train_avg_loss = train_epoch_loss / train_datapoints

            # Validation loop
            val_avg_loss: float | None = None
            if use_validation and val_dataloader is not None:
                policy_model.eval()
                val_epoch_loss = 0.0
                with th.no_grad():
                    for states, actions in val_dataloader:
                        states = states.to(device=device, non_blocking=True)
                        actions = actions.to(device=device, non_blocking=True)
                        pred_actions = policy_model(states)
                        val_epoch_loss += loss_fn(pred_actions, actions).item() * states.size(0)
                        pbar.update(bar_step * states.size(0))
                val_avg_loss = val_epoch_loss / val_datapoints

            # Update metrics
            metrics.train_loss[epoch] = float(train_avg_loss)
            metrics.val_loss[epoch] = float(val_avg_loss) if val_avg_loss is not None else np.nan
            metrics.learning_rate[epoch] = float(optimizer.param_groups[0]["lr"])
            metrics.epochs_completed = epoch + 1

            monitored_metric = val_avg_loss if val_avg_loss is not None else train_avg_loss

            # Scheduler step
            if scheduler_obj is not None:
                if isinstance(scheduler_obj, th.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_obj.step(monitored_metric)
                else:
                    scheduler_obj.step()

            # Early stopping
            if early_stopper is not None:
                early_stopper(monitored_metric, policy_model)
                if early_stopper.early_stop:
                    __logger__.info(
                        "Early stopping at epoch %d (%s loss %.6f).",
                        epoch + 1,
                        monitored_name,
                        monitored_metric,
                    )
                    pbar.update(1.0)
                    break

            # Postfix for tqdm bar
            postfix = {
                "Train": f"{train_avg_loss:.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            if val_avg_loss is not None:
                postfix["Val"] = f"{val_avg_loss:.4f}"
            pbar.set_postfix(postfix)
            

    if early_stopper is not None and restore_best_model and early_stopper.best_model_state is not None:
        early_stopper.load_best_model(policy_model)

    # Saving
    if save_folder is not None:
        save_folder = Path(save_folder)

        # Dataset saving
        train_dataset_path = save_folder / f"{save_folder.stem}_train_dataset.pt"
        train_saved = save_state_action_dataset_subset(dataloader.dataset, train_dataset_path)

        val_saved = False
        if val_dataloader is not None:
            val_dataset_path = save_folder / f"{save_folder.stem}_val_dataset.pt"
            val_saved = save_state_action_dataset_subset(val_dataloader.dataset, val_dataset_path)
        else:
            val_dataset_path = None

        resolved_train_dataset_path = str(train_dataset_path) if train_saved else None
        resolved_val_dataset_path = str(val_dataset_path) if val_saved and val_dataset_path is not None else None

        # Policy model saving
        policy_path = save_folder / f"{save_folder.stem}_policy.pt"
        if hasattr(policy_model, "save") and callable(policy_model.save):
            try:
                policy_model.save(
                    policy_path,
                    train_dataset_path=resolved_train_dataset_path,
                    val_dataset_path=resolved_val_dataset_path,
                )
            except TypeError:
                policy_model.save(policy_path)
        else:
            save_folder.parent.mkdir(parents=True, exist_ok=True)
            th.save(
                {
                    "state_dict": policy_model.state_dict(),
                    "train_dataset_path": resolved_train_dataset_path,
                    "val_dataset_path": resolved_val_dataset_path,
                },
                policy_path,
            )

        # Metrics
        metrics_path = save_folder / f"{save_folder.stem}_training_metrics.npz"
        metrics.save(metrics_path)
        
        __logger__.info("Saved training results to %s", metrics_path.parent)

    return metrics
