from __future__ import annotations


import os
import inspect
import numpy as np
import torch as th
import torch.nn as nn

from dataclasses import dataclass
from pathlib import Path
from numpy.typing import NDArray
from typing import Any, Literal
from collections.abc import Mapping
from torch.utils.data import DataLoader
from pkg_logger import get_package_logger

from .dataset import save_state_action_dataset_subset
from ..utils.early_stopping import EarlyStopping

__logger__ = get_package_logger(__name__)


@dataclass
class PolicyTrainingMetrics:
    """Per-epoch training metrics for policy optimization."""

    train_loss: NDArray
    val_loss: NDArray
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



class PolicyTrainer:
    """Trainer class for MLP policy imitation learning. Encapsulates training loop, metrics tracking, and checkpoint saving."""

    def __init__(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
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
        self.val_dataloader: DataLoader | None = val_dataloader
        self.early_stopper: EarlyStopping | None = early_stopper
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        self.device = th.device(device)
        
        loss_signature = inspect.signature(self.loss_fn.forward)
        self._loss_requires_states = "states" in loss_signature.parameters
        self._use_refs = hasattr(dataloader.dataset, "refs") and dataloader.dataset._refs is not None
        
        self.optimizer: th.optim.Optimizer | None = None
        self.scheduler: th.optim.lr_scheduler.LRScheduler | th.optim.lr_scheduler.ReduceLROnPlateau | None = None
        self.scheduler_type: Literal["none", "step", "cosine", "plateau"] | None = None
        self.scheduler_params: dict = {}
        self.metrics: PolicyTrainingMetrics | None = None
        
    def set_adam_optimizer(
        self,
        learning_rate: float = 1e-3,
        scheduler_type: Literal["none", "step", "cosine", "plateau"] = "none",
        scheduler_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the trainer to use the Adam optimizer with the specified learning rate and optional LR scheduler.
        
        Parameters
        ----------
        learning_rate : float, optional
            Learning rate for the Adam optimizer. Default is 1e-3.
        scheduler_type : str, optional
            Type of learning rate scheduler to use. One of "none", "step", "cosine", or "plateau". Default is "none" (no scheduler).
        scheduler_kwargs : dict or None, optional
            Optional dictionary of parameters for the specified scheduler. The required keys depend on the scheduler type:
            - For "step": {"step_size": int, "gamma": float}
            - For "cosine": {"eta_min": float}
            - For "plateau": {"factor": float, "patience": int, "min_lr": float, "mode": str}
            If ``None``, default parameters will be used for the chosen scheduler type.
        """
        if self.model is None:
            raise ValueError("Model must be set before configuring optimizer.")
        
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.scheduler_type = scheduler_type
        self.scheduler_params = dict(scheduler_kwargs) if scheduler_kwargs is not None else {}

    def _configure_scheduler(self, num_epochs: int) -> None:
        """Initialize the LR scheduler using the stored scheduler config.

        Some schedulers (e.g., CosineAnnealingLR) require knowledge of the
        total number of epochs in advance, so this is called at the start of training.
        """
        if self.optimizer is None:
            return

        st = self.scheduler_type or "none"
        params = self.scheduler_params or {}

        if st == "none":
            self.scheduler = None
            return

        if st == "step":
            step_size = int(params.get("step_size", 10))
            gamma = float(params.get("gamma", 0.5))
            self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif st == "cosine":
            t_max = max(int(num_epochs), 1)
            eta_min = float(params.get("eta_min", 0.0))
            self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
        elif st == "plateau":
            factor = float(params.get("factor", 0.5))
            patience = int(params.get("patience", 5))
            min_lr = float(params.get("min_lr", 0.0))
            mode = params.get("mode", "min")
            self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )
        else:
            raise ValueError(f"Unsupported scheduler '{st}'.")

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
    
    def train(
        self,
        num_epochs: int = 10,
        restore_best_model: bool = True,
    ) -> PolicyTrainingMetrics:
        """
        Train a simple MLP policy model on the provided imitation-learning dataset.
        
        Parameters
        ----------
        num_epochs : int, optional
            Number of training epochs. Default is 10.
        restore_best_model : bool, optional
            If ``True`` and early stopping is enabled, restore best weights before returning.

        Returns
        -------
        PolicyTrainingMetrics
            Container with per-epoch training and validation loss, learning rates, and total epochs completed.
        """
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        
        self.loss_fn.to(self.device)
        self.model.to(self.device)
        self.model.train()
        
        self._configure_scheduler(num_epochs=num_epochs)

        train_datapoints = max(len(self.dataloader.dataset), 1)
        use_validation = self.val_dataloader is not None
        val_datapoints = max(len(self.val_dataloader.dataset), 1) if use_validation else 0
        monitored_name = "Val" if use_validation else "Train"
        metrics = PolicyTrainingMetrics.from_num_epochs(num_epochs)
        bar_step = 1.0 / (train_datapoints + val_datapoints)

        with __logger__.tqdm(
            total=float(num_epochs),
            desc="Training Policy",
            unit="epoch",
            bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for epoch in range(num_epochs):
                train_epoch_loss = 0.0
                self.model.train()

                # Training loop
                for batch in self.dataloader:
                    nn_inputs, states, actions = self._extract_batch(batch)

                    self.optimizer.zero_grad(set_to_none=True)
                    pred_actions = self.model(nn_inputs)
                    
                    if self._loss_requires_states:
                        loss = self.loss_fn(pred_actions, actions, states=states)
                    else:
                        loss = self.loss_fn(pred_actions, actions)

                    loss.backward()
                    self.optimizer.step()

                    train_epoch_loss += loss.item() * nn_inputs.size(0)
                    pbar.update(bar_step * nn_inputs.size(0))

                train_avg_loss = train_epoch_loss / train_datapoints

                # Validation loop
                val_avg_loss: float | None = None
                if use_validation and self.val_dataloader is not None:
                    self.model.eval()
                    val_epoch_loss = 0.0
                    with th.no_grad():
                        for batch in self.val_dataloader:
                            nn_inputs, states, actions = self._extract_batch(batch)
                            pred_actions = self.model(nn_inputs)

                            if self._loss_requires_states:
                                val_epoch_loss += self.loss_fn(pred_actions, actions, states=states).item() * nn_inputs.size(0)
                            else:
                                val_epoch_loss += self.loss_fn(pred_actions, actions).item() * nn_inputs.size(0)

                            pbar.update(bar_step * nn_inputs.size(0))
                    val_avg_loss = val_epoch_loss / val_datapoints

                # Update metrics
                metrics.train_loss[epoch] = float(train_avg_loss)
                metrics.val_loss[epoch] = float(val_avg_loss) if val_avg_loss is not None else np.nan
                metrics.learning_rate[epoch] = float(self.optimizer.param_groups[0]["lr"])
                metrics.epochs_completed = epoch + 1

                monitored_metric = val_avg_loss if val_avg_loss is not None else train_avg_loss

                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, th.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(monitored_metric)
                    else:
                        self.scheduler.step()

                # Early stopping
                if self.early_stopper is not None:
                    self.early_stopper(monitored_metric, self.model)
                    if self.early_stopper.early_stop:
                        __logger__.info(
                            "Early stopping at epoch %d (%s loss %.6f).",
                            epoch + 1,
                            monitored_name,
                            monitored_metric,
                        )
                        break

                # Postfix for tqdm bar
                postfix = {
                    "Train": f"{train_avg_loss:.4f}",
                    "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
                if val_avg_loss is not None:
                    postfix["Val"] = f"{val_avg_loss:.4f}"
                pbar.set_postfix(postfix)
                
        self.metrics = metrics
        
        if self.early_stopper is not None and restore_best_model and self.early_stopper.best_model_state is not None:
            self.early_stopper.load_best_model(self.model)

        return metrics

    def save(
        self,
        save_folder: os.PathLike[str],
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
        if save_folder is not None:
            save_folder = Path(save_folder)

            # Dataset saving
            train_dataset_path = save_folder / "train_dataset.pt"
            train_saved = save_state_action_dataset_subset(self.dataloader.dataset, train_dataset_path)

            val_saved = False
            if self.val_dataloader is not None:
                val_dataset_path = save_folder / "val_dataset.pt"
                val_saved = save_state_action_dataset_subset(self.val_dataloader.dataset, val_dataset_path)
            else:
                val_dataset_path = None

            resolved_train_dataset_path = str(train_dataset_path) if train_saved else None
            resolved_val_dataset_path = str(val_dataset_path) if val_saved and val_dataset_path is not None else None

            # Policy model saving
            model_path = save_folder / "model.pt"
            self.model.save(
                model_path,
                train_dataset_path=resolved_train_dataset_path,
                val_dataset_path=resolved_val_dataset_path,
                global_config=global_config,
            )

            # Metrics
            metrics_path = save_folder / "training_metrics.npz"
            self.metrics.save(metrics_path)
            
            __logger__.info("Saved training results to %s", metrics_path.parent)
