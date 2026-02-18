from __future__ import annotations

from pathlib import Path

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.package_logger import get_package_logger

__logger__ = get_package_logger(__name__)


def train_mlp_policy(
    policy_model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module | None = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: th.device | str = "cpu",
    save_path: str | Path | None = None,
) -> None:
    """
    Train a simple MLP policy model on the provided imitation-learning dataset.
    
    Parameters
    ----------
    policy_model : nn.Module
        The MLP policy model to be trained. Should take state tensors as input and output action tensors.
    dataloader : torch.utils.data.DataLoader
        The imitation learning dataloader providing (state, action) pairs for training.
    num_epochs : int, optional
        Number of training epochs. Default is 10.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 1e-3.
    device : torch.device or str, optional
        Device to run training on (e.g., "cpu" or "cuda"). Default is "cpu".
    save_path : str | pathlib.Path or None, optional
        If provided, save the trained model state dict to this path after training.
    """
    device = th.device(device)
    policy_model.to(device)
    policy_model.train()

    optimizer = th.optim.Adam(policy_model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
    loss_fn = loss_fn.to(device)
    
    num_batches = max(len(dataloader), 1)
    batch_progress = 1.0 / num_batches
    datapoints = len(dataloader.dataset)

    with __logger__.tqdm(
        total=float(num_epochs),
        desc="Training MLP Policy",
        unit="epoch",
        bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as pbar:
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for states, actions in dataloader:
                states, actions = states.to(device), actions.to(device)

                optimizer.zero_grad()
                pred_actions = policy_model(states)
                loss = loss_fn(pred_actions, actions)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * states.size(0)
                pbar.update(batch_progress)

            avg_loss = epoch_loss / datapoints
            
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

    if save_path is not None:
        save_path = Path(save_path)
        if hasattr(policy_model, "save") and callable(policy_model.save):
            policy_model.save(save_path)
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            th.save(policy_model.state_dict(), save_path)
        __logger__.info("Saved trained policy model to %s", save_path)
