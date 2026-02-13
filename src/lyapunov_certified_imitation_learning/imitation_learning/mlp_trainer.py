from __future__ import annotations

import os

import torch as th
import torch.nn as nn
from torch.utils.data import Dataset

from ..utils.package_logger import get_package_logger

__logger__ = get_package_logger(__name__)


def train_mlp_policy(
    policy_model: nn.Module,
    dataset: Dataset,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: th.device | str = "cpu",
) -> None:
    """
    Train a simple MLP policy model on the provided imitation-learning dataset.
    
    Parameters
    ----------
    policy_model : nn.Module
        The MLP policy model to be trained. Should take state tensors as input and output action tensors.
    dataset : torch.utils.data.Dataset
        The imitation learning dataset providing (state, action) pairs for training.
    num_epochs : int, optional
        Number of training epochs. Default is 10.
    batch_size : int, optional
        Number of samples per training batch. Default is 64.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 1e-3.
    device : torch.device or str, optional
        Device to run training on (e.g., "cpu" or "cuda"). Default is "cpu".
    """
    device = th.device(device)
    policy_model.to(device)

    dataloader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = th.optim.Adam(policy_model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    num_batches = max(len(dataloader), 1)
    batch_progress = 1.0 / num_batches

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

            avg_loss = epoch_loss / len(dataset)
            
            pbar.set_postfix({"epoch": epoch + 1, "Loss": f"{avg_loss:.6f}"})
