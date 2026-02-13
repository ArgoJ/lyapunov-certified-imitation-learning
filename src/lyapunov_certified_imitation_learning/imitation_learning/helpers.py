from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from mpc_datagen import MPCDataset

from .policy_rollout import PolicyRolloutConfig


def _load_first_entry_config(mpc_dataset_path: str | os.PathLike[str]):
    dataset = MPCDataset.load(Path(mpc_dataset_path))
    if len(dataset) == 0:
        raise ValueError("MPCDataset is empty; cannot extract configuration.")
    return dataset[0].config


def get_global_input_bounds(mpc_dataset_path: str | os.PathLike[str]) -> np.ndarray:
    """Read dataset input bounds as shape `(2, nu)` from merged config constraints."""
    cfg = _load_first_entry_config(mpc_dataset_path)
    lbu = np.asarray(cfg.constraints.lbu, dtype=np.float32).reshape(-1)
    ubu = np.asarray(cfg.constraints.ubu, dtype=np.float32).reshape(-1)
    if lbu.size == 0 or ubu.size == 0:
        raise ValueError("Empty input bounds in dataset constraints ('lbu'/'ubu').")
    if lbu.size != ubu.size:
        raise ValueError(f"Mismatched input-bounds size: lbu={lbu.size}, ubu={ubu.size}.")

    return np.vstack((lbu, ubu))


def get_global_state_bounds(mpc_dataset_path: str | os.PathLike[str]) -> np.ndarray:
    """Read dataset state bounds as shape `(2, nx)` from merged config constraints."""
    cfg = _load_first_entry_config(mpc_dataset_path)
    lbx = np.asarray(cfg.constraints.lbx, dtype=np.float32).reshape(-1)
    ubx = np.asarray(cfg.constraints.ubx, dtype=np.float32).reshape(-1)
    if lbx.size == 0 or ubx.size == 0:
        raise ValueError("Empty state bounds in dataset constraints ('lbx'/'ubx').")
    if lbx.size != ubx.size:
        raise ValueError(f"Mismatched state-bounds size: lbx={lbx.size}, ubx={ubx.size}.")

    return np.vstack((lbx, ubx))


def get_policy_rollout_config_from_dataset(
    mpc_dataset_path: str | os.PathLike[str],
    t_sim: int | None = None,
) -> PolicyRolloutConfig:
    """Build `PolicyRolloutConfig` from dataset config and constraints."""
    cfg = _load_first_entry_config(mpc_dataset_path)

    return PolicyRolloutConfig(
        T_sim=int(cfg.T_sim) if t_sim is None else int(t_sim),
        dt=float(cfg.dt),
        nx=int(cfg.nx),
        nu=int(cfg.nu),
        N=int(cfg.N),
        state_bounds=get_global_state_bounds(mpc_dataset_path),
        input_bounds=get_global_input_bounds(mpc_dataset_path),
    )
