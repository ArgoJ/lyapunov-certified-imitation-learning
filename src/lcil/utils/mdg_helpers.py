from __future__ import annotations

import json

from os import PathLike
from pathlib import Path

from mpc_datagen.mpc_data import MPCConfig

from .constants import MPC_CONFIG_FILENAME


def save_mpc_config_json(mpc_config: MPCConfig, path: str | PathLike[str]) -> Path:
    """Persist an ``MPCConfig`` to a human-readable JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
     
    if not isinstance(mpc_config, MPCConfig):
        raise TypeError(f"Expected an MPCConfig instance, got {type(mpc_config)}")

    resolved_cfg = mpc_config.to_dict()
    output_path.write_text(
        json.dumps(resolved_cfg, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path

def save_mpc_config_for_run(mpc_config: MPCConfig, path: str | PathLike[str]) -> Path:
    """Persist an ``MPCConfig`` to a JSON file alongside a training run."""
    output_path = Path(path) / MPC_CONFIG_FILENAME
    return save_mpc_config_json(mpc_config, output_path)


def load_mpc_config_json(path: str | PathLike[str]) -> MPCConfig:
    """Load an ``MPCConfig`` from a JSON file."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Could not find MPC config JSON file at {input_path}")
    raw_cfg = json.loads(input_path.read_text(encoding="utf-8"))
    return MPCConfig.from_dict(raw_cfg)


def load_mpc_config_for_run(path: str | PathLike[str]) -> MPCConfig:
    """Load the MPC config JSON stored alongside a saved training run or from a direct file path."""
    resolved_path = Path(path)
    config_path = resolved_path if resolved_path.is_file() else resolved_path / MPC_CONFIG_FILENAME
    return load_mpc_config_json(config_path)