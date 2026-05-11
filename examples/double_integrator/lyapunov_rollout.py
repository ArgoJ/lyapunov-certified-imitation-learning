from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch as th
from mpc_datagen import MPCDataset

from lcil.lyapunov_learning import LyapunovRollout
from lcil.utils.base_config import ArgumentParserConfig, config_field

from . import (
    discover_latest_lyapunov_dir,
    discover_latest_policy_dir,
    load_lyapunov_model,
)
from ..example_utils import require_dir, require_file

__logger__ = logging.getLogger("lcil.examples.double_integrator.lyapunov_rollout")

_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "double_integrator"
_SOURCE_ROLLOUT_FILE_NAME = "policy_rollouts.hdf5"
_OUTPUT_ROLLOUT_FILE_NAME = "lyapunov_rollout.hdf5"


@dataclass(frozen=True)
class LyapunovRolloutScriptConfig(ArgumentParserConfig):
    lyapunov_dir: str = config_field(help="Lyapunov run directory containing lyapunov_model.pt.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")


def _build_script_defaults() -> LyapunovRolloutScriptConfig:
    default_policy_dir = discover_latest_policy_dir(_DEFAULT_RESULTS_ROOT)
    default_lyapunov_dir = discover_latest_lyapunov_dir(default_policy_dir)
    return LyapunovRolloutScriptConfig(
        lyapunov_dir=str(default_lyapunov_dir),
    )


def parse_args() -> LyapunovRolloutScriptConfig:
    script_defaults = _build_script_defaults()
    parser = argparse.ArgumentParser(
        description="Copy the policy rollout dataset into a Lyapunov run directory and add Lyapunov values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    script_defaults.add_to_argparse(parser)
    args = parser.parse_args()
    return script_defaults.from_namespace(args)


def _infer_source_rollout_path(lyapunov_dir: Path) -> Path:
    for candidate_dir in [lyapunov_dir, *lyapunov_dir.parents]:
        rollout_path = candidate_dir / _SOURCE_ROLLOUT_FILE_NAME
        if rollout_path.is_file():
            return rollout_path.resolve()

    raise FileNotFoundError(
        f"Could not find '{_SOURCE_ROLLOUT_FILE_NAME}' in '{lyapunov_dir}' or any parent directory."
    )


def main() -> None:
    script_config = parse_args()
    device = th.device(script_config.device)

    lyapunov_dir = require_dir(script_config.lyapunov_dir, name="Lyapunov run directory")
    lyapunov_path = require_file(lyapunov_dir / "lyapunov_model.pt", name="Lyapunov checkpoint")
    source_rollout_path = _infer_source_rollout_path(lyapunov_dir)
    output_path = lyapunov_dir / _OUTPUT_ROLLOUT_FILE_NAME

    lyap_model = load_lyapunov_model(lyapunov_path, device)
    rollout_dataset = MPCDataset.load(source_rollout_path)

    __logger__.info("Loaded source rollout dataset from %s", source_rollout_path)
    __logger__.info("Loaded Lyapunov model from %s", lyapunov_path)

    lyapunov_rollout = LyapunovRollout(
        mpc_dataset=rollout_dataset,
        lyap_model=lyap_model,
        device=device,
    )
    saved_path = lyapunov_rollout.rollout(output_path=output_path)
    __logger__.info("Saved Lyapunov rollout dataset to %s", saved_path)


if __name__ == "__main__":
    main()