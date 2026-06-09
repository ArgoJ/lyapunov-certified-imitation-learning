from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch as th
from mpc_datagen import MPCDataset, mdg_plt, StabilityVerifier, VerificationRender

from lcil.rollouts import build_rollout_dataset
from lcil.utils.base_config import ArgumentParserConfig, config_field

from . import (
    DoubleIntegratorDynamics,
    discover_latest_lyapunov_dir,
    discover_latest_policy_dir,
    load_policy_model,
    load_lyapunov_model,
    require_dir,
    require_file,
    find_all_lyapunov_dirs,
    build_lyapunov_func,
    get_initial_states,
)
from ..constants import *

__logger__ = logging.getLogger("lcil.examples.double_integrator.lyapunov_rollout")



@dataclass(frozen=True)
class LyapunovRolloutScriptConfig(ArgumentParserConfig):
    lyapunov_dir: str = config_field(help="Lyapunov run directory containing lyapunov_model.pt.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    time_steps: int = config_field(default=40, help="Number of time steps to rollout for each initial state.")


def _build_script_defaults() -> LyapunovRolloutScriptConfig:
    default_policy_dir = discover_latest_policy_dir()
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
        rollout_path = candidate_dir / POLICY_ROLLOUT_FILENAME
        if rollout_path.is_file():
            return rollout_path.resolve()

    raise FileNotFoundError(
        f"Could not find '{POLICY_ROLLOUT_FILENAME}' in '{lyapunov_dir}' or any parent directory."
    )


def main() -> None:
    script_config = parse_args()
    device = th.device(script_config.device)

    lyapunov_dir = require_dir(script_config.lyapunov_dir, name="Lyapunov run directory")
    source_rollout_path = _infer_source_rollout_path(lyapunov_dir)
    rollout_dataset = MPCDataset.load(source_rollout_path)
    initial_states = get_initial_states(rollout_dataset)
    del rollout_dataset

    lyapunov_dirs = find_all_lyapunov_dirs(lyapunov_dir.parent)
    if len(lyapunov_dirs) == 0:
        raise ValueError(f"No Lyapunov run directories found in {script_config.lyapunov_dir}")
    
    __logger__.info("Found %d Lyapunov run directories under %s for rollout generation", len(lyapunov_dirs), lyapunov_dir.parent)
    for lyapunov_dir in lyapunov_dirs:
        policy_path = require_file(lyapunov_dir / POLICY_MODEL_FILENAME, name="Policy checkpoint")
        lyapunov_path = require_file(lyapunov_dir / LYAPUNOV_MODEL_FILENAME, name="Lyapunov checkpoint")
        output_path = lyapunov_dir / LYAPUNOV_ROLLOUT_FILENAME

        policy_model = load_policy_model(policy_path, device)
        dyn_model = DoubleIntegratorDynamics(
            dt=policy_model.global_config.dt,
            abcrown_compatible_ops=True,
        )
        lyap_model = load_lyapunov_model(lyapunov_path, device)

        dataset = build_rollout_dataset(
            policy_model=policy_model,
            dyn_model=dyn_model,
            lyap_model=lyap_model,
            device=device,
            rollout_steps=script_config.time_steps,
            initial_states=initial_states,
        )
        dataset.save(output_path, save_ocp_trajs=False)
        __logger__.info("Saved Lyapunov rollout dataset to %s", output_path)

        veri_stats = StabilityVerifier.verify(dataset)
        VerificationRender(veri_stats).render()

        mdg_plt.lyapunov(
            lyapunov_func=build_lyapunov_func(lyap_model, device),
            dataset=dataset,
            state_labels=["$x$", "$v$"],
            plot_3d=False,
            html_path=(lyapunov_dir / LYAPUNOV_ROLLOUT_FILENAME).with_suffix(".html"),
            use_dataset_v=True,
        )
        mdg_plt.cost_descent(
            dataset=dataset,
            html_path=lyapunov_dir / "cost_descent.html",
            use_optimal_v=True,
        )


if __name__ == "__main__":
    main()