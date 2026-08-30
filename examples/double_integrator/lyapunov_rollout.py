from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
from mpc_datagen import (
    MPCDataset,
    StabilityVerifier,
    VerificationRender,
    create_error_dataset,
    mdg_plt,
)

from lcil.rollouts import build_rollout_dataset
from lcil.utils.base_config import ArgumentParserConfig, config_field

from . import (
    DoubleIntegratorDynamics,
    default_dataset_path,
    discover_latest_lyapunov_dir,
    discover_source_rollout_path,
    find_all_lyapunov_dirs,
    get_initial_states,
    load_lyapunov_model,
    load_mpc_config,
    load_policy_model,
    require_dir,
    require_file,
    resolve_dataset_path,
    resolve_expert_dataset_path,
    build_lyapunov_func,
)
from ..constants import (
    LYAPUNOV_MODEL_FILENAME,
    LYAPUNOV_ROLLOUT_FILENAME,
    POLICY_MODEL_FILENAME,
    TRAINING_CONFIG_FILENAME,
)

__logger__ = logging.getLogger("lcil.examples.double_integrator.lyapunov_rollout")


@dataclass(frozen=True)
class LyapunovRolloutScriptConfig(ArgumentParserConfig):
    lyapunov_dir: str = config_field(help="Lyapunov run directory containing lyapunov_model.pt.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    time_steps: int = config_field(default=40, help="Number of time steps to rollout for each initial state.")


def _build_script_defaults() -> LyapunovRolloutScriptConfig:
    try:
        default_lyapunov_dir = discover_latest_lyapunov_dir()
        default_dir_str = str(default_lyapunov_dir)
    except Exception as e:
        __logger__.debug("Could not discover default lyapunov directory: %s", e)
        default_dir_str = ""
    return LyapunovRolloutScriptConfig(
        lyapunov_dir=default_dir_str,
    )


def parse_args() -> LyapunovRolloutScriptConfig:
    script_defaults = _build_script_defaults()
    parser = argparse.ArgumentParser(
        description="Copy the policy rollout dataset into a Lyapunov run directory and add Lyapunov values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    script_defaults.add_to_argparse(parser)
    args = parser.parse_args()
    return script_defaults.from_namespace(args)


def main() -> None:
    script_config = parse_args()
    if not script_config.lyapunov_dir:
        raise ValueError("No --lyapunov-dir specified and no default Lyapunov directory could be discovered.")
    device = th.device(script_config.device)

    lyapunov_dir = require_dir(script_config.lyapunov_dir, name="Lyapunov run directory")

    # Discover expert dataset for 1:1 error comparison
    expert_dataset_path = resolve_expert_dataset_path(lyapunov_dir)
    expert_subset: MPCDataset | None = None
    initial_states: np.ndarray | None = None

    if expert_dataset_path is not None and expert_dataset_path.is_file():
        __logger__.info("Using expert MPC dataset for 1:1 comparison: %s", expert_dataset_path)
        try:
            full_expert_ds = MPCDataset.load(expert_dataset_path)
            expert_subset = full_expert_ds
            initial_states = np.stack(
                [entry.trajectory.states[0] for entry in expert_subset],
                axis=0,
            )
        except Exception as e:
            __logger__.warning("Failed to load expert dataset from %s: %s", expert_dataset_path, e)
            expert_subset = None
            initial_states = None

    try:
        source_rollout_path = discover_source_rollout_path(lyapunov_dir)
        rollout_dataset = MPCDataset.load(source_rollout_path)
        initial_states = get_initial_states(rollout_dataset)
        if expert_subset is not None:
            expert_subset = expert_subset[:len(initial_states)]
        del rollout_dataset
    except Exception as e:
        __logger__.debug("Source rollout dataset discovery note: %s", e)

    if initial_states is None:
        raise ValueError(
            f"Could not determine initial states for rollout from either expert dataset or "
            f"source rollout in '{lyapunov_dir}'."
        )

    if (lyapunov_dir / LYAPUNOV_MODEL_FILENAME).is_file():
        lyapunov_dirs = find_all_lyapunov_dirs(lyapunov_dir.parent)
        if not lyapunov_dirs:
            lyapunov_dirs = [lyapunov_dir]
    else:
        lyapunov_dirs = find_all_lyapunov_dirs(lyapunov_dir)

    if len(lyapunov_dirs) == 0:
        raise ValueError(f"No Lyapunov run directories found in {script_config.lyapunov_dir}")

    __logger__.info("Found %d Lyapunov run directories under %s for rollout generation", len(lyapunov_dirs), lyapunov_dir.parent)
    for curr_lyapunov_dir in lyapunov_dirs:
        policy_path = require_file(curr_lyapunov_dir / POLICY_MODEL_FILENAME, name="Policy checkpoint")
        lyapunov_path = require_file(curr_lyapunov_dir / LYAPUNOV_MODEL_FILENAME, name="Lyapunov checkpoint")
        output_path = curr_lyapunov_dir / LYAPUNOV_ROLLOUT_FILENAME

        policy_model = load_policy_model(policy_path, device)
        mpc_cfg = load_mpc_config(curr_lyapunov_dir)
        dyn_model = DoubleIntegratorDynamics(
            dt=mpc_cfg.dt,
            abcrown_compatible_ops=True,
        )
        lyap_model = load_lyapunov_model(lyapunov_path, device)

        dataset = build_rollout_dataset(
            policy_model=policy_model,
            mpc_config=mpc_cfg,
            dyn_model=dyn_model,
            lyap_model=lyap_model,
            device=device,
            rollout_steps=script_config.time_steps,
            initial_states=initial_states,
        )
        if dataset is None:
            __logger__.warning("Failed to generate rollout dataset for %s", curr_lyapunov_dir)
            continue

        dataset.validate()
        dataset.save(output_path, save_ocp_trajs=False)
        __logger__.info("Saved Lyapunov rollout dataset to %s", output_path)

        veri_stats = StabilityVerifier.verify(dataset)
        VerificationRender(veri_stats).render()

        state_labels = [r"$x$", r"$v$"]
        control_labels = [r"$u$"]

        mdg_plt.mpc_trajectories(
            dataset=dataset,
            state_labels=state_labels,
            control_labels=control_labels,
            plot_predictions=False,
            html_path=curr_lyapunov_dir / "trajectories.html",
        )

        mdg_plt.lyapunov(
            lyapunov_func=build_lyapunov_func(lyap_model, device),
            dataset=dataset,
            state_labels=state_labels,
            plot_3d=False,
            html_path=(curr_lyapunov_dir / LYAPUNOV_ROLLOUT_FILENAME).with_suffix(".html"),
            use_dataset_v=True,
        )
        mdg_plt.cost_descent(
            dataset=dataset,
            html_path=curr_lyapunov_dir / "cost_descent.html",
            use_optimal_v=True,
        )

        # Error dataset and error bands plot against expert MPC using mpc_datagen
        if expert_subset is not None and len(expert_subset) > 0:
            __logger__.info("Creating error dataset and error bands plot via mpc_datagen...")
            diff_output_path = curr_lyapunov_dir / "policy_diff_rollouts.hdf5"
            error_dataset = create_error_dataset(
                dataset_a=dataset,
                dataset_b=expert_subset[:len(dataset)],
                file_path=diff_output_path,
            )

            if len(error_dataset) > 0:
                error_plot_path = curr_lyapunov_dir / "policy_error_bands.html"
                mdg_plt.trajectory_error_bands(
                    errors_dataset=error_dataset,
                    state_labels=[r"$\Delta x$", r"$\Delta v$"],
                    control_labels=[r"$\Delta u$"],
                    plot_controls=True,
                    show_individual=False,
                    show_median=True,
                    html_path=str(error_plot_path),
                )
                __logger__.info("Saved error bands plot to %s", error_plot_path)

            # Comparative trajectory plot for 5 random initial condition pairs
            n_pairs = min(5, len(dataset), len(expert_subset))
            if n_pairs > 0:
                rng = np.random.default_rng(seed=42)
                selected_indices = rng.choice(
                    min(len(dataset), len(expert_subset)),
                    size=n_pairs,
                    replace=False,
                ).tolist()
                comp_plot_path = curr_lyapunov_dir / "policy_vs_expert_trajectories.html"
                mdg_plt.mpc_trajectories(
                    dataset[selected_indices],
                    expert_subset[selected_indices],
                    dataset_labels=["Co-trained Policy", "Expert MPC"],
                    state_labels=state_labels,
                    control_labels=control_labels,
                    plot_predictions=False,
                    html_path=comp_plot_path,
                )
                __logger__.info("Saved trajectory comparison plot to %s", comp_plot_path)


if __name__ == "__main__":
    main()