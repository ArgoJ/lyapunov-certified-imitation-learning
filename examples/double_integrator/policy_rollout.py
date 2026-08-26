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
from lcil.imitation_learning import load_imitation_dataset
from lcil.rollouts import FeasibleSetSampler, build_policy_rollout_dataset
from lcil.utils import ArgumentParserConfig, GridSearchHelper, IntegrationMethod, config_field

from . import (
    DoubleIntegratorDynamics,
    compute_riccati_value_matrix,
    default_dataset_path,
    discover_latest_policy_dir,
    find_all_policy_dirs,
    load_mpc_config,
    load_policy_model,
    require_dir,
    resolve_dataset_path,
)
from ..constants import (
    POLICY_MODEL_FILENAME,
    POLICY_ROLLOUT_FILENAME,
    TRAINING_CONFIG_FILENAME,
)

__logger__ = logging.getLogger("lcil.examples.double_integrator.policy_rollout")


@dataclass(frozen=True)
class PolicyRolloutScriptConfig(ArgumentParserConfig):
    """Configuration for double-integrator policy rollout script."""

    policy_dir: str = config_field(help="Policy run directory containing policy_model.pt.")
    dataset_path: str | None = config_field(
        default=None,
        help="Path to expert MPC dataset for 1:1 error comparison. Auto-discovered if None.",
    )
    n_samples: int = config_field(default=500, help="Number of rollout initial states.")
    time_steps: int = config_field(default=40, help="Number of time steps to rollout for each initial state.")
    device: str = config_field(default="cpu", help="Torch device string (e.g. cpu, cuda).")


def _build_script_defaults() -> PolicyRolloutScriptConfig:
    default_policy_dir = discover_latest_policy_dir()
    return PolicyRolloutScriptConfig(
        policy_dir=str(default_policy_dir),
    )


def parse_cli_args() -> GridSearchHelper[PolicyRolloutScriptConfig]:
    parser = argparse.ArgumentParser(
        description="Roll out a trained double-integrator imitation policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    script_defaults = _build_script_defaults()
    script_defaults.add_to_argparse(parser)

    args = parser.parse_args()
    return GridSearchHelper.from_namespace(args, script_defaults)


def _set_quadratic_vn(dataset: MPCDataset, P: np.ndarray) -> None:
    """Populate ``trajectory.V_N`` with the quadratic surrogate ``x.T @ P @ x``."""
    for entry in dataset:
        x = np.asarray(entry.trajectory.states[:-1], dtype=np.float64)
        entry.trajectory.V_N = np.einsum("bi,ij,bj->b", x, P, x)


def _resolve_expert_dataset_path(
    policy_dir: Path,
    explicit_path: str | None,
) -> Path | None:
    """Find the matching expert MPC dataset path for the policy run."""
    if explicit_path is not None and str(explicit_path).strip():
        try:
            resolved = resolve_dataset_path(explicit_path)
            if resolved.is_file():
                return resolved
        except Exception as e:
            __logger__.warning("Could not resolve explicit dataset path '%s': %s", explicit_path, e)

    # Check training_config.json in policy_dir or timestamp_dir
    for search_dir in (policy_dir, policy_dir.parent):
        cfg_file = search_dir / TRAINING_CONFIG_FILENAME
        if cfg_file.is_file():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                ds_path_str = cfg_data.get("dataset_path")
                if ds_path_str:
                    resolved = Path(ds_path_str).resolve()
                    if resolved.is_file():
                        return resolved
            except Exception as e:
                __logger__.debug("Error reading dataset_path from %s: %s", cfg_file, e)

    # Fall back to latest default dataset
    try:
        def_path = Path(default_dataset_path()).resolve()
        if def_path.is_file():
            return def_path
    except Exception as e:
        __logger__.debug("Could not discover default dataset path: %s", e)

    return None


def main() -> None:
    sweep = parse_cli_args()
    processed_policy_dirs: set[Path] = set()

    for run_idx, script_config in enumerate(sweep.configs, start=1):
        device = th.device(script_config.device)
        init_policy_dir = require_dir(script_config.policy_dir, name="Policy run directory")
        timestamp_dir = (
            init_policy_dir.parent
            if (init_policy_dir / POLICY_MODEL_FILENAME).is_file()
            else init_policy_dir
        )

        policy_dirs = find_all_policy_dirs(timestamp_dir)
        if not policy_dirs:
            if (init_policy_dir / POLICY_MODEL_FILENAME).is_file():
                policy_dirs = [init_policy_dir]
            else:
                raise FileNotFoundError(
                    f"No policy directories containing '{POLICY_MODEL_FILENAME}' found in '{timestamp_dir}'."
                )

        __logger__.info(
            "[%d/%d] Found %d policy directories under %s for rollout generation",
            run_idx,
            len(sweep.configs),
            len(policy_dirs),
            timestamp_dir,
        )

        for p_dir in policy_dirs:
            p_dir = p_dir.resolve()
            if p_dir in processed_policy_dirs:
                continue
            processed_policy_dirs.add(p_dir)

            __logger__.info("Rolling out policy from %s", p_dir)
            net = load_policy_model(p_dir, device)
            mpc_cfg = load_mpc_config(p_dir)

            # Discover expert dataset for 1:1 comparison
            expert_dataset_path = _resolve_expert_dataset_path(p_dir, script_config.dataset_path)
            expert_subset: MPCDataset | None = None
            initial_states: np.ndarray | None = None

            if expert_dataset_path is not None and expert_dataset_path.is_file():
                __logger__.info("Using expert MPC dataset for 1:1 comparison: %s", expert_dataset_path)
                try:
                    full_expert_ds = MPCDataset.load(expert_dataset_path)
                    n_eval = min(script_config.n_samples, len(full_expert_ds))
                    expert_subset = full_expert_ds[:n_eval]
                    initial_states = np.stack(
                        [entry.trajectory.states[0] for entry in expert_subset],
                        axis=0,
                    )
                except Exception as e:
                    __logger__.warning("Failed to load expert dataset from %s: %s", expert_dataset_path, e)
                    expert_subset = None
                    initial_states = None

            sampler = None
            if initial_states is None:
                val_dataset_path = p_dir / "val_dataset.pt"
                if not val_dataset_path.exists() and timestamp_dir != p_dir:
                    alt_val_path = timestamp_dir / "val_dataset.pt"
                    if alt_val_path.exists():
                        val_dataset_path = alt_val_path

                if val_dataset_path.exists():
                    val_dataset = load_imitation_dataset(val_dataset_path)
                    sampler = FeasibleSetSampler(dataset=val_dataset)
                else:
                    __logger__.warning(
                        "Validation dataset not found at %s. Falling back to rollout sampling from policy bounds.",
                        val_dataset_path,
                    )

            dt = mpc_cfg.dt
            simulator = DoubleIntegratorDynamics(
                dt=dt,
                method=IntegrationMethod.CLASSICAL_RK4,
            ).to(device)

            solved_dataset = build_policy_rollout_dataset(
                policy_model=net,
                mpc_config=mpc_cfg,
                dyn_model=simulator,
                rollout_steps=script_config.time_steps,
                device=device,
                initial_states=initial_states,
                sampler=sampler,
                n_samples=script_config.n_samples,
            )
            if solved_dataset is None:
                __logger__.warning("Failed to generate policy rollout dataset for %s", p_dir)
                continue

            p_matrix = compute_riccati_value_matrix(dt)
            _set_quadratic_vn(solved_dataset, p_matrix)
            solved_dataset.validate()

            output_path = p_dir / POLICY_ROLLOUT_FILENAME
            plot_path = output_path.with_suffix(".html")
            solved_dataset.save(path=output_path, save_ocp_trajs=False)

            veri_stats = StabilityVerifier.verify(solved_dataset)
            VerificationRender(veri_stats).render()

            state_labels = [r"$x$", r"$v$"]
            control_labels = [r"$u$"]

            mdg_plt.mpc_trajectories(
                dataset=solved_dataset,
                state_labels=state_labels,
                control_labels=control_labels,
                plot_predictions=False,
                html_path=str(plot_path),
            )

            # Error dataset and error bands plot against expert MPC using mpc_datagen
            if expert_subset is not None and len(expert_subset) > 0:
                __logger__.info("Creating error dataset and error bands plot via mpc_datagen...")
                diff_output_path = p_dir / "policy_diff_rollouts.hdf5"
                error_dataset = create_error_dataset(
                    dataset_a=solved_dataset,
                    dataset_b=expert_subset,
                    file_path=diff_output_path,
                )

                if len(error_dataset) > 0:
                    error_plot_path = p_dir / "policy_error_bands.html"
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


if __name__ == "__main__":
    main()
