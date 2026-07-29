import argparse
import logging
import numpy as np

from dataclasses import dataclass
from pathlib import Path

from mpc_datagen import mdg_plt, MPCDataset
from mpc_datagen.verification import StabilityVerifier, VerificationRender
from lcil.imitation_learning import load_imitation_dataset
from lcil.rollouts import FeasibleSetSampler, build_policy_rollout_dataset
from lcil.utils import ArgumentParserConfig, GridSearchHelper, config_field

from . import (
    CartpoleDynamics,
    compute_riccati_value_matrix,
    discover_latest_policy_dir,
    find_all_policy_dirs,
    load_mpc_config,
    load_policy_model,
    require_dir,
)
from ..constants import POLICY_MODEL_FILENAME, POLICY_ROLLOUT_FILENAME

__logger__ = logging.getLogger("lcil.examples.cartpole.policy_rollout")


@dataclass(frozen=True)
class PolicyRolloutScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(help="Policy run directory containing policy_model.pt.")
    n_samples: int = config_field(default=500, help="Number of rollout initial states.")
    time_steps: int = config_field(default=200, help="Number of time steps to rollout for each initial state.")
    device: str = config_field(default="cpu", help="Torch device string (e.g. cpu, cuda).")


def _build_script_defaults() -> PolicyRolloutScriptConfig:
    default_policy_dir = discover_latest_policy_dir()
    return PolicyRolloutScriptConfig(
        policy_dir=str(default_policy_dir),
    )


def parse_cli_args() -> GridSearchHelper[PolicyRolloutScriptConfig]:
    parser = argparse.ArgumentParser(
        description="Roll out trained inverted pendulum on cart imitation policy.",
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


def main() -> None:
    sweep = parse_cli_args()
    processed_policy_dirs: set[Path] = set()

    for run_idx, script_config in enumerate(sweep.configs, start=1):
        device = script_config.device
        init_policy_dir = require_dir(script_config.policy_dir, name="Policy run directory")
        timestamp_dir = init_policy_dir.parent if (init_policy_dir / POLICY_MODEL_FILENAME).is_file() else init_policy_dir

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

            val_dataset_path = p_dir / "val_dataset.pt"
            if not val_dataset_path.exists() and timestamp_dir != p_dir:
                alt_val_path = timestamp_dir / "val_dataset.pt"
                if alt_val_path.exists():
                    val_dataset_path = alt_val_path

            sampler = None
            if val_dataset_path.exists():
                val_dataset = load_imitation_dataset(val_dataset_path)
                sampler = FeasibleSetSampler(dataset=val_dataset)
            else:
                __logger__.warning(
                    "Validation dataset not found at %s. Falling back to rollout sampling from policy bounds.",
                    val_dataset_path,
                )

            dt = float(mpc_cfg.dt)

            solved_dataset = build_policy_rollout_dataset(
                policy_model=net,
                mpc_config=mpc_cfg,
                dyn_model=CartpoleDynamics(dt=dt).to(device),
                rollout_steps=int(script_config.time_steps),
                device=device,
                sampler=sampler,
                n_samples=int(script_config.n_samples),
            )

            p_matrix = compute_riccati_value_matrix(float(dt))
            _set_quadratic_vn(solved_dataset, p_matrix)
            solved_dataset.validate()

            output_path = p_dir / POLICY_ROLLOUT_FILENAME
            plot_path = output_path.with_suffix(".html")
            solved_dataset.save(path=output_path, save_ocp_trajs=False)

            veri_stats = StabilityVerifier.verify(solved_dataset)
            VerificationRender(veri_stats).render()

            mdg_plt.mpc_trajectories(
                dataset=solved_dataset,
                state_labels=[r"$x$", r"$v$", r"$\theta$", r"$\dot{\theta}$"],
                control_labels=[r"$u$"],
                plot_predictions=False,
                html_path=str(plot_path),
            )


if __name__ == "__main__":
    main()

