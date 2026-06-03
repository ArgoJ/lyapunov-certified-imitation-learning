import argparse
import logging
import numpy as np

from dataclasses import dataclass, replace
from pathlib import Path

from mpc_datagen import mdg_plt, MPCDataset
from mpc_datagen.verification import StabilityVerifier, VerificationRender
from lcil.imitation_learning import StateActionDataset
from lcil.rollouts import FeasibleSetSampler, PolicyRolloutGenerator
from lcil.utils import ArgumentParserConfig, GridSearchHelper, config_field

from . import (
    CARTPOLE_RESULTS_DIR,
    CartpoleDynamics,
    compute_riccati_value_matrix,
    discover_latest_policy_dir,
    load_policy_model,
)
from ..constants import POLICY_ROLLOUT_FILENAME

__logger__ = logging.getLogger("lcil.examples.cartpole.rollout")


@dataclass(frozen=True)
class PolicyRolloutScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(help="Policy run directory containing policy_model.pt.")
    n_samples: int = config_field(default=500, help="Number of rollout initial states.")
    t_sim: int = config_field(default=200, help="Closed-loop rollout horizon in simulation steps.")
    device: str = config_field(default="cpu", help="Torch device string (e.g. cpu, cuda).")


def _build_script_defaults() -> PolicyRolloutScriptConfig:
    return PolicyRolloutScriptConfig(
        policy_dir=str(discover_latest_policy_dir(CARTPOLE_RESULTS_DIR)),
    )


def parse_cli_args() -> GridSearchHelper[PolicyRolloutScriptConfig]:
    parser = argparse.ArgumentParser(
        description="Roll out a trained inverted pendulum on cart imitation policy.",
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
    total_runs = len(sweep.configs)

    for run_idx, script_config in enumerate(sweep.configs, start=1):
        __logger__.info(
            "[%d/%d] Rolling out policy from %s",
            run_idx,
            total_runs,
            script_config.policy_dir,
        )
        device = script_config.device
        policy_dir = Path(script_config.policy_dir).resolve()

        net = load_policy_model(policy_dir, device)

        val_dataset_path = policy_dir / "val_dataset.pt"
        sampler = None
        if val_dataset_path.exists():
            val_dataset = StateActionDataset.load(val_dataset_path)
            sampler = FeasibleSetSampler(dataset=val_dataset)
        else:
            __logger__.warning(
                "Validation dataset not found at %s. Falling back to rollout sampling from policy bounds.",
                val_dataset_path,
            )

        cfg = replace(net.net.global_config, T_sim=int(script_config.t_sim))

        simulator = CartpoleDynamics(dt=cfg.dt)
        policy_rollout_generator = PolicyRolloutGenerator(
            policy=net,
            simulator=simulator,
            cfg=cfg,
            sampler=sampler,
            device=device,
        )
        solved_dataset = policy_rollout_generator.generate(int(script_config.n_samples))

        p_matrix = compute_riccati_value_matrix(float(cfg.dt))
        _set_quadratic_vn(solved_dataset, p_matrix)
        solved_dataset.validate()

        output_path = policy_dir / POLICY_ROLLOUT_FILENAME
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
