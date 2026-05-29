import argparse
import logging
import numpy as np
import sys

from pathlib import Path
from scipy.linalg import solve_discrete_are

from mpc_datagen import mdg_plt, StabilityVerifier, VerificationRender, mdg_linalg, MPCDataset
from lcil.utils import IntegrationMethod
from lcil.imitation_learning import load_imitation_dataset
from lcil.rollouts import (
    FeasibleSetSampler,
    build_policy_rollout_dataset,
)

from . import (
    DoubleIntegratorDynamics,
    default_model_path,
    load_policy_model,
    compute_riccati_value_matrix,
)
from ..constants import *

__logger__ = logging.getLogger("lcil.examples.double_integrator.policy_rollout")


def _set_quadratic_vn(dataset: MPCDataset, P: np.ndarray) -> None:
    """Populate ``trajectory.V_N`` with the quadratic surrogate ``x.T @ P @ x``."""
    for entry in dataset:
        x = np.asarray(entry.trajectory.states[:-1], dtype=np.float64)
        entry.trajectory.V_N = np.einsum("bi,ij,bj->b", x, P, x)


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for policy rollout settings."""
    parser = argparse.ArgumentParser(
        description="Roll out a trained double-integrator imitation policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-samples", type=int, default=500, help="Number of rollout initial states.")
    parser.add_argument(
        "--policy-dir",
        type=str,
        default=default_model_path(),
        help="Path to a trained policy checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string (e.g. cpu, cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    device = args.device
    policy_dir = Path(args.policy_dir)
    n_samples = args.n_samples

    net = load_policy_model(policy_dir / POLICY_MODEL_FILENAME, device)

    val_dataset_path = policy_dir / "val_dataset.pt"
    __logger__.info(f"dataset path: {val_dataset_path}")
    if val_dataset_path.exists():
        dataset = load_imitation_dataset(val_dataset_path)
        sampler = FeasibleSetSampler(dataset=dataset)
    else:
        sampler = None
        __logger__.warning(
            "Policy model does not have a validation dataset path. "
            "FeasibleSetSampler will not be able to sample from the dataset for rollouts."
        )
    dt = net.global_config.dt

    simulator = DoubleIntegratorDynamics(
        dt=dt,
        method=IntegrationMethod.CLASSICAL_RK4
    ).to(device)

    solved_dataset = build_policy_rollout_dataset(
        policy_model=net,
        dyn_model=simulator,
        rollout_steps=40.0,
        device=device,
        sampler=sampler,
        n_samples=n_samples,
    )

    p_matrix = compute_riccati_value_matrix(dt)
    _set_quadratic_vn(solved_dataset, p_matrix)
    solved_dataset.validate()

    output_path = policy_dir / POLICY_ROLLOUT_FILENAME
    plot_path = output_path.with_suffix(".html")
    solved_dataset.save(path=output_path, save_ocp_trajs=False)

    veri_stats = StabilityVerifier.verify(solved_dataset)
    VerificationRender(veri_stats).render()

    mdg_plt.mpc_trajectories(
        dataset=solved_dataset,
        state_labels=["$x$", "$v$"],
        control_labels=["$u$"],
        plot_predictions=False,
        html_path=str(plot_path),
    )

if __name__ == "__main__":
    main()
