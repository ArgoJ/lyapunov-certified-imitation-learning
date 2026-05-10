import argparse
import logging
import numpy as np
import sys

from pathlib import Path
from scipy.linalg import solve_discrete_are

from mpc_datagen import mdg_plt, StabilityVerifier, VerificationRender, mdg_linalg, MPCDataset
from lcil.utils import IntegrationMethod
from lcil.imitation_learning_mlp import MLPPolicy, StateActionDataset
from lcil.imitation_learning_mlp.policy_rollout import (
    PolicyRolloutGenerator,
    PolicyRolloutConfig,
    FeasibleSetSampler,
)

try:
    from . import DoubleIntegratorDynamics, default_model_path
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.double_integrator import DoubleIntegratorDynamics, default_model_path

__logger__ = logging.getLogger("lcil.examples.double_integrator.rollout")


def _compute_mpc_quadratic_p(dt: float) -> np.ndarray:
    """Compute the DARE Lyapunov matrix used by the MPC terminal ingredients."""
    A=np.array([[0.0, 1.0], [0.0, 0.0]])
    B=np.array([[0.0], [1.0]])

    a_d, b_d = mdg_linalg.lin_c2d_rk4(A, B, dt, num_steps=1)
    q = np.diag([15.0, 1.0])
    r = np.diag([0.1])
    return solve_discrete_are(a_d, b_d, q, r)


def _set_quadratic_vn(dataset: MPCDataset, P: np.ndarray) -> None:
    """Populate ``trajectory.V_N`` with the quadratic surrogate ``x.T @ P @ x``."""
    for entry in dataset:
        x = np.asarray(entry.trajectory.states[:-1], dtype=np.float64)
        entry.trajectory.V_N = np.einsum("bi,ij,bj->b", x, P, x)


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for policy rollout settings."""
    parser = argparse.ArgumentParser(description="Roll out a trained double-integrator imitation policy.")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of rollout initial states.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=default_model_path(),
        help="Path to a trained policy checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string (e.g. cpu, cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    device = args.device
    model_path = Path(args.model_path)
    n_samples = args.n_samples

    net = MLPPolicy.load(model_path, map_location=device)
    net.to(device)
    net.eval()

    val_dataset_path = model_path.parent / "val_dataset.pt"
    __logger__.info(f"dataset path: {val_dataset_path}")
    if val_dataset_path.exists():
        dataset = StateActionDataset.load(val_dataset_path)
        sampler = FeasibleSetSampler(dataset=dataset)
    else:
        sampler = None
        __logger__.warning(
            "Policy model does not have a validation dataset path. "
            "FeasibleSetSampler will not be able to sample from the dataset for rollouts."
        )

    cfg = PolicyRolloutConfig.from_mpc_config(net.global_config, t_sim=40.0)
    simulator = DoubleIntegratorDynamics(
        dt=cfg.dt,
        method=IntegrationMethod.CLASSICAL_RK4
    ).to(device)
    policy_rollout_generator = PolicyRolloutGenerator(
        policy=net,
        simulator=simulator,
        cfg=cfg,
        sampler=sampler,
        device=device,
    )
    solved_dataset = policy_rollout_generator.generate(n_samples)

    p_matrix = _compute_mpc_quadratic_p(cfg.dt)
    _set_quadratic_vn(solved_dataset, p_matrix)
    solved_dataset.validate()
    
    base_folder = model_path.parent
    output_path = base_folder / "policy_rollouts.hdf5"
    plot_path = base_folder / "policy_rollouts.html"
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
