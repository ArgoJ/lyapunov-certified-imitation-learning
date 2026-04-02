import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

from mpc_datagen import mdg_plt, mdg_linalg, MPCDataset
from mpc_datagen.verification import StabilityVerifier, VerificationRender
from lcil.imitation_learning_mlp import MLPPolicy, StateActionDataset
from lcil.imitation_learning_mlp.policy_rollout import (
    PolicyRolloutGenerator,
    PolicyRolloutConfig,
    FeasibleSetSampler,
)
from cartpole_dyn import InvertedPendulumOnCartDynamics
from model import InvertedPendulumOnCartPolicy
from sys_cfg import PendulumOnCartConfig
from acados_ocp import _linearized_inverted_pendulum_on_cart_matrices



def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for policy rollout settings."""
    parser = argparse.ArgumentParser(description="Roll out a trained inverted pendulum on cart imitation policy.")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of rollout initial states.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/cartpole/20260402_203645/model.pt",
        help="Path to a trained policy checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string (e.g. cpu, cuda).")
    return parser.parse_args()


def _compute_mpc_quadratic_p(dt: float) -> np.ndarray:
    """Compute the DARE Lyapunov matrix used by the MPC terminal ingredients."""
    cfg = PendulumOnCartConfig()
    a_c, b_c = _linearized_inverted_pendulum_on_cart_matrices(cfg=cfg)

    a_d, b_d = mdg_linalg.lin_c2d_rk4(a_c, b_c, dt, num_steps=1)
    q = np.diag([1e2, 1e1, 1e2, 1e-2])
    r = np.diag([5e-1])
    return solve_discrete_are(a_d, b_d, q, r)


def _set_quadratic_vn(dataset: MPCDataset, P: np.ndarray) -> None:
    """Populate ``trajectory.V_N`` with the quadratic surrogate ``x.T @ P @ x``."""
    for entry in dataset:
        x = np.asarray(entry.trajectory.states[:-1], dtype=np.float64).copy()
        entry.trajectory.V_N = np.einsum("bi,ij,bj->b", x, P, x)


def main() -> None:
    args = parse_cli_args()
    device = args.device
    model_path = Path(args.model_path)
    n_samples = args.n_samples

    feature_net_path = model_path
    feature_net = MLPPolicy.load(feature_net_path, map_location=device)
    net = InvertedPendulumOnCartPolicy(feature_net=feature_net).to(device)
    net.eval()
    
    # Override dataset path using the absolute or relative location of the model path
    val_dataset_path = model_path.parent / "val_dataset.pt"

    print(f"dataset path: {val_dataset_path}")
    cfg = PolicyRolloutConfig.from_mpc_config(net.net.global_config, t_sim=40.0)
    dataset = StateActionDataset.load(val_dataset_path)
    sampler = FeasibleSetSampler(dataset=dataset)

    simulator = InvertedPendulumOnCartDynamics(dt=cfg.dt)
    policy_rollout_generator = PolicyRolloutGenerator(
        policy=net,
        simulator=simulator,
        rollout_config=cfg,
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
        state_labels=[r"$x$", r"$v$", r"$\theta$", r"$\dot{\theta}$"],
        control_labels=[r"$u$"],
        plot_predictions=False,
        html_path=str(plot_path),
    )

if __name__ == "__main__":
    main()
