import argparse
from pathlib import Path

from mpc_datagen import MPCDataset, mdg_plt

from lyapunov_certified_imitation_learning.imitation_learning_mlp import MLPPolicy, StateActionDataset
from lyapunov_certified_imitation_learning.imitation_learning_mlp.policy_rollout import (
    PolicyRolloutConfig,
    PolicyRolloutGenerator,
    RandomBoundsSampler,
    FeasibleSetSampler,
)
from double_integrator_dyn import DoubleIntegratorDynamics


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for policy rollout settings."""
    parser = argparse.ArgumentParser(description="Roll out a trained double-integrator imitation policy.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/josua/programming_stuff/projects/mpc-datagen/data/double_integrator_regional_N20_data.hdf5",
        help="Path to the source MPC dataset (HDF5).",
    )
    parser.add_argument("--n-samples", type=int, default=500, help="Number of rollout initial states.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/models/double_integrator_policy.pt",
        help="Path to a trained policy checkpoint.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/data/policy_rollouts.hdf5",
        help="Path where rollout trajectories (HDF5) will be saved.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="results/plots/policy_rollout_trajectories.html",
        help="Path where the rollout plot HTML will be saved.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string (e.g. cpu, cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    device = args.device
    dataset_path = args.dataset_path
    model_path = Path(args.model_path)
    n_samples = args.n_samples
    output_path = Path(args.output_path)
    plot_path = Path(args.plot_path)

    source_dataset = MPCDataset.load(Path(dataset_path))
    if len(source_dataset) == 0:
        raise ValueError("MPCDataset is empty; cannot extract configuration.")
    rollout_config = PolicyRolloutConfig.from_mpc_config(source_dataset[0].config, t_sim=40)

    net = MLPPolicy.load(model_path, map_location=device)
    net.to(device)
    net.eval()

    if net.train_dataset_path is None:
        sampler = RandomBoundsSampler(bounds=rollout_config.state_bounds)
    else:
        val_dataset = StateActionDataset.load(net.train_dataset_path)
        sampler = FeasibleSetSampler(dataset=val_dataset)

    simulator = DoubleIntegratorDynamics(dt=rollout_config.dt)
    policy_rollout_generator = PolicyRolloutGenerator(
        policy=net,
        simulator=simulator,
        rollout_config=rollout_config,
        sampler=sampler,
        device=device,
    )
    solved_dataset = policy_rollout_generator.generate(n_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    solved_dataset.save(path=output_path, save_ocp_trajs=False)
    mdg_plt.mpc_trajectories(
        dataset=solved_dataset,
        state_labels=["x", "v"],
        control_labels=["u"],
        plot_predictions=False,
        html_path=str(plot_path),
    )

if __name__ == "__main__":
    main()