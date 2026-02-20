import argparse
from pathlib import Path

from mpc_datagen import mdg_plt
from lcil.imitation_learning_mlp import MLPPolicy, StateActionDataset
from lcil.imitation_learning_mlp.policy_rollout import (
    PolicyRolloutGenerator,
    FeasibleSetSampler,
)
from double_integrator_dyn import DoubleIntegratorDynamics


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for policy rollout settings."""
    parser = argparse.ArgumentParser(description="Roll out a trained double-integrator imitation policy.")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of rollout initial states.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/double_integrator/20260220_160911/model.pt",
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

    print(f"Loaded policy model from {model_path} with config: {net.global_config}")
    print(f"dataset path: {net.val_dataset_path}")
    dataset = StateActionDataset.load(net.val_dataset_path)
    sampler = FeasibleSetSampler(dataset=dataset)
    cfg = net.global_config

    simulator = DoubleIntegratorDynamics(dt=net.global_config["dt"])
    policy_rollout_generator = PolicyRolloutGenerator(
        policy=net,
        simulator=simulator,
        sampler=sampler,
        device=device,
    )
    solved_dataset = policy_rollout_generator.generate(n_samples)
    
    base_folder = model_path.parent
    output_path = base_folder / "policy_rollouts.hdf5"
    plot_path = base_folder / "policy_rollouts.html"
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