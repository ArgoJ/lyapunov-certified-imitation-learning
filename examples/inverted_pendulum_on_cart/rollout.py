import argparse
from pathlib import Path

from mpc_datagen import mdg_plt
from lcil.imitation_learning_mlp import MLPPolicy, StateActionDataset
from lcil.imitation_learning_mlp.policy_rollout import (
    PolicyRolloutGenerator,
    PolicyRolloutConfig,
    FeasibleSetSampler,
)
from inv_pend_cart_dyn import InvertedPendulumOnCartDynamics
from model import InvertedPendulumOnCartPolicy



def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for policy rollout settings."""
    parser = argparse.ArgumentParser(description="Roll out a trained inverted pendulum on cart imitation policy.")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of rollout initial states.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/inverted_pendulum_on_cart/20260318_104704/model.pt",
        help="Path to a trained policy checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string (e.g. cpu, cuda).")
    return parser.parse_args()


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
    solved_dataset.validate()
    
    base_folder = model_path.parent
    output_path = base_folder / "policy_rollouts.hdf5"
    plot_path = base_folder / "policy_rollouts.html"
    solved_dataset.save(path=output_path, save_ocp_trajs=False)
    mdg_plt.mpc_trajectories(
        dataset=solved_dataset,
        state_labels=["x", "v", "theta", "theta_dot"],
        control_labels=["u"],
        plot_predictions=False,
        html_path=str(plot_path),
    )

if __name__ == "__main__":
    main()
