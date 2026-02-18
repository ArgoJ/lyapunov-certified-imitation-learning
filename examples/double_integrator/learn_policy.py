import argparse
import torch as th
import torch.nn as nn
from pathlib import Path

from mpc_datagen import MPCDataset, mdg_plt

from lyapunov_certified_imitation_learning.imitation_learning_mlp import (
    train_mlp_policy,
    create_imitation_learning_dataloader,
    MLPPolicy,
    ReferenceWeightedMSELoss,
)
from lyapunov_certified_imitation_learning.imitation_learning_mlp.policy_rollout import (
    PolicyRolloutConfig,
    PolicyRolloutGenerator,
)


class DoubleIntegratorDynamics(nn.Module):
    def __init__(self, dt: float) -> None:
        super().__init__()
        self.dt = float(dt)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        if u.ndim == 1:
            u = u.unsqueeze(1)
        x_pos = x[:, 0:1]
        x_vel = x[:, 1:2]
        x_next_pos = x_pos + self.dt * x_vel
        x_next_vel = x_vel + self.dt * u
        return th.cat([x_next_pos, x_next_vel], dim=1)


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for training and rollout settings."""
    parser = argparse.ArgumentParser(description="Train and rollout a double-integrator imitation policy.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/josua/programming_stuff/projects/mpc-datagen/data/double_integrator_regional_N20_data_.hdf5",
        help="Path to the source MPC dataset (HDF5).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of policy training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Optimizer learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of rollout initial states.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/models/double_integrator_policy.pt",
        help="Path where the trained model state dict will be saved.",
    )
    parser.add_argument(
        "--near-duplicate-radius",
        type=float,
        default=1e-4,
        help="Optional near-duplicate L2 radius in normalized feature space.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string (e.g. cpu, cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    device = args.device
    dataset_path = args.dataset_path
    n_samples = args.n_samples

    source_dataset = MPCDataset.load(Path(dataset_path))
    if len(source_dataset) == 0:
        raise ValueError("MPCDataset is empty; cannot extract configuration.")
    rollout_config = PolicyRolloutConfig.from_mpc_config(source_dataset[0].config, t_sim=80)
    
    if rollout_config.input_bounds is None:
        u_bounds = (None, None)
    else:
        u_bounds = rollout_config.input_bounds

    net = MLPPolicy(
        [2, 16, 16, 1],
        ["relu", "tanh", "identity"],
        u_min=u_bounds[0],
        u_max=u_bounds[1],
    )

    dataloader = create_imitation_learning_dataloader(
        mpc_dataset=dataset_path,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        dtype=th.float32,
        near_duplicate_radius=args.near_duplicate_radius,
    )
    
    train_mlp_policy(
        policy_model=net,
        dataloader=dataloader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        loss_fn=ReferenceWeightedMSELoss(reference=[0.0], alpha=1.0, max_weight=10.0),
        save_path=args.model_path,
    )

    simulator = DoubleIntegratorDynamics(dt=rollout_config.dt)
    policy_rollout_generator = PolicyRolloutGenerator(
        policy=net,
        simulator=simulator,
        rollout_config=rollout_config,
        device=device,
    )
    solved_dataset = policy_rollout_generator.generate(n_samples)

    solved_dataset.save(path="results/data/policy_rollouts.hdf5", save_ocp_trajs=False)
    mdg_plt.mpc_trajectories(
        dataset=solved_dataset,
        state_labels=["x", "v"],
        control_labels=["u"],
        plot_predictions=False,
        html_path="results/plots/policy_rollout_trajectories.html",
    )

if __name__ == "__main__":
    main()