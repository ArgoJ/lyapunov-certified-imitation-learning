import torch as th
import torch.nn as nn

from mpc_datagen import mdg_plt

from lyapunov_certified_imitation_learning.imitation_learning import (
    train_mlp_policy,
    create_imitation_learning_dataloader,
    MLPPolicy,
)
from lyapunov_certified_imitation_learning.imitation_learning.helpers import (
    get_policy_rollout_config_from_dataset,
    get_global_input_bounds,
)
from lyapunov_certified_imitation_learning.imitation_learning.policy_rollout import PolicyRolloutGenerator


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


def main() -> None:
    device = "cpu" #th.device("cuda" if th.cuda.is_available() else "cpu")
    dataset_path = "/home/josua/programming_stuff/projects/mpc-datagen/data/double_integrator_regional_N20_data.hdf5"
    n_samples = 500

    u_bounds = get_global_input_bounds(dataset_path)

    net = MLPPolicy(
        [2, 16, 16, 1],
        ["relu", "tanh", "identity"],
        u_min=u_bounds[0],
        u_max=u_bounds[1],
    )

    dataloader = create_imitation_learning_dataloader(
        mpc_dataset=dataset_path,
        batch_size=256,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        dtype=th.float32,
    )
    
    train_mlp_policy(
        policy_model=net,
        dataloader=dataloader,
        num_epochs=1,
        learning_rate=1e-3,
        device=device,
    )

    rollout_config = get_policy_rollout_config_from_dataset(dataset_path, t_sim=80)
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