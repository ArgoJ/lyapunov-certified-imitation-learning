import torch as th

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


def main() -> None:
    device = "cpu" #th.device("cuda" if th.cuda.is_available() else "cpu")
    dataset_path = "/home/josua/programming_stuff/projects/mpc-datagen/data/double_integrator_regional_N20_data.hdf5"
    n_samples = 500

    u_bounds = get_global_input_bounds(dataset_path)

    net = MLPPolicy(
        [2, 16, 16, 1],
        ["tanh", "tanh", "identity"],
        u_min=u_bounds[0],
        u_max=u_bounds[1],
    ).to(device)

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
        dataset=dataloader.dataset,
        num_epochs=1,
        batch_size=256,
        learning_rate=1e-3,
        device=device,
    )

    rollout_config = get_policy_rollout_config_from_dataset(dataset_path, t_sim=80)
    policy_rollout_generator = PolicyRolloutGenerator(
        policy=net,
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