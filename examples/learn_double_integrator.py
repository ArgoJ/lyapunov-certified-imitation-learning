import torch as th
import numpy as np

from mpc_datagen import mdg_plt

from lyapunov_certified_imitation_learning.imitation_learning import train_mlp_policy, create_imitation_learning_dataloader
from lyapunov_certified_imitation_learning.utils import ResNet


def main() -> None:
    device = "cpu" #th.device("cuda" if th.cuda.is_available() else "cpu")
    dataset_path = "/home/josua/programming_stuff/projects/mpc-datagen/data/double_integrator_regional_N20_data.hdf5"
    dt = 0.1
    t_sim = 80
    u_min, u_max = -2.0, 2.0

    net = ResNet([2, 16, 16, 1], ["tanh", "tanh", "identity"]).to(device)
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
        num_epochs=4,
        batch_size=256,
        learning_rate=1e-3,
        device=device,
    )

    initial_states = [
        np.array([1.5, 0.0], dtype=np.float32),
        np.array([1.0, -0.5], dtype=np.float32),
        np.array([-1.25, 0.8], dtype=np.float32),
        np.array([0.6, -1.0], dtype=np.float32),
        np.array([-0.8, -0.6], dtype=np.float32),
    ]

    solved_dataset = PolicyRolloutGenerator(
        policy=net,
        t_sim=t_sim,
        dt=dt,
        u_min=u_min,
        u_max=u_max,
    )

    solved_dataset.save(path="results/policy_rollouts.hdf5", mode="w", save_ocp_trajs=False)
    mdg_plt.mpc_trajectories(
        dataset=solved_dataset,
        state_labels=["x", "v"],
        control_labels=["u"],
        plot_predictions=False,
        html_path="plots/policy_rollout_trajectories.html",
    )

if __name__ == "__main__":
	main()