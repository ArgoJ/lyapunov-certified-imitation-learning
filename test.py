from lyapunov_certified_imitation_learning.data_generation import MPCDataset
import lyapunov_certified_imitation_learning.utils as lcil_utils
import numpy as np


if "__main__" == __name__:
    P =  np.array([[26.07384505, 6.34428877],
                   [6.34428877, 8.6123929 ]])
    dataset = MPCDataset.load("double_integrator_mpc_dataset.hdf5")
    dataset.validate()
    lcil_utils.plot.mpc_trajectories(
        dataset=dataset,
        state_labels=["Position", "Velocity"],
        control_labels=["Acceleration"],
        plot_predictions=True
    )

    lyap = lambda x: x.T @ P @ x

    lcil_utils.plot.lyapunov(
        dataset=dataset, 
        lyapunov_func=lyap,
        plot_3d=True,
        limits=[[-12, 12], [-8, 8]]
    )