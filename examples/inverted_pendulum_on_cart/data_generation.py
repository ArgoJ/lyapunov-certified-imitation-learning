# %% [markdown] 
# # Inverted Pendulum on Cart - Data Generation
# This script generates MPC closed-loop datasets for the inverted pendulum on cart 
# system using an actual MPC solver. 
# It simulates trajectories starting from random initial states, collects the data, and
# performs some verification and visualization.


# %% General Imports
import argparse
import numpy as np
import logging

import mpc_datagen.linalg as mdg_linalg
import mpc_datagen.plots as mdg_plots
from mpc_datagen import *
from mpc_datagen.verification import (
    StabilityVerifier,
    VerificationRender,
    ROAVerifier,
)

from acados_ocp import get_ocp_solver
from pkg_logger import get_package_logger

__logger__ = get_package_logger("mpc_datagen")

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MPC imitation datasets for the inverted pendulum on cart."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20000,
        help="Number of trajectories to generate.",
    )
    parser.add_argument(
        "--t-sim",
        type=int,
        default=200,
        help="Simulation horizon length (number of MPC steps).",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="results/inverted_pendulum_on_cart/data",
        help="Base output path for generated datasets and plots.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, runs in debug mode with fewer samples and shorter simulation time.",
    )
    return parser


# %%  
if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if args.debug:
        __logger__.setLevel(logging.DEBUG)

    # Cost matrices
    Q = np.diag([1.0, 1e-4, 2.0, 1e-4])
    R = np.diag([1e-4])

    base_path = args.base_path

    T_sim = args.t_sim
    n_samples = args.n_samples

    dt = 0.01
    solver, info = get_ocp_solver(
        Q=Q,
        R=R,
        dt=dt,
        N=20,
        tol=1e-6,
        terminal_mode="regional",
    )

    sampler = UniqueBoundedSampler(
        bounds=np.array([solver.acados_ocp.constraints.lbx, solver.acados_ocp.constraints.ubx]),
        min_dist=np.array([1e-2, 1e-3, 1e-2, 1e-3]),
        seed=4597525,
    )
    eps_cfg = EpsBandConfig(
        eps_band=np.array([1e-1, 1e-2, 1e-2, 1e-1]), 
        eps_consecutive=3
    )
    generator = MPCDataGenerator(
        solver=solver,
        T_sim=T_sim,
        sampler=sampler,
        reset_solver=True,
        xeps_cfg=eps_cfg,
    )
    dataset = generator.generate(n_samples=n_samples, only_feasible=True)
    dataset.validate()
    dataset.save(f"{base_path}/inverted_pendulum_on_cart_N{T_sim}_data.hdf5")

    veri_stats = StabilityVerifier.verify(dataset, solver, alpha_required=1e-4)
    VerificationRender(veri_stats).render()

    P = info["P"]
    lyap_fun = lambda x: 0.5 * mdg_linalg.weighted_quadratic_norm(x, P)
    roa_lyap_fun = lambda x: mdg_linalg.weighted_quadratic_norm(x, P)
    roa_cert = ROAVerifier(dataset[0].config)
    roa_bounds, c_min = roa_cert.roa_bounds()

    mdg_plots.all(
        dataset=dataset[:min(150, n_samples)],
        state_labels=["x", "v", "theta", "theta_dot"],
        control_labels=["a"],
        time_bound=T_sim * dt,
        plot_3d=False,
        plot_predictions=False,
        alpha=1.0, 
        use_optimal_v=False,
        lyapunov_func=lyap_fun,
        lyap_state_indices=[1, 2],
        lyap_use_dataset_v=True,
        roa_lyapunov_func=roa_lyap_fun,
        c_level=c_min,
        roa_bounds=roa_bounds,
        base_path=f"{base_path}/inverted_pendulum_on_cart_N{T_sim}_plots",
    )