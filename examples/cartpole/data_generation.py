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
from pathlib import Path
from datetime import datetime

import mpc_datagen.linalg as mdg_linalg
import mpc_datagen.plots as mdg_plots
from mpc_datagen import *
from mpc_datagen.verification import (
    StabilityVerifier,
    VerificationRender,
    ROAVerifier,
)

from acados_ocp import get_batch_ocp_solver, get_ocp_solver
from pkg_logger import get_package_logger

__logger__ = get_package_logger("mpc_datagen")

def _setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MPC imitation datasets for the inverted pendulum on cart."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
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

def _normalize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def normalize_dataset(dataset: MPCDataset) -> MPCDataset:
    """
    Normalize the angle component of the dataset to be within [-pi, pi].
    """
    for entry in dataset:
        entry.trajectory.states[:, 2] = _normalize_angle(entry.trajectory.states[:, 2])
    return dataset

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    if args.debug:
        __logger__.setLevel(logging.DEBUG)

    # Cost matrices
    Q = np.diag([1e2, 1e1, 1e2, 1e-2])
    R = np.diag([5e-1])

    # Sample in a tighter local region around the equilibrium to improve feasibility
    sample_percentages = np.array([1.0, 1.0, 0.333, 1.0], dtype=float)
    sample_bias = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    base_path = Path(args.base_path) / datetime.now().strftime('%Y%m%d_%H%M%S')

    T_sim = args.t_sim
    n_samples = args.n_samples
    N = 40

    dt = 0.05
    solver, info = get_batch_ocp_solver(
        Q, R,
        dt=dt,
        N=N,
        terminal_mode="regional",
    )

    constraints = solver.ocp_solvers[0].acados_ocp.constraints if hasattr(solver, "ocp_solvers") else solver.acados_ocp.constraints
    bounds = np.vstack((constraints.lbx, constraints.ubx))

    sampler = UniqueBoundedSampler(
        bounds=bounds,
        bias=sample_bias,
        percentages=sample_percentages,
        min_dist=np.array([1e-2, 1e-3, 1e-3, 1e-3]),
        seed=4597525,
    )
    eps_cfg = EpsBandConfig(
        eps_band=np.array([1e-3, 1e-2, 1e-3, 1e-2]), 
        eps_consecutive=3
    )
    generator = MPCDataGenerator(
        solver=solver,
        T_sim=T_sim,
        sampler=sampler,
        xeps_cfg=eps_cfg,
        solver_regen_interval=20,
        # noise_std=1e-4,
    )
    dataset = generator.generate(n_samples=n_samples, only_feasible=True)
    # dataset = normalize_dataset(dataset)
    dataset.validate(tol_stability=0.1)
    dataset.save(f"{base_path}/inverted_pendulum_on_cart_N{N}_data.hdf5")

    veri_stats = StabilityVerifier.verify(dataset, alpha_required=1e-4)
    VerificationRender(veri_stats).render()

    P = info["P"]
    lyap_fun = lambda x: 0.5 * mdg_linalg.weighted_quadratic_norm(x, P)
    roa_lyap_fun = lambda x: mdg_linalg.weighted_quadratic_norm(x, P)
    roa_cert = ROAVerifier(dataset[0].config)
    roa_bounds, c_min = roa_cert.roa_bounds()

    alpha = 1.0 if veri_stats.details.get("asym_stab_report", None) is None else veri_stats.details["asym_stab_report"].min_alpha
    mdg_plots.all(
        dataset=dataset[:min(150, n_samples)],
        state_labels=["$x$", "$v$", "$\\theta$", "$\\dot{\\theta}$"],
        control_labels=["$a$"],
        time_bound=T_sim * dt,
        plot_3d=False,
        plot_predictions=False,
        alpha=alpha,
        use_optimal_v=False,
        lyapunov_func=lyap_fun,
        # lyap_state_indices=[1, 2],
        lyap_use_dataset_v=True,
        roa_lyapunov_func=roa_lyap_fun,
        c_level=c_min,
        roa_bounds=roa_bounds,
        base_path=f"{base_path}/inverted_pendulum_on_cart_N{N}_plots",
    )


if __name__ == "__main__":
    main()