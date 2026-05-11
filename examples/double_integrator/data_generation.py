import argparse
import numpy as np

from datetime import datetime
from pathlib import Path

import mpc_datagen.linalg as mdg_linalg
import mpc_datagen.plots as mdg_plots
from mpc_datagen import *
from mpc_datagen.verification import (
    StabilityVerifier,
    VerificationRender,
    ROAVerifier,
)

try:
    from . import get_batch_ocp_solver
except ImportError:
    from acados_ocp import get_batch_ocp_solver


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MPC imitation datasets for the double integrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        default=40,
        help="Simulation horizon length (number of MPC steps).",
    )
    parser.add_argument(
        "--bound-scale",
        type=float,
        default=10.0,
        help="Scale for box constraints on states and inputs in the OCP.",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="results/double_integrator/data",
        help="Base output path for generated datasets and plots.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Initial acados batch solver size for faster MPC data generation.",
    )
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    # Continuous-time double integrator matrices (standard)
    A_c = np.array([[0, 1],
                    [0, 0]])
    B_c = np.array([[0],
                    [1]])

    # Cost matrices
    Q = np.diag([15.0, 1.0])
    R = np.diag([0.1])

    iso = datetime.now().strftime('%Y%m%d_%H%M%S').replace(" ", "_").replace(":", "-")
    base_path = Path(args.base_path) / iso

    T_sim = args.t_sim
    n_samples = args.n_samples
    bounds_scale = args.bound_scale
    batch_size = args.batch_size
    terminal_box_halfwidth = 2.0
    N = 20
    terminal_mode = "regional"

    dt = 0.1
    solver, info = get_batch_ocp_solver(
        A_c, B_c,
        Q, R,
        dt=dt,
        N=N,
        tol=1e-8,
        batch_size=batch_size,
        terminal_mode=terminal_mode,
        bounds_scale=bounds_scale,
        terminal_box_halfwidth=terminal_box_halfwidth,
    )

    constraints = solver.ocp_solvers[0].acados_ocp.constraints if hasattr(solver, "ocp_solvers") else solver.acados_ocp.constraints

    sampler = UniqueBoundedSampler(
        bounds=np.array([constraints.lbx, constraints.ubx]),
        min_dist=np.array([1e-3, 1e-4]),
        seed=4597525,
    )
    eps_cfg = EpsBandConfig(
        eps_band=np.array([0.1, 1e-3]), 
        eps_consecutive=3
    )
    generator = MPCDataGenerator(
        solver=solver,
        T_sim=T_sim,
        sampler=sampler,
        xeps_cfg=eps_cfg,
        solver_regen_interval=20,
    )
    dataset = generator.generate(n_samples=n_samples, only_feasible=True)
    dataset.validate()
    dataset.save(f"{base_path}/double_integrator_{terminal_mode}_N{N}_data.hdf5")

    veri_stats = StabilityVerifier.verify(dataset, alpha_required=1e-4)
    VerificationRender(veri_stats).render()

    if terminal_mode == "regional":
        P = info["P"]
        lyap_fun = lambda x: 0.5 * mdg_linalg.weighted_quadratic_norm(x, P)
        roa_lyap_fun = lambda x: mdg_linalg.weighted_quadratic_norm(x, P)
        roa_cert = ROAVerifier(dataset[0].config)
        roa_bounds, c_min = roa_cert.roa_bounds()
    else:
        lyap_fun = None
        roa_lyap_fun = None
        roa_bounds = None
        c_min = None

    mdg_plots.all(
        dataset=dataset[:min(150, n_samples)],
        state_labels=["x", "v"],
        control_labels=["a"],
        time_bound=T_sim * dt,
        plot_3d=True,
        plot_predictions=False,
        alpha=1.0, # veri_stats.details["asym_stab_report"].min_alpha,
        use_optimal_v=False,
        lyapunov_func=lyap_fun,
        lyap_use_dataset_v=True,
        roa_lyapunov_func=roa_lyap_fun,
        c_level=c_min,
        roa_bounds=roa_bounds,
        base_path=f"{base_path}/double_integrator_{terminal_mode}_N{N}",
    )


if __name__ == "__main__":
    main()