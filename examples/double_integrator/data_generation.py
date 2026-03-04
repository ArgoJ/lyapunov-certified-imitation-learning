# %% [markdown] 
# # Double Integrator Example


# %% General Imports
import argparse
import numpy as np

import mpc_datagen.linalg as mdg_linalg
import mpc_datagen.plots as mdg_plots
from mpc_datagen import *
from mpc_datagen.verification import (
    StabilityVerifier,
    VerificationRender,
    ROAVerifier,
)

from acados_ocp import get_ocp_solver


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MPC imitation datasets for the double integrator."
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
    return parser


# %%  
if __name__ == "__main__":
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

    base_path = args.base_path

    T_sim = args.t_sim
    n_samples = args.n_samples
    bounds_scale = args.bound_scale
    terminal_box_halfwidth = 2.0

    def run_case(
        name: str,
        terminal_mode: str,
        N: int,
    ) -> None:
        print("\n" + "=" * 80)
        print(f"CASE: {name} (terminal_mode={terminal_mode}, N={N})")
        print("=" * 80)

        dt = 0.1
        solver, info = get_ocp_solver(
            A_c, B_c,
            Q, R,
            dt=dt,
            N=N,
            tol=1e-8,
            terminal_mode=terminal_mode,
            bounds_scale=bounds_scale,
            terminal_box_halfwidth=terminal_box_halfwidth,
        )

        sampler = UniqueBoundedSampler(
            bounds=np.array([solver.acados_ocp.constraints.lbx, solver.acados_ocp.constraints.ubx]),
            min_dist=np.array([1e-2, 1e-3]),
            seed=4597525,
        )
        eps_cfg = EpsBandConfig(
            eps_band=np.array([0.2, 1e-2]), 
            eps_consecutive=3
        )
        generator = MPCDataGenerator(
            solver=solver,
            T_sim=T_sim,
            sampler=sampler,
            reset_solver=True,
            xeps_cfg=eps_cfg,
        )
        dataset = generator.generate(n_samples=n_samples)
        dataset.validate()
        dataset.save(f"{base_path}/double_integrator_{terminal_mode}_N{N}_data.hdf5")

        veri_stats = StabilityVerifier.verify(dataset, solver, alpha_required=1e-4)
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
            lyap_use_optimal_v=True,
            roa_lyapunov_func=roa_lyap_fun,
            c_level=c_min,
            roa_bounds=roa_bounds,
            base_path=f"{base_path}/double_integrator_{terminal_mode}_N{N}",
        )


    # Case 1: regional terminal cost + small terminal set (should pass regional proof + empirical)
    run_case(
        name="Regional terminal ingredients",
        terminal_mode="regional",
        N=20,
    )

    # # Case 2: equilibrium terminal constraint x(N)=0 (sample close so feasibility is easy)
    # run_case(
    #     name="Equilibrium terminal constraint",
    #     terminal_mode="equilibrium",
    #     N=25,
    # )

    # # Case 3: no terminal ingredients (zero terminal weight, no terminal bounds)
    # run_case(
    #     name="No terminal ingredients (Grüne horizon condition)",
    #     terminal_mode="none",
    #     N=70,
    # )
