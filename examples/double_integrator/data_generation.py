import argparse
import logging
import numpy as np

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mpc_datagen.linalg as mdg_linalg
from mpc_datagen import *
from mpc_datagen.verification import (
    StabilityVerifier,
    VerificationRender,
    ROAVerifier,
)
from lcil.utils import ArgumentParserConfig, GridSearchHelper, config_field
from . import get_batch_ocp_solver, A_c, B_c, Q, R, DATA_DIR

__logger__ = logging.getLogger("mpc_datagen")


@dataclass(frozen=True)
class DataGenerationScriptConfig(ArgumentParserConfig):
    n_samples: int = config_field(default=20000, help="Number of trajectories to generate.")
    t_sim: int = config_field(default=40, help="Simulation horizon length (number of MPC steps).")
    base_path: str = config_field(default=str(DATA_DIR), help="Base output path for generated datasets and plots.")
    debug: bool = config_field(default=False, help="Enable debug logging during dataset generation.")
    only_feasible: bool = config_field(default=True, help="Save only feasible trajectories.")
    bound_scale: float = config_field(default=10.0, help="Scale for box constraints on states and inputs in the OCP.")
    batch_size: int = config_field(default=128, help="Initial acados batch solver size for faster MPC data generation.")


def parse_cli_args() -> GridSearchHelper[DataGenerationScriptConfig]:
    parser = argparse.ArgumentParser(
        description="Generate MPC imitation datasets for the double integrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    script_defaults = DataGenerationScriptConfig()
    script_defaults.add_to_argparse(parser)

    args = parser.parse_args()
    sweep = GridSearchHelper.from_namespace(
        args,
        script_defaults,
        sweep_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    sweep.set_output_root(Path(sweep.configs[0].base_path))
    return sweep


def main() -> None:
    sweep = parse_cli_args()

    for run in sweep:
        script_config = run.config
        if script_config.debug:
            __logger__.setLevel(logging.DEBUG)

        base_path = run.output_dir.resolve()
        T_sim = script_config.t_sim
        n_samples = script_config.n_samples
        bounds_scale = script_config.bound_scale
        batch_size = script_config.batch_size
        only_feasible = script_config.only_feasible
        terminal_box_halfwidth = 2.0
        N = 20
        terminal_mode = "regional"

        dt = 0.1
        solver, info = get_batch_ocp_solver(
            A_c,
            B_c,
            Q,
            R,
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
            eps_consecutive=3,
        )
        generator = MPCDataGenerator(
            solver=solver,
            T_sim=T_sim,
            sampler=sampler,
            xeps_cfg=eps_cfg,
            solver_regen_interval=20,
        )
        dataset = generator.generate(n_samples=n_samples, only_feasible=only_feasible)
        dataset.validate()

        dataset_path = base_path / f"double_integrator_{terminal_mode}_N{N}_data.hdf5"
        dataset.save(dataset_path)

        veri_stats = StabilityVerifier.verify(dataset, alpha_required=1e-4)
        VerificationRender(veri_stats).render()

        if terminal_mode == "regional":
            P = info["P"]
            lyap_fun = lambda x: 0.5 * mdg_linalg.weighted_quadratic_norm(x, P)
            roa_cert = ROAVerifier(dataset[0].config)
            c_min = roa_cert.compute_min_c()
        else:
            lyap_fun = None
            c_min = None

        alpha = 1.0 if veri_stats.details.get("asym_stab_report", None) is None else veri_stats.details["asym_stab_report"].min_alpha
        mdg_plt.all(
            dataset=dataset[:min(150, n_samples)],
            state_labels=["$x$", "$v$"],
            control_labels=["$a$"],
            time_bound=T_sim * dt,
            plot_3d=True,
            plot_predictions=False,
            alpha=alpha,
            use_optimal_v=False,
            lyapunov_func=lyap_fun,
            lyap_use_dataset_v=True,
            c_level=c_min,
            base_path=str(base_path / f"double_integrator_{terminal_mode}_N{N}"),
        )


if __name__ == "__main__":
    main()