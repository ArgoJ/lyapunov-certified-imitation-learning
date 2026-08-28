import argparse
import logging
import numpy as np

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mpc_datagen import *
from mpc_datagen.verification import ROAVerifier, StabilityVerifier, VerificationRender
from lcil.utils import ArgumentParserConfig, GridSearchHelper, config_field

from . import DATA_DIR, Q, R, compute_riccati_value_matrix, get_batch_ocp_solver

__logger__ = logging.getLogger("mpc_datagen")


@dataclass(frozen=True)
class DataGenerationScriptConfig(ArgumentParserConfig):
    n_samples: int = config_field(default=2000, help="Number of trajectories to generate.")
    t_sim: int = config_field(default=200, help="Simulation horizon length (number of MPC steps).")
    base_path: str = config_field(default=str(DATA_DIR), help="Base output path for generated datasets and plots.")
    debug: bool = config_field(default=False, help="Enable debug logging during dataset generation.")
    only_feasible: bool = config_field(default=False, help="Save only feasible trajectories.")


def parse_cli_args() -> GridSearchHelper[DataGenerationScriptConfig]:
    parser = argparse.ArgumentParser(
        description="Generate MPC imitation datasets for the inverted pendulum on cart.",
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

def main() -> None:
    sweep = parse_cli_args()

    for run in sweep:
        script_config = run.config
        if script_config.debug:
            __logger__.setLevel(logging.DEBUG)

        # Sample in a tighter local region around the equilibrium to improve feasibility
        sample_percentages = np.array([1.0, 1.0, 0.333, 1.0], dtype=float)
        sample_bias = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

        base_path = run.output_dir.resolve()
        T_sim = int(script_config.t_sim)
        n_samples = int(script_config.n_samples)
        N = 40
        dt = 0.05

        solver, _ = get_batch_ocp_solver(
            Q,
            R,
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
            eps_consecutive=3,
        )
        generator = MPCDataGenerator(
            solver=solver,
            T_sim=T_sim,
            sampler=sampler,
            xeps_cfg=eps_cfg,
            solver_regen_interval=20,
        )
        dataset = generator.generate(n_samples=n_samples, only_feasible=True)
        dataset.validate(tol_stability=0.1)

        dataset_path = base_path / f"cartpole_N{N}_data.hdf5"
        dataset.save(dataset_path)

        veri_stats = StabilityVerifier.verify(dataset, alpha_required=1e-4)
        VerificationRender(veri_stats).render()

        p_matrix = compute_riccati_value_matrix(dt)
        lyap_fun = lambda x: 0.5 * mdg_utils.weighted_quadratic_norm(x, p_matrix)
        roa_cert = ROAVerifier(dataset[0].config)
        roa_bounds, c_min = roa_cert.roa_bounds()

        alpha = 1.0 if veri_stats.details.get("asym_stab_report", None) is None else veri_stats.details["asym_stab_report"].min_alpha
        mdg_plt.all(
            dataset=dataset[:min(150, n_samples)],
            state_labels=["$x$", "$v$", "$\\theta$", "$\\dot{\\theta}$"],
            control_labels=["$a$"],
            time_bound=T_sim * dt,
            plot_3d=False,
            plot_predictions=False,
            alpha=alpha,
            use_optimal_v=False,
            lyapunov_func=lyap_fun,
            lyap_use_dataset_v=True,
            c_level=c_min,
            base_path=str(base_path / f"cartpole_N{N}_plots"),
        )


if __name__ == "__main__":
    main()