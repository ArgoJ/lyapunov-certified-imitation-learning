import numpy as np
from typing import Optional
from acados_template import AcadosOcpSolver

from .mpc_solve import solve_mpc_closed_loop
from .mpc_data import MPCDataset

def generate_random_dataset(
    solver: AcadosOcpSolver,
    n_samples: int,
    x0_lower_bound: np.ndarray,
    x0_upper_bound: np.ndarray,
    N_sim: int,
    break_on_infeasible: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True
) -> MPCDataset:
    """
    Generates a dataset of MPC closed-loop trajectories starting from random initial states.

    Parameters
    ----------
    solver : AcadosOcpSolver
        The initialized Acados OCP solver instance.
    n_samples : int
        Number of trajectories to generate.
    x0_lower_bound : np.ndarray
        Lower bound for initial state sampling (shape: (nx,)).
    x0_upper_bound : np.ndarray
        Upper bound for initial state sampling (shape: (nx,)).
    N_sim : int
        Number of simulation steps per trajectory.
    break_on_infeasible : bool
        Whether to stop simulation if the solver fails.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        If True, prints progress.

    Returns
    -------
    MPCDataset
        A dataset containing the generated trajectories.
    """
    if seed is not None:
        np.random.seed(seed)

    dataset = MPCDataset()
    nx = solver.acados_ocp.dims.nx

    # Validate bounds dimensions
    if x0_lower_bound.shape[0] != nx or x0_upper_bound.shape[0] != nx:
        raise ValueError(f"Bounds must have shape ({nx},). Got {x0_lower_bound.shape} and {x0_upper_bound.shape}.")

    for i in range(n_samples):
        # Sample random initial state
        x0 = np.random.uniform(x0_lower_bound, x0_upper_bound)

        if verbose:
            print(f"Generating sample {i+1}/{n_samples} with x0={x0}")

        # Run closed-loop simulation
        config = {
            "generation_id": i,
            "x0_sampled": x0.tolist(),
            "sim_length": N_sim
        }
        
        mpc_data = solve_mpc_closed_loop(
            solver=solver,
            x0=x0,
            N_sim=N_sim,
            config=config,
            break_on_infeasible=break_on_infeasible
        )

        dataset.add(mpc_data)

    return dataset
