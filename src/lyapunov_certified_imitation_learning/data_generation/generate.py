import numpy as np
from typing import Optional
from acados_template import AcadosOcpSolver

from .mpc_solve import solve_mpc_closed_loop
from .mpc_data import MPCDataset

def generate_mpc_dataset(
    solver: AcadosOcpSolver,
    n_samples: int,
    x0_lower_bound: np.ndarray,
    x0_upper_bound: np.ndarray,
    N_sim: int,
    break_on_infeasible: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
    bound_type: str = "absolute"
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
        In 'percentage' mode this is the single percentage array (0-1) used to shrink the solver's
        state bounds toward their midpoint.
    x0_upper_bound : np.ndarray
        Upper bound for initial state sampling (shape: (nx,)).
        Ignored when bound_type is 'percentage'.
    N_sim : int
        Number of simulation steps per trajectory.
    break_on_infeasible : bool
        Whether to stop simulation if the solver fails.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        If True, prints progress.
    bound_type : str
        Type of bounds: 'absolute' (default) or 'percentage'.
        'percentage' shrinks the solver's lbx/ubx around the midpoint with a single percentage array.

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

    if bound_type == "percentage":
        percentages = x0_lower_bound

        # Basic validation
        if percentages.shape[0] != nx:
            raise ValueError(f"Percentage array must have shape ({nx},). Got {percentages.shape}.")
        if np.any(percentages <= 0) or np.any(percentages > 1):
            raise ValueError("Percentages must be in the interval (0, 1].")

        # Pull state bounds from the solver
        full_lbx = np.full(nx, -np.inf)
        full_ubx = np.full(nx, np.inf)

        c = solver.acados_ocp.constraints

        if hasattr(c, "idxbx") and np.size(c.idxbx) > 0:
            idx = c.idxbx
            if hasattr(c, "lbx") and c.lbx is not None:
                full_lbx[idx] = c.lbx
            if hasattr(c, "ubx") and c.ubx is not None:
                full_ubx[idx] = c.ubx

        if np.any(~np.isfinite(full_lbx)) or np.any(~np.isfinite(full_ubx)):
            raise ValueError("Percentage mode requires finite lbx/ubx for all states.")

        # Shrink bounds symmetrically around the midpoint using the provided percentages
        mid = 0.5 * (full_lbx + full_ubx)
        half_range = 0.5 * (full_ubx - full_lbx)
        shrink = (1.0 - percentages) * half_range

        sample_lb = mid - (half_range - shrink)
        sample_ub = mid + (half_range - shrink)

        if np.any(sample_lb >= sample_ub):
            raise ValueError("Computed sampling bounds are invalid (lower >= upper). Check percentages and solver bounds.")

    elif bound_type == "absolute":
        sample_lb = x0_lower_bound
        sample_ub = x0_upper_bound

    else:
        raise ValueError(f"Unknown bound_type: {bound_type}. Use 'absolute' or 'percentage'.")

    for i in range(n_samples):
        # Sample random initial state
        x0 = np.random.uniform(sample_lb, sample_ub)

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