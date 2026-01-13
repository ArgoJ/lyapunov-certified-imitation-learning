import numpy as np
from typing import Optional, Tuple
from acados_template import AcadosOcpSolver
from tqdm import tqdm

from .mpc_solve import solve_mpc_closed_loop
from .mpc_data import MPCDataset
from ..utils.package_logger import PackageLogger

class MPCDataGenerator:
    """
    Generator for MPC closed-loop datasets.
    """
    def __init__(
        self,
        solver: AcadosOcpSolver,
        x0_lower_bound: np.ndarray,
        x0_upper_bound: np.ndarray,
        N_sim: int,
        break_on_infeasible: bool = True,
        seed: Optional[int] = None,
        verbose: bool = True,
        bound_type: str = "absolute",
        reset_solver: bool = False,
    ):
        """
        Initializes the MPC Data Generator.

        Parameters
        ----------
        solver : AcadosOcpSolver
            The initialized Acados OCP solver instance.
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
        reset_solver : bool
            If True, resets the solver states to zero before each simulation.
        """
        self.solver = solver
        self.N_sim = N_sim
        self.break_on_infeasible = break_on_infeasible
        self.verbose = verbose
        self.reset_solver = reset_solver
        
        if seed is not None:
            np.random.seed(seed)
            
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

            self.sample_lb, self.sample_ub = self._calculate_percentage_bounds(full_lbx, full_ubx, percentages)

        elif bound_type == "absolute":
            self.sample_lb = x0_lower_bound
            self.sample_ub = x0_upper_bound

        else:
            raise ValueError(f"Unknown bound_type: {bound_type}. Use 'absolute' or 'percentage'.")
            
    def generate(self, n_samples: int) -> MPCDataset:
        """
        Generates a dataset of MPC closed-loop trajectories starting from random initial states.

        Parameters
        ----------
        n_samples : int
             Number of trajectories to generate.

        Returns
        -------
        MPCDataset
            A dataset containing the generated trajectories.
        """
        dataset = MPCDataset()
        
        # Configure Tqdm handler for logging if verbose is enabled
        tqdm_handler = None
        restored_handlers = []
        if self.verbose:
            tqdm_handler, restored_handlers = PackageLogger.add_tqdm_handler()
        
        iterator = range(n_samples)
        if self.verbose:
            iterator = tqdm(iterator, desc="Generating Trajectories")

        try:
            for i in iterator:
                x0 = np.random.uniform(self.sample_lb, self.sample_ub)

                if self.reset_solver:
                    self.solver.reset()

                # Run closed-loop simulation
                config = {
                    "generation_id": i,
                    "x0_sampled": x0.tolist(),
                    "sim_length": self.N_sim
                }

                mpc_data = solve_mpc_closed_loop(
                    solver=self.solver,
                    x0=x0,
                    N_sim=self.N_sim,
                    config=config,
                    break_on_infeasible=self.break_on_infeasible
                )

                dataset.add(mpc_data)
        finally:
            if tqdm_handler:
                PackageLogger.restore_handlers("lyapunov_certified_imitation_learning", tqdm_handler, restored_handlers)

        return dataset

    def _calculate_percentage_bounds(self, full_lbx: np.ndarray, full_ubx: np.ndarray, percentages: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Shrink bounds symmetrically around the midpoint using the provided percentages"""
        mid = 0.5 * (full_lbx + full_ubx)
        half_range = 0.5 * (full_ubx - full_lbx)
        shrink = (1.0 - percentages) * half_range

        sample_lb = mid - (half_range - shrink)
        sample_ub = mid + (half_range - shrink)

        if np.any(sample_lb >= sample_ub):
            raise ValueError("Computed sampling bounds are invalid (lower >= upper). Check percentages and solver bounds.")
        
        return sample_lb, sample_ub
