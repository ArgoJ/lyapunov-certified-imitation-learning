import numpy as np
from typing import Optional, Tuple
from acados_template import AcadosOcpSolver
from tqdm import tqdm
from dataclasses import replace

from .extractor import MPCConfigExtractor
from .mpc_solve import solve_mpc_closed_loop
from .mpc_data import MPCDataset, MPCConfig
from ..utils.package_logger import PackageLogger, DEFAULT_MODULE_NAME

class MPCDataGenerator:
    """
    Generator for MPC closed-loop datasets.
    """
    def __init__(
        self,
        solver: AcadosOcpSolver,
        x0_bounds: np.ndarray,
        T_sim: int,
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
        x0_bounds : np.ndarray
            Bounds for initial state sampling (shape: (2, nx)) (lower_bounds, upper_bounds).
            In 'percentage' mode this is the single (shape: (nx,))  percentage array (0-1) used to shrink the solver's
            state bounds toward their midpoint.
        T_sim : int
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
        self.break_on_infeasible = break_on_infeasible
        self.verbose = verbose
        self.reset_solver = reset_solver
        self.bound_type = bound_type
        self.x0_bounds = x0_bounds
        self.T_sim = T_sim
        
        self.mpc_config = MPCConfigExtractor.get_cfg(self.solver)
        self.mpc_config.T_sim = T_sim
        
        self.sample_lb = None
        self.sample_ub = None
        
        if seed is not None:
            np.random.seed(seed)
            
        self.calc_sample_bounds()
        
    def calc_sample_bounds(self) -> None:
        nx = self.mpc_config.nx
        
        if self.bound_type == "percentage":
            percentages = self.x0_bounds

            # Basic validation
            if percentages.shape[0] != nx:
                raise ValueError(f"Percentage array must have shape ({nx},). Got {percentages.shape}.")
            if np.any(percentages <= 0) or np.any(percentages > 1):
                raise ValueError("Percentages must be in the interval (0, 1].")

            if np.any(~np.isfinite(self.mpc_config.state_bounds)):
                raise ValueError("Percentage mode requires finite lbx/ubx for all states.")

            self.sample_lb, self.sample_ub = self._calculate_percentage_bounds(
                self.mpc_config.state_bounds[0], self.mpc_config.state_bounds[1], percentages)

        elif self.bound_type == "absolute":
            if self.x0_bounds.shape != (2, nx):
                raise ValueError(f"Bounds must have shape (2, {nx}) for absolute mode. Got {self.x0_bounds.shape}.")
            self.sample_lb = self.x0_bounds[0]
            self.sample_ub = self.x0_bounds[1]
        else:
            raise ValueError(f"Unknown bound_type: {self.bound_type}. Use 'absolute' or 'percentage'.")
        
            
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
            for _ in iterator:
                x0 = np.random.uniform(self.sample_lb, self.sample_ub)
                temp_cfg = replace(self.mpc_config, x0=x0)

                if self.reset_solver:
                    self.solver.reset()

                # TODO: add an epsilon band around the x_target
                mpc_data = solve_mpc_closed_loop(
                    solver=self.solver,
                    cfg=temp_cfg,
                    break_on_infeasible=self.break_on_infeasible
                )

                dataset.add(mpc_data)
        finally:
            if tqdm_handler:
                PackageLogger.restore_handlers(DEFAULT_MODULE_NAME, tqdm_handler, restored_handlers)

        return dataset

    @staticmethod
    def _calculate_percentage_bounds(lbx: np.ndarray, ubx: np.ndarray, percentages: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Shrink bounds symmetrically around the midpoint using the provided percentages"""
        mid = 0.5 * (lbx + ubx)
        half_range = 0.5 * (ubx - lbx)
        shrink = (1.0 - percentages) * half_range

        sample_lb = mid - (half_range - shrink)
        sample_ub = mid + (half_range - shrink)

        if np.any(sample_lb >= sample_ub):
            raise ValueError("Computed sampling bounds are invalid (lower >= upper). Check percentages and solver bounds.")
        
        return sample_lb, sample_ub