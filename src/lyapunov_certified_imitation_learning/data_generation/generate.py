import numpy as np
from typing import Optional, Tuple
from acados_template import AcadosOcpSolver
from tqdm import tqdm

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
        
        self.sample_lb = None
        self.sample_ub = None
        
        if seed is not None:
            np.random.seed(seed)
            
        self.extract_solver_config()
        
    
    def extract_constraints(self, nx: int, nu: int, nx_e: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        full_bu = np.vstack((np.full(nu, -np.inf), np.full(nu, np.inf)))
        full_bx = np.vstack((np.full(nx, -np.inf), np.full(nx, np.inf)))
        full_bx_e = np.vstack((np.full(nx_e, -np.inf), np.full(nx_e, np.inf)))

        c = self.solver.acados_ocp.constraints
        
        if hasattr(c, "idxbu") and np.size(c.idxbu) > 0:
            idx = c.idxbu
            if hasattr(c, "lbu") and c.lbu is not None:
                full_bu[0, idx] = c.lbu
            if hasattr(c, "ubu") and c.ubu is not None:
                full_bu[1, idx] = c.ubu

        # State constraints
        if hasattr(c, "idxbx") and np.size(c.idxbx) > 0:
            idx = c.idxbx
            if hasattr(c, "lbx") and c.lbx is not None:
                full_bx[0, idx] = c.lbx
            if hasattr(c, "ubx") and c.ubx is not None:
                full_bx[1, idx] = c.ubx

        # Terminal state constraints
        if hasattr(c, "idxbx_e") and np.size(c.idxbx_e) > 0:
            idx = c.idxbx_e
            if hasattr(c, "lbx_e") and c.lbx_e is not None:
                full_bx_e[0, idx] = c.lbx_e
            if hasattr(c, "ubx_e") and c.ubx_e is not None:
                full_bx_e[1, idx] = c.ubx_e
        
        return full_bu, full_bx, full_bx_e

    def extract_cost_weights(self, nx: int, nu: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract (Q, R, Qf) from the acados OCP cost matrices.

        For the common LINEAR_LS setup with y = [x; u] and W = block_diag(Q, R),
        this returns the intended Q and R blocks.
        """
        cost = self.solver.acados_ocp.cost

        W = np.asarray(cost.W)
        if W.ndim == 2 and W.shape == (nx + nu, nx + nu):
            Q = W[:nx, :nx]
            R = W[nx:, nx:]
        else:
            Q = W
            R = W

        Qf = None
        if hasattr(cost, "W_e") and cost.W_e is not None:
            Qf = np.asarray(cost.W_e)

        return Q, R, Qf

    def extract_goal_state(self, nx: int) -> Optional[np.ndarray]:
        """Try to extract a goal state from yref_e/yref."""
        cost = self.solver.acados_ocp.cost
        if hasattr(cost, "yref_e") and cost.yref_e is not None:
            yref_e = np.asarray(cost.yref_e).reshape(-1)
            if yref_e.size >= nx:
                return yref_e[:nx].copy()
        if hasattr(cost, "yref") and cost.yref is not None:
            yref = np.asarray(cost.yref).reshape(-1)
            if yref.size >= nx:
                return yref[:nx].copy()
        return None
        
        
    def extract_solver_config(self) -> None:
        nx = self.solver.acados_ocp.dims.nx
        nu = self.solver.acados_ocp.dims.nu
        nx_e = self.solver.acados_ocp.dims.nbx_e
        
        N_horizon = self.solver.acados_ocp.solver_options.N_horizon
        tf = float(self.solver.acados_ocp.solver_options.tf)
        dt = tf / int(N_horizon)

        full_bu, full_bx, full_bx_e = self.extract_constraints(nx, nu, nx_e)

        Q, R, Qf = self.extract_cost_weights(nx, nu)
        goal_state = self.extract_goal_state(nx)

        self.mpc_config = MPCConfig(
            Q=Q,
            R=R,
            Qf=Qf,
            dt=dt,
            N=int(N_horizon),
            T_sim=self.T_sim,
            state_bounds=full_bx,
            input_bounds=full_bu,
            terminal_state_bounds=full_bx_e,
            goal_state=goal_state
        )
        
        if self.bound_type == "percentage":
            percentages = self.x0_bounds

            # Basic validation
            if percentages.shape[0] != nx:
                raise ValueError(f"Percentage array must have shape ({nx},). Got {percentages.shape}.")
            if np.any(percentages <= 0) or np.any(percentages > 1):
                raise ValueError("Percentages must be in the interval (0, 1].")

            if np.any(~np.isfinite(full_bx)):
                raise ValueError("Percentage mode requires finite lbx/ubx for all states.")

            self.sample_lb, self.sample_ub = self._calculate_percentage_bounds(full_bx[0], full_bx[1], percentages)

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

                if self.reset_solver:
                    self.solver.reset()

                # TODO: add an epsilon band around the x_target
                mpc_data = solve_mpc_closed_loop(
                    solver=self.solver,
                    x0=x0,
                    N_sim=self.T_sim,
                    dt=self.mpc_config.dt,
                    config=self.mpc_config,
                    break_on_infeasible=self.break_on_infeasible
                )

                dataset.add(mpc_data)
        finally:
            if tqdm_handler:
                PackageLogger.restore_handlers(DEFAULT_MODULE_NAME, tqdm_handler, restored_handlers)

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