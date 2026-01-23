import time
import logging
from typing import Dict, Optional, Any
import numpy as np
from acados_template import AcadosOcpSolver, AcadosSimSolver

from .mpc_data import MPCData, MPCTrajectory, MPCMeta, MPCConfig
from .extractor import MPCConfigExtractor

__logger__ = logging.getLogger(__name__)


def solve_mpc_closed_loop(
    solver: AcadosOcpSolver,
    integrator: Optional[AcadosSimSolver] = None,
    cfg: Optional[MPCConfig] = None,
    break_on_infeasible: bool = True
) -> MPCData:
    """
    Simulates a closed-loop MPC run using an Acados solver.

    Parameters
    ----------
    solver : AcadosOcpSolver
        The initialized Acados OCP solver.
    integrator : AcadosSimSolver, optional
        Acados integrator for accurate simulation steps.
    cfg : MPCConfig, optional
        Configuration dictionary to store in MPCData.
    break_on_infeasible : bool
        If True, stops simulation if the solver returns a non-zero status.

    Returns
    -------
    MPCData
        The collected data from the closed-loop run.
    """
    
    # Initialize Trajectory container with NaNs
    if cfg is None:
        cfg = MPCConfigExtractor.get_cfg(solver)
    traj = MPCTrajectory.init(T_sim=cfg.T_sim, N=cfg.N, nx=cfg.nx, nu=cfg.nu, dt=cfg.dt)
    
    # Set initial state
    traj.states[0, :] = cfg.x0
    
    solve_times = []
    status_codes = []
    
    current_x = cfg.x0.copy()
    is_feasible_run = True

    sim_start_time = time.time()

    for i in range(cfg.T_sim):
        solver.set(0, "lbx", current_x)
        solver.set(0, "ubx", current_x)
        
        status = solver.solve()
        status_codes.append(status)
        
        if status != 0:
            if break_on_infeasible:
                __logger__.warning(f"Solver failed at step {i} with status {status}. Stopping.")
                is_feasible_run = False
                break
        
        solve_times.append(solver.get_stats("time_tot"))
                
        # Retrieve Predictions
        pred_x = np.zeros((cfg.N + 1, cfg.nx))
        for k in range(cfg.N + 1):
            pred_x[k, :] = solver.get(k, "x")
            
        pred_u = np.zeros((cfg.N, cfg.nu))
        for k in range(cfg.N):
            pred_u[k, :] = solver.get(k, "u")
            
        # Store predictions
        traj.solved_states[i, :, :] = pred_x
        traj.solved_inputs[i, :, :] = pred_u
        traj.cost[i] = solver.get_cost()
        
        # Apply Control
        u_applied = pred_u[0, :].flatten()
        traj.inputs[i, :] = u_applied
        
        # Simulate System
        if integrator is not None:
            integrator.set("x", current_x)
            integrator.set("u", u_applied)
            
            status_sim = integrator.solve()
            if status_sim != 0:
                __logger__.error(f"Integrator failed at step {i} with status {status_sim}")
            
            current_x = integrator.get("x")
        else:
            current_x = pred_x[1, :].flatten()
        
        traj.states[i+1, :] = current_x

    sim_end_time = time.time()
    
    # Update feasibility
    traj.feasible = is_feasible_run and (status_codes[-1] == 0 if status_codes else False)

    # Construct Meta Information
    meta = MPCMeta(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sim_start_time)),
        solve_time_mean=float(np.mean(solve_times)) if solve_times else 0.0,
        solve_time_max=float(np.max(solve_times)) if solve_times else 0.0,
        solve_time_total=float(np.sum(solve_times)) if solve_times else 0.0,
        sim_duration_wall=sim_end_time - sim_start_time,
        steps_simulated=len(status_codes),
        status_codes=status_codes
    )
        
    return MPCData(
        config=cfg,
        trajectory=traj,
        meta=meta
    )
