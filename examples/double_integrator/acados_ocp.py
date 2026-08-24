import os

import numpy as np

from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are, block_diag
from casadi import SX

from mpc_datagen import add_temp_folder, mdg_utils

from acados_template import AcadosModel, AcadosOcp, AcadosOcpBatchSolver, AcadosOcpSolver


# %% Model Definition
def get_model(A: NDArray, B: NDArray, dynamics_type: str = "continuous") -> AcadosModel:
    """Create and return an `AcadosModel` for the linear system.
    
    Parameters
    ----------
    A : NDArray
        State matrix of the discrete-time double integrator.
    B : NDArray
        Input matrix of the discrete-time double integrator.

    Returns
    -------
    model : AcadosModel
        Configured AcadosModel object.
    """
    nx = A.shape[0]
    nu = B.shape[1]

    # states
    x = SX.sym("x", nx)

    # control
    u = SX.sym("u", nu)
    p_global = SX.sym("p_global", 1)
    
    A_sx = SX(A)
    B_sx = SX(B)
    
    # Continuous dynamics
    dynamics_expr = A_sx @ x + B_sx @ u

    model = AcadosModel()
    model.name = "double_integrator"
    model.x = x
    model.u = u
    model.p_global = p_global

    if dynamics_type == "continuous":
        model.f_expl_expr = dynamics_expr
        model.disc_dyn_expr = None
    elif dynamics_type == "discrete":
        model.f_expl_expr = None
        model.disc_dyn_expr = dynamics_expr
    else:
        raise ValueError(f"Unsupported dynamics_type '{dynamics_type}'. Use 'continuous' or 'discrete'.")

    return model


# %% OCP Solver Definition
def get_ocp(
    A_c: NDArray, 
    B_c: NDArray, 
    Q: NDArray, 
    R: NDArray,
    P: NDArray | None = None,
    dt: float = 0.1, 
    N: int = 20,
    tol: float = 1e-8,
    terminal_mode: str = "regional",
    bounds_scale: float = 10.0,
    dynamics_type: str = "continuous",
    terminal_box_halfwidth: float = 1.0,
) -> tuple[AcadosOcp, dict]:
    """Create an acados OCP for a continuous-time linear system.

    Parameters
    ----------
    A_c, B_c : NDArray
        Continuous system matrices (dot(x) = Ax + Bu).
    Q, R : NDArray
        Stage cost matrices (x'Qx + u'Ru).
    P : NDArray, optional
        Terminal cost matrix (x_N' P x_N). If None, calculated via DARE on discretized system.
    dt : float
        Sampling time in seconds.
    N : int
        Number of control intervals.
    tol : float
        Solver tolerances for the QP solver.

    Returns
    -------
    ocp : AcadosOcp
        Constructed acados OCP.
    info : dict
        Useful information about the problem (A_d, B_d, P used).
    """
    nx = A_c.shape[0]
    nu = B_c.shape[1]

    A_d, B_d = mdg_utils.lin_c2d_rk4(A_c, B_c, dt, num_steps=1)

    ocp = AcadosOcp()
    if dynamics_type == "continuous":
        ocp.model = get_model(A_c, B_c, dynamics_type="continuous")
        ocp.solver_options.integrator_type = 'ERK'
    elif dynamics_type == "discrete":
        ocp.model = get_model(A_d, B_d, dynamics_type="discrete")
        ocp.solver_options.integrator_type = 'DISCRETE'
    else:
        raise ValueError(f"Unsupported dynamics_type '{dynamics_type}'. Use 'continuous' or 'discrete'.")

    ocp.p_global_values = np.zeros((1,))

    if P is None and terminal_mode == "regional":
        P = solve_discrete_are(A_d, B_d, Q, R)

    # Solver options
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = dt * N
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.qp_solver_tol_stat = tol  # Gradienten-Check
    ocp.solver_options.qp_solver_tol_eq   = tol  # Equality constraints
    ocp.solver_options.qp_solver_tol_ineq = tol  # Inequality constraints
    ocp.solver_options.qp_solver_tol_comp = tol  # Complementarity

    # Erhöhe zur Sicherheit die maximalen Iterationen, falls er länger braucht
    ocp.solver_options.qp_solver_iter_max = 100

    # Cost setup
    ocp.cost.cost_type_0 = "LINEAR_LS"
    ocp.cost.cost_type = "LINEAR_LS"

    W = block_diag(Q, R)
    ocp.cost.W_0 = W
    ocp.cost.W = W
    ocp.cost.Vx_0 = np.vstack((np.eye(nx), np.zeros((nu, nx))))
    ocp.cost.Vu_0 = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
    ocp.cost.yref_0 = np.zeros((nx + nu,))
    ocp.cost.Vx = np.vstack((np.eye(nx), np.zeros((nu, nx))))
    ocp.cost.Vu = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
    ocp.cost.yref = np.zeros((nx + nu,))

    # Terminal cost / ingredients
    ocp.cost.cost_type_e = "LINEAR_LS"
    if terminal_mode == "regional":
        ocp.cost.W_e = P
    else:
        # For "no terminal" scheme and for equilibrium terminal constraints,
        ocp.cost.W_e = np.zeros((nx, nx))
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.zeros((nx,))

    # Constraints
    ocp.constraints.x0 = np.zeros((nx,))

    # (Large) box constraints, used for feasibility and for the LQR terminal certificate sizing.
    ocp.constraints.lbu = -bounds_scale * np.ones((nu,))
    ocp.constraints.ubu = bounds_scale * np.ones((nu,))
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.lbx = -bounds_scale * np.ones((nx,))
    ocp.constraints.ubx = bounds_scale * np.ones((nx,))
    ocp.constraints.idxbx = np.arange(nx)

    if terminal_mode == "regional":
        # A small terminal box around the equilibrium (proxy for a local terminal set X_f).
        hw = float(terminal_box_halfwidth)
        ocp.constraints.lbx_e = -hw * np.ones((nx,))
        ocp.constraints.ubx_e = hw * np.ones((nx,))
        ocp.constraints.idxbx_e = np.arange(nx)
    if terminal_mode == "equilibrium":
        # Exact equilibrium terminal constraint x(N) = 0.
        ocp.constraints.lbx_e = np.zeros((nx,))
        ocp.constraints.ubx_e = np.zeros((nx,))
        ocp.constraints.idxbx_e = np.arange(nx)

    info = {
        "A_d": A_d,
        "B_d": B_d,
        "P": P,
        "terminal_mode": terminal_mode,
        "bounds_scale": bounds_scale,
    }

    return ocp, info


def get_ocp_solver(
    A_c: NDArray,
    B_c: NDArray,
    Q: NDArray,
    R: NDArray,
    P: NDArray | None = None,
    dt: float = 0.1,
    N: int = 20,
    tol: float = 1e-8,
    terminal_mode: str = "regional",
    bounds_scale: float = 10.0,
    dynamics_type: str = "continuous",
    terminal_box_halfwidth: float = 1.0,
    use_temp_dir: bool = True,
) -> tuple[AcadosOcpSolver, dict]:
    """Convenience function to directly get the OCP solver instance."""
    ocp, info = get_ocp(
        A_c=A_c,
        B_c=B_c,
        Q=Q,
        R=R,
        P=P,
        dt=dt,
        N=N,
        tol=tol,
        terminal_mode=terminal_mode,
        bounds_scale=bounds_scale,
        dynamics_type=dynamics_type,
        terminal_box_halfwidth=terminal_box_halfwidth,
    )
    file_name = f"{ocp.model.name}_ocp.json"
    if use_temp_dir:
        ocp, file_name = add_temp_folder(ocp, file_name)

    solver = AcadosOcpSolver(ocp, json_file=file_name, verbose=False)
    return solver, info


def get_batch_ocp_solver(
    A_c: NDArray,
    B_c: NDArray,
    Q: NDArray,
    R: NDArray,
    P: NDArray | None = None,
    dt: float = 0.1,
    N: int = 20,
    tol: float = 1e-8,
    batch_size: int = 100,
    terminal_mode: str = "regional",
    bounds_scale: float = 10.0,
    dynamics_type: str = "continuous",
    terminal_box_halfwidth: float = 1.0,
    use_temp_dir: bool = True,
) -> tuple[AcadosOcpBatchSolver, dict]:
    """Convenience function to directly get a batch OCP solver instance."""
    ocp, info = get_ocp(
        A_c=A_c,
        B_c=B_c,
        Q=Q,
        R=R,
        P=P,
        dt=dt,
        N=N,
        tol=tol,
        terminal_mode=terminal_mode,
        bounds_scale=bounds_scale,
        dynamics_type=dynamics_type,
        terminal_box_halfwidth=terminal_box_halfwidth,
    )

    num_threads = min(int(batch_size), os.cpu_count() or 1)
    ocp.solver_options.with_batch_functionality = True
    ocp.solver_options.num_threads_in_batch_solve = num_threads

    file_name = f"{ocp.model.name}_batch_ocp.json"
    if use_temp_dir:
        ocp, file_name = add_temp_folder(ocp, file_name)

    batch_solver = AcadosOcpBatchSolver(
        ocp,
        N_batch_init=int(batch_size),
        num_threads_in_batch_solve=num_threads,
        json_file=file_name,
        verbose=False,
    )
    return batch_solver, info