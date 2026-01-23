# %% General Imports
import numpy as np
from scipy.linalg import solve_discrete_are, block_diag
from casadi import SX
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from typing import Optional, Tuple

import lyapunov_certified_imitation_learning.utils as lcil_utils



# %% Model Definition
def get_model(A: np.ndarray, B: np.ndarray) -> AcadosModel:
    """Create and return an `AcadosModel` for the linear system.
    
    Parameters
    ----------
    A : np.ndarray
        State matrix of the discrete-time double integrator.
    B : np.ndarray
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
    
    A_sx = SX(A)
    B_sx = SX(B)
    
    # Continuous dynamics
    x_dot = A_sx @ x + B_sx @ u

    model = AcadosModel()
    model.name = "double_integrator"
    model.x = x
    model.u = u
    model.f_expl_expr = x_dot
    model.disc_dyn_expr = None

    return model


# %% OCP Solver Definition
def get_ocp_solver(
    A_c: np.ndarray, 
    B_c: np.ndarray, 
    Q: np.ndarray, 
    R: np.ndarray,
    P: Optional[np.ndarray] = None,
    dt: float = 0.1, 
    N: int = 20,
    tol: float = 1e-8,
    terminal_mode: str = "regional",
    bounds_scale: float = 10.0,
    terminal_box_halfwidth: float = 1.0,
) -> Tuple[AcadosOcpSolver, dict]:
    """Create an acados OCP solver for a continuous-time linear system.

    Parameters
    ----------
    A_c, B_c : np.ndarray
        Continuous system matrices (dot(x) = Ax + Bu).
    Q, R : np.ndarray
        Stage cost matrices (x'Qx + u'Ru).
    P : np.ndarray, optional
        Terminal cost matrix (x_N' P x_N). If None, calculated via DARE on discretized system.
    dt : float
        Sampling time in seconds.
    N : int
        Number of control intervals.
    tol : float
        Solver tolerances for the QP solver.

    Returns
    -------
    solver : AcadosOcpSolver
        Constructed acados OCP solver.
    info : dict
        Useful information about the problem (A_d, B_d, P used).
    """
    nx = A_c.shape[0]
    nu = B_c.shape[1]

    ocp = AcadosOcp()
    ocp.model = get_model(A_c, B_c)

    # Calculate DARE
    A_d, B_d = lcil_utils.linalg.lin_c2d_rk4(A_c, B_c, dt, num_steps=1)

    if P is None and terminal_mode == "regional":
        P = solve_discrete_are(A_d, B_d, Q, R)

    # Solver options
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = dt * N
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.qp_solver_tol_stat = tol  # Gradienten-Check
    ocp.solver_options.qp_solver_tol_eq   = tol  # Equality constraints
    ocp.solver_options.qp_solver_tol_ineq = tol  # Inequality constraints
    ocp.solver_options.qp_solver_tol_comp = tol  # Complementarity

    # Erhöhe zur Sicherheit die maximalen Iterationen, falls er länger braucht
    ocp.solver_options.qp_solver_iter_max = 100

    # Cost setup
    ocp.cost.cost_type = "LINEAR_LS"

    W = block_diag(Q, R)
    ocp.cost.W = W
    ocp.cost.Vx = np.vstack((np.eye(nx), np.zeros((nu, nx))))
    ocp.cost.Vu = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
    ocp.cost.yref = np.zeros((nx + nu,))

    # Terminal cost / ingredients
    ocp.cost.cost_type_e = "LINEAR_LS"
    if terminal_mode == "regional":
        ocp.cost.W_e = P
    else:
        # For the "no terminal" scheme (and also for pure equilibrium terminal constraints),
        # keep terminal weight at zero. Downstream code treats this as "no terminal cost".
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
    elif terminal_mode == "equilibrium":
        # Exact equilibrium terminal constraint x(N) = 0.
        ocp.constraints.lbx_e = np.zeros((nx,))
        ocp.constraints.ubx_e = np.zeros((nx,))
        ocp.constraints.idxbx_e = np.arange(nx)
    else:
        # No terminal bounds.
        pass

    solver = AcadosOcpSolver(ocp, json_file=f"{ocp.model.name}_ocp.json")

    info = {
        "A_d": A_d,
        "B_d": B_d,
        "P": P,
        "terminal_mode": terminal_mode,
        "bounds_scale": bounds_scale,
    }

    return solver, info


def get_basic_double_integrator_ocp_solver() -> Tuple[AcadosOcpSolver, dict]:
    """Get a basic double integrator OCP solver with default parameters.

    Returns
    -------
    solver : AcadosOcpSolver
        Configured acados OCP solver for the double integrator.
    info : dict
        Useful information about the problem (A_d, B_d, P used).
    """
    A_c = np.array([[0, 1],
                    [0, 0]])
    B_c = np.array([[0],
                    [1]])
    Q = np.diag([1.0, 1.0])
    R = np.diag([0.1])

    solver, info = get_ocp_solver(
        A_c=A_c,
        B_c=B_c,
        Q=Q,
        R=R,
        dt=0.1,
        N=20,
        terminal_mode="regional",
        bounds_scale=10.0,
        terminal_box_halfwidth=1.0,
    )

    return solver, info