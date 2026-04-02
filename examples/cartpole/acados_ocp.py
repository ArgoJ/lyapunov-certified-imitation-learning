import numpy as np
import casadi as ca
import os

from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are, block_diag
from mpc_datagen import mdg_linalg, add_temp_folder
from sys_cfg import PendulumOnCartConfig

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosOcpBatchSolver


def _linearized_inverted_pendulum_on_cart_matrices(
    cfg: PendulumOnCartConfig
) -> tuple[np.ndarray, np.ndarray]:    
    """Linearized inverted pendulum on cart dynamics around the upright equilibrium
    with optional damping.

    Parameters
    ----------
    cfg : PendulumOnCartConfig
        System configuration parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Discrete-time state transition matrix (Ad) and control input matrix (Bd)
    """
    m_c = cfg.m_cart
    m_p = cfg.m_pole
    l = cfg.length
    g = cfg.gravity
    d = cfg.damping
    ac = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -d / m_c, -(m_p * g) / m_c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, - d / (m_c * l), ((m_c + m_p) * g) / (m_c * l), 0.0],
        ],
        dtype=np.float64,
    )
    bc = np.array(
        [
            [0.0],
            [1.0 / m_c],
            [0.0],
            [1.0 / (m_c * l)],
        ],
        dtype=np.float64,
    )

    return ac, bc


# %% Model Definition
def get_model(
    cfg: PendulumOnCartConfig,
) -> AcadosModel:
    """Create an acados model for inverted pendulum on cart.

    Parameters
    ----------
    cfg : PendulumOnCartConfig
        System configuration parameters.
    
    Returns
    -------
    AcadosModel
        Acados model object representing the inverted pendulum on cart dynamics.

    Note
    ----
    State is ``x = [cart_pos, cart_vel, pole_angle, pole_ang_vel]`` and input is
    cart force ``u``.
    """
    model_name = 'pendulum'

    # constants
    m_c = cfg.m_cart
    m_p = cfg.m_pole
    l = cfg.length
    g = cfg.gravity
    d = cfg.damping

    # set up states & controls
    x1      = ca.SX.sym('x1')
    theta   = ca.SX.sym('theta')
    v1      = ca.SX.sym('v1')
    dtheta  = ca.SX.sym('dtheta')

    x = ca.vertcat(x1, v1, theta, dtheta)

    F = ca.SX.sym('F')
    u = ca.vertcat(F)

    # parameters
    p = []

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    
    total_mass = m_c + m_p

    effective_force = F - d * v1

    denom = total_mass - m_p * cos_theta * cos_theta # always positive for m_c > 0 and m_p > 0
    theta2_sin_ml = m_p * l * dtheta * dtheta * sin_theta

    p_ddot = (
        effective_force
        - theta2_sin_ml
        + m_p * g * sin_theta * cos_theta
    ) / denom

    theta_ddot = (
        effective_force * cos_theta
        - theta2_sin_ml * cos_theta
        + total_mass * g * sin_theta
    ) / (l * denom)

    f_expl = ca.vertcat(v1, p_ddot, dtheta, theta_ddot)

    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = p
    model.name = model_name

    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$ [N]']
    model.t_label = '$t$ [s]'

    return model


# %% OCP Solver Definition
def get_ocp(
    Q: NDArray, 
    R: NDArray,
    dt: float = 0.05, 
    N: int = 40,
    terminal_mode: str = "regional",
    sys_cfg: PendulumOnCartConfig = PendulumOnCartConfig()
) -> tuple[AcadosOcp, dict]:
    """Create an acados OCP solver for inverted pendulum on cart.

    Parameters
    ----------
    Q, R : NDArray
        Stage cost matrices (x'Qx + u'Ru).
    dt : float
        Sampling time in seconds.
    N : int
        Number of control intervals.
    tol : float
        Solver tolerances for the QP solver.
    terminal_mode : str
        Terminal ingredients mode:
        - "regional" (terminal constraints, DARE cost), 
        - "lqr" (DARE cost only, no terminal constraints),
        - "none" (no terminal constraints, no terminal cost),

    Returns
    -------
    solver : AcadosOcpSolver
        Constructed acados OCP solver.
    info : dict
        Useful information about the problem (A_d, B_d, P used).
    """
    nx = 4
    nu = 1
    ny = nx + nu

    ocp = AcadosOcp()
    ocp.model = get_model(cfg=sys_cfg)

    A_c, B_c = _linearized_inverted_pendulum_on_cart_matrices(cfg=sys_cfg)
    A_d, B_d = mdg_linalg.lin_c2d_rk4(A_c, B_c, dt, num_steps=1)
    P = solve_discrete_are(A_d, B_d, Q, R)

    # Solver options
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = dt * N
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver_cond_N = int(N/4)
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.regularize_method = 'PROJECT'
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.print_level = 0
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.nlp_solver_max_iter = 200


    # Cost setup
    ocp.cost.cost_type_0 = "LINEAR_LS"
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    W = block_diag(Q, R)
    ocp.cost.W_0 = W
    ocp.cost.W = W
    ocp.cost.Vx_0 = np.vstack((np.eye(nx), np.zeros((nu, nx))))
    ocp.cost.Vu_0 = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
    ocp.cost.yref_0 = np.zeros((ny,))
    ocp.cost.Vx = np.vstack((np.eye(nx), np.zeros((nu, nx))))
    ocp.cost.Vu = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
    ocp.cost.yref = np.zeros((ny,))

    # Terminal cost
    if terminal_mode in ("regional", "lqr"):
        ocp.cost.W_e = P
    elif terminal_mode == "stage":
        ocp.cost.W_e = Q
    else:
        ocp.cost.W_e = np.zeros((nx, nx))
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.zeros((nx,))

    # Constraints
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])
    # ocp.remove_x0_elimination()

    # Hardcoded realistic bounds
    F_MAX = 80.0
    X_MAX = 2.0
    V_MAX = 10.0
    THETA_MAX = 3*np.pi
    THETA_DOT_MAX = 10.0

    ocp.constraints.lbu = np.array([-F_MAX])
    ocp.constraints.ubu = np.array([F_MAX])
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.lbx = np.array([-X_MAX, -V_MAX, -THETA_MAX, -THETA_DOT_MAX])
    ocp.constraints.ubx = np.array([X_MAX, V_MAX, THETA_MAX, THETA_DOT_MAX])
    ocp.constraints.idxbx = np.arange(nx)

    if terminal_mode == "regional":
        ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
        ocp.constraints.ubx_e = ocp.constraints.ubx.copy()
        ocp.constraints.idxbx_e = ocp.constraints.idxbx.copy()

    info = {
        "A_c": A_c,
        "B_c": B_c,
        "A_d": A_d,
        "B_d": B_d,
        "P": P,
        "terminal_mode": terminal_mode,
    }
    
    return ocp, info


def get_ocp_solver(
    Q: NDArray, 
    R: NDArray,
    dt: float = 0.05, 
    N: int = 40,
    terminal_mode: str = "regional",
    sys_cfg: PendulumOnCartConfig = PendulumOnCartConfig(),
    use_temp_dir: bool = True,
) -> tuple[AcadosOcpSolver, dict]:
    """Convenience function to directly get the OCP solver instance."""
    ocp, info = get_ocp(Q, R, dt, N, terminal_mode, sys_cfg)
    if use_temp_dir:
        ocp, file_name = add_temp_folder(ocp, f"{ocp.model.name}_ocp.json")

    solver = AcadosOcpSolver(ocp, json_file=file_name, verbose=False)
    return solver, info


def get_batch_ocp_solver(
    Q: NDArray, 
    R: NDArray,
    dt: float = 0.05, 
    N: int = 40,
    batch_size: int = 100,
    terminal_mode: str = "regional",
    sys_cfg: PendulumOnCartConfig = PendulumOnCartConfig(),
    use_temp_dir: bool = True,
) -> tuple[AcadosOcpBatchSolver, dict]:
    ocp, info = get_ocp(Q, R, dt, N, terminal_mode, sys_cfg)
    
    num_threads = min(batch_size, os.cpu_count() or 1)
    # Acados batch solver template requires these attributes
    ocp.solver_options.with_batch_functionality = True
    ocp.solver_options.num_threads_in_batch_solve = num_threads
    if use_temp_dir:
        ocp, file_name = add_temp_folder(ocp, f"{ocp.model.name}_batch_ocp.json")

    batch_solver = AcadosOcpBatchSolver(
        ocp,
        N_batch_init=batch_size,
        num_threads_in_batch_solve=num_threads,
        json_file=file_name,
        verbose=False
    )
    return batch_solver, info