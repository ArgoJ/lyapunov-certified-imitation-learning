import numpy as np

from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are, block_diag
from casadi import SX, vertcat
from mpc_datagen import mdg_linalg
from sys_cfg import PendulumOnCartConfig

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


def _linearized_inverted_pendulum_on_cart_matrices(
    cfg: PendulumOnCartConfig
) -> tuple[np.ndarray, np.ndarray]:    
    """Linearized inverted pendulum on cart dynamics around the upright equilibrium
    with optional damping.

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

    State is ``x = [cart_pos, cart_vel, pole_angle, pole_ang_vel]`` and input is
    cart force ``u``.
    """
    nx = 4
    nu = 1

    x = SX.sym("x", nx)
    u = SX.sym("u", nu)
    p_global = SX.sym("p_global", 1)

    cart_pos = x[0]
    cart_vel = x[1]
    theta = x[2]
    theta_dot = x[3]
    force = u[0]
    del cart_pos  # symbolic state is kept for readability.

    m_c = cfg.m_cart
    m_p = cfg.m_pole
    l = cfg.length
    g = cfg.gravity
    d = cfg.damping

    sin_theta = SX.sin(theta)
    cos_theta = SX.cos(theta)
    total_mass = m_c + m_p

    effective_force = force - d * cart_vel

    denom = total_mass - m_p * cos_theta * cos_theta # always positive for m_c > 0 and m_p > 0
    theta2_sin_ml = m_p * l * theta_dot * theta_dot * sin_theta

    p_ddot = (
        effective_force
        + theta2_sin_ml
        - m_p * g * sin_theta * cos_theta
    ) / denom

    theta_ddot = (
        effective_force * cos_theta
        + theta2_sin_ml * cos_theta
        + total_mass * g * sin_theta
    ) / (l * denom)

    f_expl = vertcat(cart_vel, p_ddot, theta_dot, theta_ddot)

    model = AcadosModel()
    model.name = "inv_pend_cart"
    model.x = x
    model.u = u
    model.p_global = p_global
    model.f_expl_expr = f_expl
    return model


# %% OCP Solver Definition
def get_ocp_solver(
    Q: NDArray, 
    R: NDArray,
    dt: float = 0.1, 
    N: int = 20,
    tol: float = 1e-8,
    terminal_mode: str = "regional",
    sys_cfg: PendulumOnCartConfig = PendulumOnCartConfig(),
) -> tuple[AcadosOcpSolver, dict]:
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

    ocp = AcadosOcp()
    ocp.model = get_model(cfg=sys_cfg)
    ocp.p_global_values = np.zeros((1,))

    A_c, B_c = _linearized_inverted_pendulum_on_cart_matrices(cfg=sys_cfg)
    A_d, B_d = mdg_linalg.lin_c2d_rk4(A_c, B_c, dt, num_steps=1)
    P = solve_discrete_are(A_d, B_d, Q, R)

    # Solver options
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = dt * N
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.qp_solver_tol_stat = tol  # Gradienten-Check
    ocp.solver_options.qp_solver_tol_eq   = tol  # Equality constraints
    ocp.solver_options.qp_solver_tol_ineq = tol  # Inequality constraints
    ocp.solver_options.qp_solver_tol_comp = tol  # Complementarity
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
    if terminal_mode in ("regional", "lqr"):
        ocp.cost.W_e = P
    else:
        # For "no terminal" scheme and for equilibrium terminal constraints,
        ocp.cost.W_e = np.zeros((nx, nx))
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.zeros((nx,))

    # Constraints
    ocp.constraints.x0 = np.zeros((nx,))

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
    
    solver = AcadosOcpSolver(ocp, json_file=f"{ocp.model.name}_ocp.json", verbose=False)
    return solver, info
