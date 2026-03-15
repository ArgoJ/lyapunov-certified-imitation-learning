import numpy as np

from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are, block_diag
from casadi import SX, vertcat
from dataclasses import dataclass

from mpc_datagen import mdg_linalg

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


@dataclass
class PendulumOnCartConfig:
    """
    Configuration class for the inverted pendulum on cart system.

    Parameters
    ----------
    m_cart : float, optional
        Mass of the cart, by default 1.0
    m_pole : float, optional
        Mass of the pendulum, by default 0.1
    length : float, optional
        Length of the pendulum, by default 0.5
    gravity : float, optional
        Gravitational acceleration, by default 9.81
    damping : float, optional
        Damping coefficient, by default 1.0
    """
    m_cart: float = 1.0
    m_pole: float = 0.1
    length: float = 0.5
    gravity: float = 9.81
    damping: float = 1.0


def _linearized_inverted_pendulum_on_cart_matrices(
    cfg: PendulumOnCartConfig
) -> tuple[np.ndarray, np.ndarray]:    
    """Discretize the linearized inverted pendulum on cart dynamics 
    around the upright (`s=1`) or down (`s=-1`) equilibrium 
    with optional damping and sign flip.

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

    p_ddot = (
        effective_force
        + m_p * l * theta_dot * theta_dot * sin_theta
        - m_p * g * sin_theta * cos_theta
    ) / denom

    theta_ddot = (
        effective_force * cos_theta
        + m_p * l * theta_dot * theta_dot * sin_theta * cos_theta
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
    bounds_scale: float = 10.0,
    terminal_box_halfwidth: float = 1.0,
    sys_cfg: PendulumOnCartConfig = PendulumOnCartConfig(),
) -> tuple[AcadosOcpSolver, dict]:
    """Create an acados OCP solver for inverted pendulum on cart.

    Parameters
    ----------
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
    solver : AcadosOcpSolver
        Constructed acados OCP solver.
    info : dict
        Useful information about the problem (A_d, B_d, P used).
    """
    nx = 4
    nu = 1

    ocp = AcadosOcp()
    ocp.model = get_model(cfg=sys_cfg)
    ocp.solver_options.integrator_type = "ERK"

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
        "A_c": A_c,
        "B_c": B_c,
        "A_d": A_d,
        "B_d": B_d,
        "P": P,
        "terminal_mode": terminal_mode,
        "bounds_scale": bounds_scale,
    }
    
    solver = AcadosOcpSolver(ocp, json_file=f"{ocp.model.name}_ocp.json")
    return solver, info