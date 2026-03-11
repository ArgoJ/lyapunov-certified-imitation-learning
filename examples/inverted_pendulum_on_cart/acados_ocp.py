import numpy as np

from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are, block_diag
from casadi import SX

from mpc_datagen import mdg_linalg

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


# %% Model Definition
def get_model(
    m_cart: float = 1.0,
    m_pole: float = 0.1,
    length: float = 0.5,
    gravity: float = 9.81,
    dynamics_type: str = "continuous",
    ad: NDArray | None = None,
    bd: NDArray | None = None,
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

    sin_theta = SX.sin(theta)
    cos_theta = SX.cos(theta)
    denom = m_cart + m_pole * sin_theta * sin_theta

    x_ddot = (
        force + m_pole * sin_theta * (length * theta_dot * theta_dot + gravity * cos_theta)
    ) / denom
    theta_ddot = (
        -force * cos_theta
        - m_pole * length * theta_dot * theta_dot * cos_theta * sin_theta
        - (m_cart + m_pole) * gravity * sin_theta
    ) / (length * denom)

    f_expl = SX.vertcat(cart_vel, x_ddot, theta_dot, theta_ddot)

    model = AcadosModel()
    model.name = "inv_pend_cart"
    model.x = x
    model.u = u
    model.p_global = p_global

    if dynamics_type == "continuous":
        model.f_expl_expr = f_expl
        model.disc_dyn_expr = None
    elif dynamics_type == "discrete":
        if ad is None or bd is None:
            raise ValueError("ad and bd must be provided for discrete dynamics_type.")
        ad_sx = SX(ad)
        bd_sx = SX(bd)
        model.f_expl_expr = None
        model.disc_dyn_expr = ad_sx @ x + bd_sx @ u
    else:
        raise ValueError(
            f"Unsupported dynamics_type '{dynamics_type}'. Use 'continuous' or 'discrete'."
        )

    return model


def _upright_linearized_matrices(
    m_cart: float,
    m_pole: float,
    length: float,
    gravity: float,
) -> tuple[NDArray, NDArray]:
    """Linearization around upright equilibrium ``theta=0``."""
    a_c = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -(m_pole * gravity) / m_cart, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -((m_cart + m_pole) * gravity) / (m_cart * length), 0.0],
        ],
        dtype=float,
    )
    b_c = np.array(
        [
            [0.0],
            [1.0 / m_cart],
            [0.0],
            [-1.0 / (m_cart * length)],
        ],
        dtype=float,
    )
    return a_c, b_c


# %% OCP Solver Definition
def get_ocp_solver(
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
    m_cart: float = 1.0,
    m_pole: float = 0.1,
    length: float = 0.5,
    gravity: float = 9.81,
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

    A_c, B_c = _upright_linearized_matrices(
        m_cart=m_cart,
        m_pole=m_pole,
        length=length,
        gravity=gravity,
    )
    A_d, B_d = mdg_linalg.lin_c2d_rk4(A_c, B_c, dt, num_steps=1)

    ocp = AcadosOcp()
    if dynamics_type == "continuous":
        ocp.model = get_model(
            m_cart=m_cart,
            m_pole=m_pole,
            length=length,
            gravity=gravity,
            dynamics_type="continuous",
        )
        ocp.solver_options.integrator_type = "ERK"
    elif dynamics_type == "discrete":
        ocp.model = get_model(
            m_cart=m_cart,
            m_pole=m_pole,
            length=length,
            gravity=gravity,
            dynamics_type="discrete",
            ad=A_d,
            bd=B_d,
        )
        ocp.solver_options.integrator_type = "DISCRETE"
    else:
        raise ValueError(f"Unsupported dynamics_type '{dynamics_type}'. Use 'continuous' or 'discrete'.")

    ocp.p_global_values = np.zeros((1,))

    if P is None and terminal_mode == "regional":
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