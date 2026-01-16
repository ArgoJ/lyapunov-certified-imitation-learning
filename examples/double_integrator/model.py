# %% [markdown] 
# # Double Integrator Example


# %% General Imports
import numpy as np
from scipy.linalg import solve_discrete_are, block_diag
from casadi import SX
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from typing import Optional, Tuple

import lyapunov_certified_imitation_learning.utils as lcil_utils
from lyapunov_certified_imitation_learning.data_generation import MPCDataGenerator
from lyapunov_certified_imitation_learning.lyapunov_verification import verify_mpc_asymptotic_stability



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
    tol: float = 1e-8
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

    # Calculate DARE
    A_d, B_d = lcil_utils.linalg.c2d_rk4(A_c, B_c, dt)
    # TODO: try to find a function in acados that returns discrete Matrices

    if P is None:
        P = solve_discrete_are(A_d, B_d, Q, R)

    ocp = AcadosOcp()
    ocp.model = get_model(A_c, B_c)

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

    # Terminal cost
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W_e = P
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.zeros((nx,))

    # Constraints
    ocp.constraints.x0 = np.zeros((nx,))

    ocp.constraints.lbu = np.array([-5.0])
    ocp.constraints.ubu = np.array([5.0])
    ocp.constraints.idxbu = np.arange(nu)
    
    ocp.constraints.lbx = np.array([-10.0, -10.0])
    ocp.constraints.ubx = np.array([10.0, 10.0])
    ocp.constraints.idxbx = np.arange(nx)
    
    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx_e = np.arange(nx)

    solver = AcadosOcpSolver(ocp, json_file=f"{ocp.model.name}_ocp.json")

    info = {
        "A_d": A_d,
        "B_d": B_d,
        "P": P,
    }

    return solver, info


# %%  
if __name__ == "__main__":
    # Continuous-time double integrator matrices
    A_c = np.array([[0, 1],
                    [0.2, 0.1]])
    B_c = np.array([[0],
                    [1]])

    # Cost matrices
    Q = np.diag([15.0, 1.0])
    R = np.diag([0.1])

    # Create OCP solver
    solver, info = get_ocp_solver(A_c, B_c, Q, R, N=15)

    print("OCP solver created.")
    print("Discrete A matrix:\n", info["A_d"])
    print("Discrete B matrix:\n", info["B_d"])
    print("Terminal cost P matrix:\n", info["P"])

    generator = MPCDataGenerator(
        solver=solver, 
        x0_bounds=np.array([[-7.0, -5.0], [7.0, 5.0]]),
        T_sim=50, 
        verbose=True,
        reset_solver=True,
    )
    dataset = generator.generate(n_samples=50)
    dataset.validate()
    dataset.save("double_integrator_mpc_dataset.hdf5")
    

    lcil_utils.plot.mpc_trajectories(
        dataset=dataset,
        state_labels=["Position", "Velocity"],
        control_labels=["Acceleration"],
        plot_predictions=True,
        html_path="double_integrator_mpc_trajectories.html",
    )

    lyap = lambda x: x.T @ info["P"] @ x

    lcil_utils.plot.lyapunov(
        dataset=dataset, 
        lyapunov_func=lyap,
        plot_3d=False,
        limits=[[-12, 12], [-8, 8]],
        html_path="double_integrator_lyapunov_landscape.html",
    )
    
    is_stable = verify_mpc_asymptotic_stability(
        dataset=dataset,
        Q=Q,
        R=R,
        mode="hybrid",
    )