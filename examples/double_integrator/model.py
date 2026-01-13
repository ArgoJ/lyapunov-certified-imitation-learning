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
    T: float = 1.0, 
    N: int = 20
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
    T : float
        Prediction horizon length in seconds.
    N : int
        Number of control intervals.

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
    dt = T / N
    A_d, B_d = lcil_utils.linalg.c2d(A_c, B_c, dt)

    if P is None:
        P = solve_discrete_are(A_d, B_d, Q, R)

    ocp = AcadosOcp()
    ocp.model = get_model(A_c, B_c)

    # Solver options
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = T
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"

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
    ocp.constraints.idxbu = np.array([0])

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
                    [0, 0]])
    B_c = np.array([[0],
                    [1]])

    # Cost matrices
    Q = np.diag([1.0, 1.0])
    R = np.diag([0.1])

    # Create OCP solver
    solver, info = get_ocp_solver(A_c, B_c, Q, R)

    print("OCP solver created.")
    print("Discrete A matrix:\n", info["A_d"])
    print("Discrete B matrix:\n", info["B_d"])
    print("Terminal cost P matrix:\n", info["P"])

    generator = MPCDataGenerator(
        solver=solver, 
        x0_bounds=np.array([[-8.0, -5.0], [8.0, 5.0]]),
        N_sim=50, 
        verbose=True,
        reset_solver=True,
    )
    dataset = generator.generate(n_samples=1000)
    dataset.validate()
    dataset.save("double_integrator_mpc_dataset.hdf5", mode="w")

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
        plot_3d=True,
        limits=[[-12, 12], [-8, 8]],
        html_path="double_integrator_lyapunov_landscape.html",
    )