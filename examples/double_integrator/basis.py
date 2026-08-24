import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are
from mpc_datagen import mdg_utils


A_c = np.array([[0.0, 1.0],
                [0.0, 0.0]],
                dtype=np.float64)
B_c = np.array([[0.0],
                [1.0]],
                dtype=np.float64)
Q = np.diag([15.0, 1.0]).astype(np.float64)
R = np.diag([0.1]).astype(np.float64)

def compute_discrete_double_integrator(dt: float) -> tuple[NDArray, NDArray]:
    a_d, b_d = mdg_utils.lin_c2d_rk4(A_c, B_c, float(dt), num_steps=1)
    return a_d, b_d

def compute_riccati_value_matrix(dt: float) -> NDArray:
    a_d, b_d = compute_discrete_double_integrator(dt)
    return solve_discrete_are(a_d, b_d, Q, R)


__all__ = [
    "A_c",
    "B_c",
    "Q",
    "R",
    "compute_discrete_double_integrator",
    "compute_riccati_value_matrix",
]