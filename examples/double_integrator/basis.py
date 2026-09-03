import numpy as np
import torch as th

from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are
from lcil.lyapunov_learning import check_kappa


A_c = np.array([[0.0, 1.0],
                [0.0, 0.0]],
                dtype=np.float64)
B_c = np.array([[0.0],
                [1.0]],
                dtype=np.float64)
Q = np.diag([15.0, 1.0]).astype(np.float64)
R = np.diag([0.1]).astype(np.float64)


def compute_discrete_double_integrator(dt: float) -> tuple[NDArray, NDArray]:
    dt_val = float(dt)
    a_d = np.array([[1.0, dt_val], [0.0, 1.0]], dtype=np.float64)
    b_d = np.array([[0.0], [dt_val]], dtype=np.float64)
    return a_d, b_d


def compute_riccati_value_matrix(dt: float, kappa: float | None = None) -> NDArray:
    a_d, b_d = compute_discrete_double_integrator(dt)
    p = solve_discrete_are(a_d, b_d, Q, R)
    if kappa is not None:
        k_gain = np.linalg.solve(R + b_d.T @ p @ b_d, b_d.T @ p @ a_d)
        check_kappa(
            kappa=kappa,
            riccati_p=th.as_tensor(p),
            q_matrix=th.as_tensor(Q),
            r_matrix=th.as_tensor(R),
            k_gain=th.as_tensor(k_gain),
        )
    return p


__all__ = [
    "A_c",
    "B_c",
    "Q",
    "R",
    "compute_discrete_double_integrator",
    "compute_riccati_value_matrix",
]