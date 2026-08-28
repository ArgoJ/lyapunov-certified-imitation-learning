import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are

from mpc_datagen import mdg_utils

from .sys_cfg import PendulumOnCartConfig

Q = np.diag([1e2, 1e1, 1e2, 1e-2]).astype(np.float64)
R = np.diag([5e-1]).astype(np.float64)



def linearized_inverted_pendulum_on_cart_matrices(
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
            [0.0, d / (m_c * l), ((m_c + m_p) * g) / (m_c * l), 0.0],
        ],
        dtype=np.float64,
    )
    bc = np.array(
        [
            [0.0],
            [1.0 / m_c],
            [0.0],
            [-1.0 / (m_c * l)],
        ],
        dtype=np.float64,
    )

    return ac, bc


def compute_discrete_cartpole(
    dt: float,
    sys_cfg: PendulumOnCartConfig | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cfg = PendulumOnCartConfig() if sys_cfg is None else sys_cfg
    a_c, b_c = linearized_inverted_pendulum_on_cart_matrices(cfg=cfg)
    a_d, b_d = mdg_utils.lin_c2d_rk4(a_c, b_c, float(dt), num_steps=1)
    return a_d, b_d


def compute_riccati_value_matrix(
    dt: float,
    sys_cfg: PendulumOnCartConfig | None = None,
    q: NDArray[np.float64] | None = None,
    r: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    q_matrix = Q if q is None else np.asarray(q, dtype=np.float64)
    r_matrix = R if r is None else np.asarray(r, dtype=np.float64)
    a_d, b_d = compute_discrete_cartpole(dt=dt, sys_cfg=sys_cfg)
    return solve_discrete_are(a_d, b_d, q_matrix, r_matrix)


__all__ = [
    "PendulumOnCartConfig",
    "Q",
    "R",
    "compute_discrete_cartpole",
    "compute_riccati_value_matrix",
]