import numpy as np
import torch as th
import torch.nn as nn
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from dataclasses import dataclass

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
    dt : float
        Sampling time step for discretization, by default 0.01
    """
    m_cart: float = 1.0
    m_pole: float = 0.1
    length: float = 0.8
    gravity: float = 9.81
    damping: float = 1.0
    dt: float = 0.05


def discrete_inverted_pendulum_on_cart_matrices(
        config: PendulumOnCartConfig
) -> tuple[np.ndarray, np.ndarray]:    
    """Discretize the linearized inverted pendulum on cart dynamics 
    around the upright (`s=1`) or down (`s=-1`) equilibrium 
    with optional damping and sign flip.

    P

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Discrete-time state transition matrix (Ad) and control input matrix (Bd)
    """
    m_c = config.m_cart
    m_p = config.m_pole
    l = config.length
    g = config.gravity
    d = config.damping
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

    ad, bd, _, _, _ = cont2discrete(
        (ac, bc, np.eye(4, dtype=np.float64), np.zeros((4, 1), dtype=np.float64)),
        config.dt,
    )
    return ad, bd


def riccati_gain_and_value_matrix(
    sys_cfg: PendulumOnCartConfig,
    q: np.ndarray | None = None,
    r: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the discrete-time LQR gain matrix K and value matrix P 
    for the linearized inverted pendulum on cart dynamics.

    Parameters
    ----------
    sys_cfg : PendulumOnCartConfig
        Configuration for the inverted pendulum on cart system
    q : np.ndarray | None, optional
        State cost matrix, by default None
    r : np.ndarray | None, optional
        Control cost matrix, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        LQR gain matrix K and value matrix P for the discretized inverted pendulum on cart
    """
    ad, bd = discrete_inverted_pendulum_on_cart_matrices(sys_cfg)
    if q is None:
        q = np.diag([5.0, 1.0, 20.0, 50.0]).astype(np.float64)
    if r is None:
        r = np.array([[2.0]], dtype=np.float64)

    p = solve_discrete_are(ad, bd, q, r)
    k = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
    return k, p


class RiccatiPolicy(nn.Module):
    def __init__(self, k_gain: np.ndarray, max_action: float = 10.0):
        super().__init__()
        self.register_buffer("k_gain", th.as_tensor(k_gain, dtype=th.float32))
        self.register_buffer("max_action", th.as_tensor(max_action, dtype=th.float32))

    def forward(self, x: th.Tensor) -> th.Tensor:
        action = -(x @ self.k_gain.transpose(0, 1))
        return th.clamp(action, min=-self.max_action, max=self.max_action)


class NonlinearInvertedPendulumOnCartDynamics(nn.Module):
    def __init__(self, cfg: PendulumOnCartConfig):
        super().__init__()
        self.cfg = cfg

    def _continuous_dynamics(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        p = x[:, 0]
        p_dot = x[:, 1]
        theta = x[:, 2]
        theta_dot = x[:, 3]
        force = u[:, 0]

        m_c = self.cfg.m_cart
        m_p = self.cfg.m_pole
        l = self.cfg.length
        g = self.cfg.gravity
        d = self.cfg.damping

        sin_theta = th.sin(theta)
        cos_theta = th.cos(theta)
        total_mass = m_c + m_p

        effective_force = force - d * p_dot

        denom = total_mass - m_p * cos_theta * cos_theta # always positive for m_c > 0 and m_p > 0

        p_ddot = (
            effective_force
            + m_p * l * theta_dot.pow(2) * sin_theta
            - m_p * g * sin_theta * cos_theta
        ) / denom

        theta_ddot = (
            effective_force * cos_theta
            + m_p * l * theta_dot.pow(2) * sin_theta * cos_theta
            + total_mass * g * sin_theta
        ) / (l * denom)

        return th.stack([p_dot, p_ddot, theta_dot, theta_ddot], dim=1)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        x_dot = self._continuous_dynamics(x, u)
        return x + self.cfg.dt * x_dot


class RiccatiQuadraticLyapunov(nn.Module):
    def __init__(self, p_matrix: np.ndarray):
        super().__init__()
        p_sym = 0.5 * (p_matrix + p_matrix.T)
        self.register_buffer("p_matrix", th.as_tensor(p_sym, dtype=th.float32))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return ((x @ self.p_matrix) * x).sum(dim=1, keepdim=True)