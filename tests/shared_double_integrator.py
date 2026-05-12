from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn
from scipy.linalg import solve_discrete_are


@dataclass
class DoubleIntegratorConfig:
    """Configuration for the discrete-time double integrator."""

    dt: float = 0.1


def discrete_double_integrator_matrices(
    config: DoubleIntegratorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact zero-order-hold discrete-time system matrices.

    Parameters
    ----------
    config : DoubleIntegratorConfig
        Double integrator configuration containing the sampling time.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Discrete-time state transition matrix and control input matrix.
    """
    dt = float(config.dt)
    ad = np.array(
        [
            [1.0, dt],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    bd = np.array(
        [
            [0.5 * dt * dt],
            [dt],
        ],
        dtype=np.float64,
    )
    return ad, bd


def riccati_gain_and_value_matrix(
    sys_cfg: DoubleIntegratorConfig,
    q: np.ndarray | None = None,
    r: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the discrete-time LQR gain and value matrix.

    Parameters
    ----------
    sys_cfg : DoubleIntegratorConfig
        Double integrator configuration.
    q : np.ndarray | None, optional
        State cost matrix, by default diag([1.0, 0.5]).
    r : np.ndarray | None, optional
        Control cost matrix, by default [[0.25]].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        LQR gain matrix K and quadratic value matrix P.
    """
    ad, bd = discrete_double_integrator_matrices(sys_cfg)
    if q is None:
        q = np.diag([1.0, 0.5]).astype(np.float64)
    if r is None:
        r = np.array([[0.25]], dtype=np.float64)

    p = solve_discrete_are(ad, bd, q, r)
    k = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
    return k, p


class RiccatiPolicy(nn.Module):
    """Linear discrete-time LQR policy."""

    def __init__(self, k_gain: np.ndarray):
        super().__init__()
        self.register_buffer("k_gain", th.as_tensor(k_gain, dtype=th.float32))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return -(x @ self.k_gain.transpose(0, 1))


class DoubleIntegratorDynamics(nn.Module):
    """Exact discrete-time double-integrator dynamics."""

    def __init__(self, cfg: DoubleIntegratorConfig):
        super().__init__()
        ad, bd = discrete_double_integrator_matrices(cfg)
        self.register_buffer("ad", th.as_tensor(ad, dtype=th.float32))
        self.register_buffer("bd", th.as_tensor(bd, dtype=th.float32))

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return x @ self.ad.transpose(0, 1) + u @ self.bd.transpose(0, 1)


class RiccatiQuadraticLyapunov(nn.Module):
    """Quadratic Lyapunov model induced by the LQR value matrix."""

    def __init__(self, p_matrix: np.ndarray):
        super().__init__()
        p_sym = 0.5 * (p_matrix + p_matrix.T)
        self.register_buffer("p_matrix", th.as_tensor(p_sym, dtype=th.float32))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return ((x @ self.p_matrix) * x).sum(dim=1, keepdim=True)