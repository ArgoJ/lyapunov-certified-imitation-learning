import numpy as np
from scipy.linalg import expm
from typing import Tuple

def c2d_exact(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize a continuous-time linear system using Zero-Order Hold (ZOH)."""
    n_x = A.shape[0]
    n_u = B.shape[1]
    
    # Construct block matrix for matrix exponential
    M = np.zeros((n_x + n_u, n_x + n_u))
    M[:n_x, :n_x] = A
    M[:n_x, n_x:] = B
    
    M_exp = expm(M * dt)
    
    Ad = M_exp[:n_x, :n_x]
    Bd = M_exp[:n_x, n_x:]
    
    return Ad, Bd


def c2d_rk4(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Discretize a continuous-time linear system using the RK4 method.
    """
    n_x = A.shape[0]
    I = np.eye(n_x)
    
    # A_discret (Ad)
    A_dt = A * dt
    Ad = I + A_dt + (A_dt @ A_dt) / 2 + (A_dt @ A_dt @ A_dt) / 6 + (A_dt @ A_dt @ A_dt @ A_dt) / 24
    
    # B_discret (Bd)
    # Bd = (I*dt + A*dt^2/2 + A^2*dt^3/6 + A^3*dt^4/24) * B
    # Bd * u = integration over [0,dt] Bu using RK4
    T1 = I * dt
    T2 = A * (dt**2 / 2)
    T3 = (A @ A) * (dt**3 / 6)
    T4 = (A @ A @ A) * (dt**4 / 24)
    
    Bd = (T1 + T2 + T3 + T4) @ B
    
    return Ad, Bd