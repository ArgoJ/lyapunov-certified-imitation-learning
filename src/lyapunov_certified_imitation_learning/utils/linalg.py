import numpy as np
from scipy.linalg import expm
from typing import Tuple

def c2d(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
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