import numpy as np
import casadi as ca

from scipy.linalg import expm, solve_discrete_are
from typing import Tuple


def is_psd(M: np.ndarray, tol: float = 1e-12) -> bool:
    """Check if matrix M is positive semi-definite."""
    M = 0.5*(M + M.T)
    eig = np.linalg.eigvalsh(M)
    return bool(np.min(eig) >= -tol)

def is_pd(M: np.ndarray, tol: float = 1e-12) -> bool:
    """Check if matrix M is positive definite."""
    M = 0.5*(M + M.T)
    eig = np.linalg.eigvalsh(M)
    return bool(np.min(eig) > tol)

def sqrt_psd(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Compute the principal square root of a PSD matrix M."""
    M = 0.5*(M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T   # V diag(sqrt(w)) V^T

def pbh_stabilizable(A: np.ndarray, B: np.ndarray, tol: float = 1e-9) -> bool:
    """Check if the pair (A,B) is stabilizable using PBH test."""
    A = np.asarray(A); B = np.asarray(B)
    n = A.shape[0]
    eigvals = np.linalg.eigvals(A)
    for lam in eigvals:
        if abs(lam) >= 1.0 - 1e-12:  # unstable/marginal discrete-time
            M = np.hstack([lam*np.eye(n) - A, B])
            if np.linalg.matrix_rank(M, tol=tol) < n:
                return False
    return True

def pbh_detectable(A: np.ndarray, Q: np.ndarray, tol_rank: float = 1e-9) -> bool:
    """Check if the pair (A,Q) is detectable using PBH test."""
    A = np.asarray(A); Q = np.asarray(Q)
    n = A.shape[0]
    C = sqrt_psd(Q)  # so that C^T C = Q
    eigvals = np.linalg.eigvals(A)
    for lam in eigvals:
        if abs(lam) >= 1.0 - 1e-12:
            M = np.vstack([lam*np.eye(n) - A, C])
            if np.linalg.matrix_rank(M, tol=tol_rank) < n:
                return False
    return True

def dare_residual(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute the residual of the discrete-time algebraic Riccati equation."""
    A = np.asarray(A); B = np.asarray(B)
    Q = np.asarray(Q); R = np.asarray(R); P = np.asarray(P)
    
    S = R + B.T @ P @ B
    K = np.linalg.solve(S, B.T @ P @ A)          # K = (R+B'PB)^{-1} B'PA
    P_rhs = A.T @ P @ A - A.T @ P @ B @ K + Q
    return 0.5*((P - P_rhs) + (P - P_rhs).T)


def rk4_step(x, u, f_fun, h):
    k1 = f_fun(x, u)
    k2 = f_fun(x + 0.5*h*k1, u)
    k3 = f_fun(x + 0.5*h*k2, u)
    k4 = f_fun(x + h*k3, u)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def discretize_and_linearize_rk4(
    x_sym: ca.SX,
    u_sym: ca.SX,
    f_expl_expr: ca.SX,
    dt: float,
    x_lin: np.ndarray,
    u_lin: np.ndarray,
    num_steps: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize and linearize continuous-time dynamics using RK4 method.

    Parameters
    ----------
    x_sym : ca.SX
        Symbolic state variable.
    u_sym : ca.SX
        Symbolic input variable.
    f_expl_expr : ca.SX
        Symbolic expression of continuous-time dynamics (x_dot = f_expl_expr).
    dt : float
        Discretization time step.
    x_lin : np.ndarray
        State around which to linearize.
    u_lin : np.ndarray
        Input around which to linearize.
    num_steps : int, optional
        Number of RK4 steps within dt, by default 1.

    Returns
    -------
    Ad : ndarray
        Discrete-time state matrix.
    Bd : ndarray
        Discrete-time input matrix.
    gd : ndarray
        Discretization offset term.
    """
    f_fun = ca.Function("f_fun", [x_sym, u_sym], [f_expl_expr])

    h = float(dt) / int(num_steps)
    x_next = x_sym
    for _ in range(int(num_steps)):
        x_next = rk4_step(x_next, u_sym, f_fun, h)
    phi = x_next

    Ad_expr = ca.jacobian(phi, x_sym)
    Bd_expr = ca.jacobian(phi, u_sym)

    phi_fun = ca.Function("phi_fun", [x_sym, u_sym], [phi])
    Ad_fun  = ca.Function("Ad_fun",  [x_sym, u_sym], [Ad_expr])
    Bd_fun  = ca.Function("Bd_fun",  [x_sym, u_sym], [Bd_expr])

    phi0 = np.array(phi_fun(x_lin, u_lin)).astype(float)
    Ad   = np.array(Ad_fun(x_lin, u_lin)).astype(float)
    Bd   = np.array(Bd_fun(x_lin, u_lin)).astype(float)

    gd = (phi0 - Ad @ np.asarray(x_lin).reshape(-1,) - Bd @ np.asarray(u_lin).reshape(-1,)).reshape(-1,)
    return Ad, Bd, gd


def lin_c2d_rk4(A: np.ndarray, B: np.ndarray, dt: float, num_steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize linear system x_dot = A x + B u using RK4 method.

    Returns
    -------
    Ad : ndarray
        Discrete-time state matrix.
    Bd : ndarray
        Discrete-time input matrix.
    """
    n = A.shape[0]
    m = B.shape[1]

    x_sym = ca.SX.sym("x", n)
    u_sym = ca.SX.sym("u", m)
    f_expl_expr = A @ x_sym + B @ u_sym

    Ad, Bd, _ = discretize_and_linearize_rk4(
        x_sym, u_sym, f_expl_expr, dt,
        x_lin=np.zeros((n,)), u_lin=np.zeros((m,)),
        num_steps=num_steps
    )
    return Ad, Bd