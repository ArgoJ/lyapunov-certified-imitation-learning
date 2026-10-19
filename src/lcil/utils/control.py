"""Shared control and matrix utilities for LCIL."""

from __future__ import annotations

from math import gamma, pi
from typing import Any, Literal, overload

import numpy as np
import torch as th
import torch.nn as nn
from scipy.linalg import solve_discrete_are as _scipy_solve_dare


@overload
def symmetrize(matrix: th.Tensor) -> th.Tensor: ...


@overload
def symmetrize(matrix: np.ndarray) -> np.ndarray: ...


def symmetrize(matrix: th.Tensor | np.ndarray) -> th.Tensor | np.ndarray:
    """Symmetrize a 2D or batched square matrix: 0.5 * (M + M^T).

    Parameters
    ----------
    matrix : torch.Tensor or numpy.ndarray
        Matrix of shape (..., n, n).

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Symmetrized matrix of the same type and shape.
    """
    if isinstance(matrix, th.Tensor):
        if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"Tensor must be at least 2D with square trailing dimensions, got {matrix.shape}.")
        return 0.5 * (matrix + matrix.transpose(-1, -2))
    if isinstance(matrix, np.ndarray):
        if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"Array must be at least 2D with square trailing dimensions, got {matrix.shape}.")
        return 0.5 * (matrix + np.swapaxes(matrix, -1, -2))
    raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(matrix).__name__}.")


def is_positive_definite(matrix: th.Tensor | np.ndarray, tol: float = 1e-8) -> bool:
    """Check if a matrix is symmetric positive definite with minimum eigenvalue > tol.

    Parameters
    ----------
    matrix : torch.Tensor or numpy.ndarray
        Square matrix of shape (n, n).
    tol : float, optional
        Minimum eigenvalue threshold, default 1e-8.

    Returns
    -------
    bool
        True if matrix is SPD with lambda_min > tol.
    """
    sym = symmetrize(matrix)
    if isinstance(sym, th.Tensor):
        min_eig = float(th.linalg.eigvalsh(sym).min().item())
    else:
        min_eig = float(np.min(np.linalg.eigvalsh(sym)))
    return min_eig > tol


@overload
def extract_p_matrix(
    obj: Any,
    *,
    as_tensor: Literal[True] = ...,
    dtype: th.dtype = ...,
    device: th.device | str = ...,
) -> th.Tensor: ...


@overload
def extract_p_matrix(
    obj: Any,
    *,
    as_tensor: Literal[False],
    dtype: th.dtype = ...,
    device: th.device | str = ...,
) -> np.ndarray: ...


def extract_p_matrix(
    obj: Any,
    *,
    as_tensor: bool = True,
    dtype: th.dtype = th.float64,
    device: th.device | str = "cpu",
) -> th.Tensor | np.ndarray:
    """Extract and symmetrize matrix P from a controller, policy, tensor, or array.

    Parameters
    ----------
    obj : BlendedLQRPolicy, LQRController, Tensor, ndarray, or object with P
        Source object containing or representing matrix P.
    as_tensor : bool, optional
        Whether to return a torch.Tensor (default True). If False, returns numpy.ndarray.
    dtype : torch.dtype, optional
        Tensor dtype if as_tensor=True (default torch.float64).
    device : torch.device or str, optional
        Tensor device if as_tensor=True (default 'cpu').

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Symmetric matrix P of shape (n, n).
    """
    if hasattr(obj, "lqr") and hasattr(obj.lqr, "P"):
        raw = obj.lqr.P
    elif hasattr(obj, "P"):
        raw = obj.P
    elif hasattr(obj, "p_matrix"):
        raw = obj.p_matrix
    elif hasattr(obj, "get_pd_matrix") and callable(obj.get_pd_matrix):
        raw = obj.get_pd_matrix()
    else:
        raw = obj

    try:
        if as_tensor:
            tensor = th.as_tensor(raw, dtype=dtype, device=device)
            return symmetrize(tensor)
        else:
            array = np.asarray(raw.detach().cpu().numpy() if isinstance(raw, th.Tensor) else raw, dtype=np.float64)
            return symmetrize(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot extract a valid square matrix P from object of type {type(obj).__name__}.") from exc


@overload
def solve_discrete_are(
    A: th.Tensor | np.ndarray,
    B: th.Tensor | np.ndarray,
    Q: th.Tensor | np.ndarray,
    R: th.Tensor | np.ndarray,
    *,
    as_tensor: Literal[True] = ...,
    dtype: th.dtype = ...,
    device: th.device | str = ...,
) -> th.Tensor: ...


@overload
def solve_discrete_are(
    A: th.Tensor | np.ndarray,
    B: th.Tensor | np.ndarray,
    Q: th.Tensor | np.ndarray,
    R: th.Tensor | np.ndarray,
    *,
    as_tensor: Literal[False],
    dtype: th.dtype = ...,
    device: th.device | str = ...,
) -> np.ndarray: ...


def solve_discrete_are(
    A: th.Tensor | np.ndarray,
    B: th.Tensor | np.ndarray,
    Q: th.Tensor | np.ndarray,
    R: th.Tensor | np.ndarray,
    *,
    as_tensor: bool = True,
    dtype: th.dtype = th.float32,
    device: th.device | str = "cpu",
) -> th.Tensor | np.ndarray:
    """Solve the Discrete Algebraic Riccati Equation (DARE) in float64 precision.

    Solves:
        A^T P A - P - (A^T P B)(R + B^T P B)^(-1)(B^T P A) + Q = 0

    Parameters
    ----------
    A : Tensor or ndarray
        Discrete-time state transition matrix of shape (nx, nx).
    B : Tensor or ndarray
        Discrete-time control input matrix of shape (nx, nu).
    Q : Tensor or ndarray
        State cost weight matrix of shape (nx, nx).
    R : Tensor or ndarray
        Control cost weight matrix of shape (nu, nu).
    as_tensor : bool, optional
        If True (default), returns torch.Tensor; if False, returns numpy.ndarray.
    dtype : torch.dtype, optional
        Output tensor dtype if as_tensor=True (default torch.float32).
    device : torch.device or str, optional
        Output tensor device if as_tensor=True (default 'cpu').

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Symmetric positive-definite solution matrix P of shape (nx, nx).
    """
    a_np = np.asarray(A.detach().cpu().numpy() if isinstance(A, th.Tensor) else A, dtype=np.float64)
    b_np = np.asarray(B.detach().cpu().numpy() if isinstance(B, th.Tensor) else B, dtype=np.float64)
    q_np = np.asarray(Q.detach().cpu().numpy() if isinstance(Q, th.Tensor) else Q, dtype=np.float64)
    r_np = np.asarray(R.detach().cpu().numpy() if isinstance(R, th.Tensor) else R, dtype=np.float64)

    p_np = _scipy_solve_dare(a_np, b_np, q_np, r_np)
    p_sym = symmetrize(p_np)

    if as_tensor:
        return th.as_tensor(p_sym, dtype=dtype, device=device)
    return p_sym


@overload
def compute_lqr_gain(
    A: th.Tensor | np.ndarray,
    B: th.Tensor | np.ndarray,
    P: th.Tensor | np.ndarray,
    R: th.Tensor | np.ndarray,
    *,
    as_tensor: Literal[True] = ...,
    dtype: th.dtype = ...,
    device: th.device | str = ...,
) -> th.Tensor: ...


@overload
def compute_lqr_gain(
    A: th.Tensor | np.ndarray,
    B: th.Tensor | np.ndarray,
    P: th.Tensor | np.ndarray,
    R: th.Tensor | np.ndarray,
    *,
    as_tensor: Literal[False],
    dtype: th.dtype = ...,
    device: th.device | str = ...,
) -> np.ndarray: ...


def compute_lqr_gain(
    A: th.Tensor | np.ndarray,
    B: th.Tensor | np.ndarray,
    P: th.Tensor | np.ndarray,
    R: th.Tensor | np.ndarray,
    *,
    as_tensor: bool = True,
    dtype: th.dtype = th.float32,
    device: th.device | str = "cpu",
) -> th.Tensor | np.ndarray:
    """Compute discrete-time LQR feedback gain K = (R + B^T P B)^(-1) B^T P A in float64.

    Parameters
    ----------
    A : Tensor or ndarray
        Discrete-time state transition matrix of shape (nx, nx).
    B : Tensor or ndarray
        Discrete-time control input matrix of shape (nx, nu).
    P : Tensor or ndarray
        Riccati / Lyapunov matrix of shape (nx, nx).
    R : Tensor or ndarray
        Control cost weight matrix of shape (nu, nu).
    as_tensor : bool, optional
        If True (default), returns torch.Tensor; if False, returns numpy.ndarray.
    dtype : torch.dtype, optional
        Output tensor dtype if as_tensor=True (default torch.float32).
    device : torch.device or str, optional
        Output tensor device if as_tensor=True (default 'cpu').

    Returns
    -------
    torch.Tensor or numpy.ndarray
        LQR feedback gain matrix K of shape (nu, nx).
    """
    A_64 = th.as_tensor(A, dtype=th.float64)
    B_64 = th.as_tensor(B, dtype=th.float64)
    P_64 = th.as_tensor(P, dtype=th.float64)
    R_64 = th.as_tensor(R, dtype=th.float64)

    M = R_64 + B_64.T @ P_64 @ B_64
    rhs = B_64.T @ P_64 @ A_64
    K_64 = th.linalg.solve(M, rhs)

    if as_tensor:
        return K_64.to(dtype=dtype, device=device)
    return K_64.detach().cpu().numpy().astype(np.float64)


@overload
def cholesky_factor(
    P: th.Tensor | np.ndarray,
    *,
    jitter: float = ...,
    as_tensor: Literal[True] = ...,
    dtype: th.dtype = ...,
    device: th.device | str = ...,
) -> th.Tensor: ...


@overload
def cholesky_factor(
    P: th.Tensor | np.ndarray,
    *,
    jitter: float = ...,
    as_tensor: Literal[False],
    dtype: th.dtype = ...,
    device: th.device | str = ...,
) -> np.ndarray: ...


def cholesky_factor(
    P: th.Tensor | np.ndarray,
    *,
    jitter: float = 0.0,
    as_tensor: bool = True,
    dtype: th.dtype = th.float32,
    device: th.device | str = "cpu",
) -> th.Tensor | np.ndarray:
    """Compute lower-triangular Cholesky factor L such that P = L L^T.

    Parameters
    ----------
    P : Tensor or ndarray
        Symmetric positive-definite matrix of shape (n, n).
    jitter : float, optional
        Diagonal regularization factor added if Cholesky fails, default 0.0.
    as_tensor : bool, optional
        If True (default), returns torch.Tensor; if False, returns numpy.ndarray.
    dtype : torch.dtype, optional
        Output tensor dtype if as_tensor=True (default torch.float32).
    device : torch.device or str, optional
        Output tensor device if as_tensor=True (default 'cpu').

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Lower-triangular factor L of shape (n, n).
    """
    p_64 = th.as_tensor(P, dtype=th.float64)
    p_sym = symmetrize(p_64)

    if jitter > 0.0:
        p_sym = p_sym + jitter * th.eye(p_sym.shape[-1], dtype=th.float64, device=p_sym.device)

    try:
        L_64 = th.linalg.cholesky(p_sym)
    except Exception:
        # Fallback with minimal numerical jitter
        eye = th.eye(p_sym.shape[-1], dtype=th.float64, device=p_sym.device)
        L_64 = th.linalg.cholesky(p_sym + 1e-7 * eye)

    if as_tensor:
        return L_64.to(dtype=dtype, device=device)
    return L_64.detach().cpu().numpy().astype(np.float64)


def quad_form(x: th.Tensor, P: th.Tensor) -> th.Tensor:
    """Evaluate quadratic form x^T P x for batched vectors: ((x @ P) * x).sum(dim=-1, keepdim=True).

    Parameters
    ----------
    x : torch.Tensor
        Input state tensor of shape (..., n).
    P : torch.Tensor
        Matrix of shape (n, n).

    Returns
    -------
    torch.Tensor
        Quadratic values of shape (..., 1).
    """
    return ((x @ P) * x).sum(dim=-1, keepdim=True)


def get_bounds_center_and_radius(
    bounds: th.Tensor | np.ndarray,
) -> tuple[Any, Any]:
    """Extract center and half-width radius from 2D bounds of shape (2, n).

    Parameters
    ----------
    bounds : torch.Tensor or numpy.ndarray
        Lower and upper bounds of shape (2, n).

    Returns
    -------
    tuple[Any, Any]
        (center, half_width) each of shape (n,).
    """
    if bounds.shape[0] != 2:
        raise ValueError(f"bounds must have leading dimension 2 ([lower, upper]), got {bounds.shape}.")
    center = 0.5 * (bounds[0] + bounds[1])
    radius = 0.5 * (bounds[1] - bounds[0])
    return center, radius


def quadratic_sublevel_set_volume(
    rho: float,
    P: th.Tensor | np.ndarray,
) -> float:
    r"""Compute the n-dimensional volume (Lebesgue measure) of the quadratic sublevel set {x : x^T P x <= rho}.

    The volume of the ellipsoid is given by:
    $$ \text{Vol}(\mathcal{E}_\rho) = \frac{\pi^{n/2}}{\Gamma(n/2 + 1)} \frac{\rho^{n/2}}{\sqrt{\det(P)}} $$

    Parameters
    ----------
    rho : float
        Non-negative sublevel set threshold.
    P : torch.Tensor or numpy.ndarray
        Positive definite matrix of shape (n, n).

    Returns
    -------
    float
        The volume (measure) of the ellipsoid.
    """
    if rho < 0.0:
        raise ValueError(f"rho must be non-negative, got {rho}.")

    P_sym = symmetrize(P)
    if isinstance(P_sym, th.Tensor):
        eigs = th.linalg.eigvalsh(P_sym).detach().cpu().numpy()
    else:
        eigs = np.linalg.eigvalsh(P_sym)

    if np.any(eigs <= 0.0):
        raise ValueError("P must be positive definite (all eigenvalues > 0).")

    n = len(eigs)
    unit_ball_vol = (pi ** (0.5 * n)) / gamma(0.5 * n + 1.0)
    return float(unit_ball_vol * (rho ** (0.5 * n)) / np.sqrt(np.prod(eigs)))


class QuadraticLyapunov(nn.Module):
    """Quadratic Lyapunov candidate model: V(x) = x^T P x.

    Parameters
    ----------
    P : Any
        Positive definite matrix of shape (n, n), or controller/policy object containing it.
    dtype : torch.dtype, optional
        Data type for the buffer (default torch.float32).
    device : torch.device or str, optional
        Device for the buffer (default 'cpu').
    """

    def __init__(
        self,
        P: Any,
        dtype: th.dtype = th.float32,
        device: th.device | str = "cpu",
    ) -> None:
        super().__init__()
        p_tensor = extract_p_matrix(P, as_tensor=True, dtype=dtype, device=device)
        self.register_buffer("P", p_tensor)
        self.register_buffer("p_matrix", p_tensor)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Evaluate V(x) = x^T P x."""
        return quad_form(x, self.P)

    def get_pd_matrix(self) -> th.Tensor:
        """Return the positive definite matrix P."""
        return self.P


RiccatiQuadraticLyapunov = QuadraticLyapunov
