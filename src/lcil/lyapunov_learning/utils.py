import logging
import numpy as np
import torch as th

from numpy.typing import NDArray
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__logger__ = logging.getLogger(__name__)


def get_th_lbx_ubx(bounds: NDArray, device: th.device | str = "cpu") -> tuple[th.Tensor, th.Tensor]:
    """Convert bounds to lbx and ubx arrays."""
    th_bounds = th.as_tensor(bounds, dtype=th.float32, device=device)
    if th_bounds.ndim != 2 or th_bounds.shape[0] != 2:
        raise ValueError("state_bounds must have shape (2, nx).")
    
    lbx = th_bounds[0].reshape(-1)
    ubx = th_bounds[1].reshape(-1)
    return lbx, ubx


def get_ema(old: float | None, new: float, decay: float) -> float:
    if old is None:
        return new
    return decay * old + (1 - decay) * new


def get_center(lbx: th.Tensor, ubx: th.Tensor) -> th.Tensor:
    """Compute the center of the state space from lbx and ubx."""
    return (lbx + ubx) / 2.0


def get_bounded_fraction(base: float, min: float, max: float) -> float:
    if base < min:
        return min
    if base > max:
        return max
    return base


def compute_max_admissible_kappa(
    riccati_p: th.Tensor,
    q_matrix: th.Tensor,
    r_matrix: th.Tensor,
    k_gain: th.Tensor,
) -> float:
    """Compute the maximum theoretical discrete decay rate kappa_max.
    
    kappa_max = min eig(Q + K^T R K, P)
    """
    p = th.as_tensor(riccati_p, dtype=th.float64)
    q = th.as_tensor(q_matrix, dtype=th.float64, device=p.device)
    r = th.as_tensor(r_matrix, dtype=th.float64, device=p.device)
    k = th.as_tensor(k_gain, dtype=th.float64, device=p.device)

    W = q + k.T @ r @ k
    eigvals = th.linalg.eigvals(th.linalg.solve(p, W)).real
    return float(th.min(eigvals).item())


def check_kappa(
    kappa: float,
    riccati_p: th.Tensor,
    q_matrix: th.Tensor,
    r_matrix: th.Tensor,
    k_gain: th.Tensor,
) -> float:
    """Validate that kappa does not exceed the theoretical maximum and warn if it does."""
    kappa_max = compute_max_admissible_kappa(riccati_p, q_matrix, r_matrix, k_gain)

    if float(kappa) >= kappa_max:
        __logger__.warning(
            "Configured decay rate kappa=%.4e exceeds the theoretical maximum "
            "admissible kappa_max=%.4e (min eig(Q + K^T R K, P)). "
            "The discrete Lyapunov decrease condition cannot be satisfied on the linearized system!",
            float(kappa), kappa_max,
        )

    return kappa_max


class TrainingAbortedError(RuntimeError):
    """Raised when Lyapunov training aborts before producing a valid result."""


@dataclass
class ThresholdMonitor:
    """Track values and detect sustained low-value runs.

    Parameters
    ----------
    threshold : float
        Values below this threshold count toward the stopping streak.
    patience : int
        Number of consecutive below-threshold values required to trigger.
    """

    threshold: float = 1.0
    patience: int = 10
    value_history: list[float] = field(default_factory=list, init=False)
    consecutive_low: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.threshold <= 0.0:
            raise ValueError("threshold must be positive.")

    def update(self, value: float) -> bool:
        """Register one value and return whether training should stop."""
        value = float(value)
        self.value_history.append(value)
        if value < self.threshold:
            self.consecutive_low += 1
        else:
            self.consecutive_low = 0
        return self.should_stop

    @property
    def should_stop(self) -> bool:
        """Return whether the low-value stopping criterion is active."""
        return self.consecutive_low >= self.patience
    
    def reset(self) -> None:
        self.consecutive_low = 0
        self.value_history.clear()


def compute_closed_loop_jacobian(
    dynamics: Callable[[th.Tensor, th.Tensor], th.Tensor],
    policy: Callable[[th.Tensor], th.Tensor],
    x_star: th.Tensor | Sequence[float] | None = None,
    state_dim: int | None = None,
    device: th.device | str = "cpu",
) -> th.Tensor:
    """Compute the Jacobian of the discrete closed-loop system at the equilibrium.

    Evaluates the linearization of x_{k+1} = dynamics(x_k, policy(x_k)) around x*.

    Parameters
    ----------
    dynamics : Callable[[th.Tensor, th.Tensor], th.Tensor]
        Discrete-time dynamics model f(x, u).
    policy : Callable[[th.Tensor], th.Tensor]
        Feedback policy model pi(x).
    x_star : th.Tensor | Sequence[float] | None, optional
        Equilibrium state around which to linearize. If None, state_dim must be provided
        and the zero state is used.
    state_dim : int | None, optional
        State dimension if x_star is None.
    device : th.device | str, optional
        Target torch device, by default "cpu".

    Returns
    -------
    th.Tensor
        Closed-loop Jacobian A_cl of shape (nx, nx).
    """
    if x_star is None:
        if state_dim is None:
            raise ValueError("Either x_star or state_dim must be provided.")
        x0 = th.zeros(state_dim, dtype=th.float32, device=device)
    else:
        x0 = th.as_tensor(x_star, dtype=th.float32, device=device).reshape(-1)

    def closed_loop(x: th.Tensor) -> th.Tensor:
        x_batch = x.unsqueeze(0)
        u_batch = policy(x_batch)
        next_batch = dynamics(x_batch, u_batch)
        return next_batch.squeeze(0)

    jacobian = th.autograd.functional.jacobian(closed_loop, x0)
    return jacobian


def compute_antiphase_eigenvectors(
    matrix: th.Tensor | np.ndarray | Sequence[Sequence[float]] | None = None,
    *,
    dynamics: Callable[[th.Tensor, th.Tensor], th.Tensor] | None = None,
    policy: Callable[[th.Tensor], th.Tensor] | None = None,
    x_star: th.Tensor | Sequence[float] | None = None,
    state_dim: int | None = None,
    antiphase_only: bool = True,
    device: th.device | str = "cpu",
) -> th.Tensor:
    """Extract unit-normalized modal directions (eigenvectors) from a closed-loop system.

    Modes are sorted by descending eigenvalue magnitude (|lambda|), placing the slowest
    decaying / dominant modes first. For complex conjugate pairs, both the real and imaginary
    modal directions are extracted.

    Parameters
    ----------
    matrix : th.Tensor | np.ndarray | Sequence[Sequence[float]] | None, optional
        Square transition matrix or Jacobian A of shape (nx, nx). If None, dynamics and policy
        must be supplied to compute the closed-loop Jacobian.
    dynamics : Callable[[th.Tensor, th.Tensor], th.Tensor] | None, optional
        Dynamics model f(x, u) used to compute the Jacobian if matrix is None.
    policy : Callable[[th.Tensor], th.Tensor] | None, optional
        Policy model pi(x) used to compute the Jacobian if matrix is None.
    x_star : th.Tensor | Sequence[float] | None, optional
        Equilibrium state around which to linearize if matrix is None.
    state_dim : int | None, optional
        State dimension if x_star is None and matrix is None.
    antiphase_only : bool, optional
        If True, filters for directions with opposing signs (having both significantly positive
        and negative components). If no mode satisfies this, all extracted modes are returned.
        Default is True.
    device : th.device | str, optional
        Target torch device, by default "cpu".

    Returns
    -------
    th.Tensor
        Extracted unit-normalized directions of shape (num_directions, nx).
    """
    if matrix is None:
        if dynamics is None or policy is None:
            raise ValueError("Either matrix or both dynamics and policy must be provided.")
        mat_t = compute_closed_loop_jacobian(dynamics, policy, x_star=x_star, state_dim=state_dim, device=device)
    else:
        mat_t = th.as_tensor(matrix, dtype=th.float32, device=device)

    if mat_t.ndim != 2 or mat_t.shape[0] != mat_t.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {tuple(mat_t.shape)}.")

    vals, vecs = th.linalg.eig(mat_t)
    order = th.argsort(th.abs(vals), descending=True)

    unique_dirs: list[th.Tensor] = []
    for idx in order:
        v = vecs[:, idx]
        parts = [th.real(v)]
        if th.abs(th.imag(vals[idx])) > 1e-6:
            parts.append(th.imag(v))
        for p in parts:
            norm = th.linalg.norm(p)
            if norm > 1e-7:
                p_unit = p / norm
                if not any(th.abs(th.dot(p_unit, u)) > 0.999 for u in unique_dirs):
                    unique_dirs.append(p_unit)

    if not unique_dirs:
        return th.eye(mat_t.shape[0], dtype=th.float32, device=device)

    candidates = th.stack(unique_dirs, dim=0)
    if antiphase_only:
        has_pos = th.any(candidates > 1e-5, dim=-1)
        has_neg = th.any(candidates < -1e-5, dim=-1)
        antiphase_mask = has_pos & has_neg
        if th.any(antiphase_mask):
            candidates = candidates[antiphase_mask]

    return candidates

