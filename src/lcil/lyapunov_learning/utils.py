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


def compute_induced_1norm_gain(
    p_matrix: th.Tensor | np.ndarray,
    a_matrix: th.Tensor | np.ndarray,
) -> th.Tensor:
    """Compute the 1-norm of each column of P @ A @ P^{-1}.

    For a Lyapunov candidate whose linear term is ||P x||_1, exponential
    decrease ||P A x||_1 <= (1 - kappa) ||P x||_1 requires the induced matrix
    1-norm (the maximum column 1-norm) of P @ A @ P^{-1} to be <= 1 - kappa.

    Parameters
    ----------
    p_matrix : th.Tensor | np.ndarray
        Symmetric positive definite matrix P of shape (nx, nx).
    a_matrix : th.Tensor | np.ndarray
        Closed-loop discrete-time transition matrix A of shape (nx, nx).

    Returns
    -------
    th.Tensor
        1D tensor of column 1-norms of shape (nx,).
    """
    p = th.as_tensor(p_matrix, dtype=th.float64)
    a = th.as_tensor(a_matrix, dtype=th.float64, device=p.device)
    # Solve X @ P = P @ A <=> P.T @ X.T = (P @ A).T
    x_t = th.linalg.solve(p.T, (p @ a).T)
    x = x_t.T
    return x.abs().sum(dim=0)


def optimize_polyhedral_contraction_matrix(
    a_matrix: th.Tensor | np.ndarray,
    p_initial: th.Tensor | np.ndarray | None = None,
    *,
    kappa: float = 0.001,
    max_cond: float = 150.0,
    steps: int = 800,
    restarts: int = 3,
    beta: float = 300.0,
    target_gain: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Find a symmetric positive definite matrix P such that ||P A P^{-1}||_1 < 1 - kappa.

    Parameters
    ----------
    a_matrix : th.Tensor | np.ndarray
        Closed-loop transition matrix or Jacobian A of shape (nx, nx).
    p_initial : th.Tensor | np.ndarray | None, optional
        Initial SPD matrix seed (e.g., Riccati/ARE solution).
    kappa : float, optional
        Required decay rate, by default 0.001.
    max_cond : float, optional
        Condition number limit above which a penalty is applied, by default 150.0.
    steps : int, optional
        Optimization steps per restart, by default 800.
    restarts : int, optional
        Number of restarts, by default 3.
    beta : float, optional
        Temperature for smooth logsumexp approximation, by default 300.0.
    target_gain : float | None, optional
        Target maximum column gain for early stopping. Defaults to 1.0 - kappa - 0.01.
    seed : int, optional
        Random seed for reproducibility, by default 0.

    Returns
    -------
    tuple[np.ndarray, float]
        The optimized SPD matrix P (normalized such that ||P||_2 = 1.0) as a numpy array,
        and its induced 1-norm gain max_j ||(P A P^{-1})_{:, j}||_1.
    """
    a = th.as_tensor(a_matrix, dtype=th.float64)
    n = a.shape[0]
    eps = 1e-4
    if target_gain is None:
        target_gain = 1.0 - kappa - 0.01

    if p_initial is not None:
        p0 = th.as_tensor(p_initial, dtype=th.float64)
        p0 = 0.5 * (p0 + p0.T)
        p0 = p0 / th.linalg.matrix_norm(p0, 2)
        initial_gain = compute_induced_1norm_gain(p0, a).max().item()
        try:
            l0 = th.linalg.cholesky(p0).T
        except Exception:
            l0 = th.eye(n, dtype=th.float64)
    else:
        p0 = th.eye(n, dtype=th.float64)
        l0 = th.eye(n, dtype=th.float64)
        initial_gain = compute_induced_1norm_gain(p0, a).max().item()

    if initial_gain <= target_gain:
        return p0.detach().cpu().numpy(), initial_gain

    best_gain = initial_gain
    best_p = p0.detach().clone()
    g = th.Generator().manual_seed(seed)

    for k in range(restarts):
        if k == 0:
            r = l0.clone()
        elif k == 1:
            r = th.eye(n, dtype=th.float64)
        else:
            r = l0 + 0.2 * th.randn(n, n, generator=g, dtype=th.float64) * l0.abs().mean()
        r = r.requires_grad_(True)
        opt = th.optim.Adam([r], lr=2e-2)

        for _ in range(steps):
            p = eps * th.eye(n, dtype=th.float64) + r.T @ r
            p = p / th.linalg.matrix_norm(p, 2)
            cs = compute_induced_1norm_gain(p, a)
            ev = th.linalg.eigvalsh(p)
            cond = ev[-1] / th.clamp(ev[0], min=1e-8)
            cond_pen = th.relu(th.log(cond) - np.log(max_cond)) ** 2
            loss = th.logsumexp(beta * cs, 0) / beta + 10.0 * cond_pen
            opt.zero_grad()
            loss.backward()
            opt.step()

        with th.no_grad():
            p = eps * th.eye(n, dtype=th.float64) + r.T @ r
            p = p / th.linalg.matrix_norm(p, 2)
            gm = compute_induced_1norm_gain(p, a).max().item()
        if gm < best_gain:
            best_gain = gm
            best_p = p.detach().clone()
        if best_gain <= target_gain:
            break

    return best_p.detach().cpu().numpy(), best_gain


def scale_riccati_matrix(
    riccati_p: th.Tensor | np.ndarray,
    scale_mode: str | float = "none",
) -> th.Tensor:
    """Scale a Riccati matrix by spectral norm, Frobenius norm, or a custom factor.

    Parameters
    ----------
    riccati_p : th.Tensor | np.ndarray
        Matrix P to scale.
    scale_mode : str | float, optional
        Scaling mode ('none', 'spectral', 'frobenius', or a numeric divisor), by default "none".

    Returns
    -------
    th.Tensor
        Scaled matrix P as a torch tensor.
    """
    p = th.as_tensor(riccati_p)
    if scale_mode != "none":
        if scale_mode == "spectral":
            scale_factor = th.linalg.norm(p, ord=2).item()
        elif scale_mode == "frobenius":
            scale_factor = th.linalg.norm(p, ord="fro").item()
        else:
            try:
                scale_factor = float(scale_mode)
            except ValueError:
                raise ValueError(
                    f"Invalid riccati_scale mode: {scale_mode}. "
                    "Must be 'none', 'spectral', 'frobenius', or a numeric value."
                )
        p = p / scale_factor
    return p


def compute_polyhedral_value_matrix(
    a_closed_loop: th.Tensor | np.ndarray,
    p_initial: th.Tensor | np.ndarray | None = None,
    *,
    kappa: float = 0.001,
    max_cond: float = 150.0,
    scale_mode: str | float = "none",
) -> np.ndarray:
    """Compute an SPD matrix P such that ||P A_cl P^{-1}||_1 < 1 - kappa.

    Parameters
    ----------
    a_closed_loop : th.Tensor | np.ndarray
        Closed-loop transition matrix or Jacobian A_cl around the origin.
    p_initial : th.Tensor | np.ndarray | None, optional
        Initial SPD matrix seed (e.g. from DARE/Riccati).
    kappa : float, optional
        Required decay rate, by default 0.001.
    max_cond : float, optional
        Maximum allowed condition number, by default 150.0.
    scale_mode : str | float, optional
        Scaling mode for the resulting matrix ('none', 'spectral', 'frobenius'),
        by default "none".

    Returns
    -------
    np.ndarray
        Optimized SPD matrix P satisfying ||P A_cl P^{-1}||_1 < 1 - kappa.
    """
    p_opt, _ = optimize_polyhedral_contraction_matrix(
        a_matrix=a_closed_loop,
        p_initial=p_initial,
        kappa=kappa,
        max_cond=max_cond,
    )
    if scale_mode != "none":
        p_opt_th = scale_riccati_matrix(th.as_tensor(p_opt), scale_mode=scale_mode)
        p_opt = p_opt_th.detach().cpu().numpy()
    return p_opt


def calculate_r_factor_from_riccati(
    riccati_p: th.Tensor | np.ndarray,
    eps: float = 0.0,
) -> tuple[th.Tensor, float]:
    """Calculate the factor R from a Riccati/SPD matrix such that eps * I + R^T R = P.

    Parameters
    ----------
    riccati_p : th.Tensor | np.ndarray
        The Riccati/SPD value matrix P.
    eps : float, optional
        A small positive constant eps for the candidate linear term, by default 0.0.

    Returns
    -------
    tuple[th.Tensor, float]
        The factor R of shape (nx, nx) and the (potentially adjusted) epsilon value.

    Raises
    ------
    ValueError
        If riccati_p is not positive semi-definite.
    """
    p = th.as_tensor(riccati_p)
    p_sym = 0.5 * (p + p.transpose(-1, -2))
    eigvals, eigvecs = th.linalg.eigh(p_sym)
    scale = max(1.0, float(th.linalg.norm(p_sym, ord=2).item()))
    tol = 1e-6 * scale
    min_eig = float(eigvals[0].item())
    if min_eig < -tol:
        raise ValueError(
            "riccati_p must satisfy P >= 0 so it can seed R^T R. "
            f"Minimum eigenvalue is {min_eig:.6e}."
        )

    min_eig_clamped = max(0.0, min_eig)
    adjusted_eps = float(eps)
    if min_eig < adjusted_eps:
        adjusted_eps = 0.5 * min_eig_clamped
        __logger__.warning(
            "Minimum eigenvalue of riccati_p (λ_min=%.6e) is less than eps=%.6e. "
            "Setting eps to 0.5 * λ_min (%.6e) so that R remains full-rank and trainable.",
            min_eig,
            eps,
            adjusted_eps,
        )

    adjusted_eigvals = (eigvals - adjusted_eps).clamp_min(0.0)
    factor = th.diag(th.sqrt(adjusted_eigvals)) @ eigvecs.transpose(-1, -2)
    return factor, adjusted_eps


