from __future__ import annotations

import torch as th

from numpy.typing import NDArray
from dataclasses import dataclass, field


def get_th_lbx_ubx(bounds: NDArray, device: th.device = "cpu") -> tuple[float, float]:
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
    if not (min <= base <= max):
        return max(min, min(base, max))
    return base


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
