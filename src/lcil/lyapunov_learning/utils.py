from __future__ import annotations

from dataclasses import dataclass, field


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
    consecutive_low: int = field(default_factory=int, init=False)

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
