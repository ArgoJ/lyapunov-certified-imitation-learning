from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class StabilityReport:
    method: str = ""
    is_stable: bool = False
    applicability: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

@dataclass
class LyapunovDecreaseReport(StabilityReport):
    method: str = "Lyapunov Decrease"
    min_alpha: Optional[float] = float("nan")
    max_violation: Optional[float] = float("nan")
    applicability: bool = True # Always applicable

@dataclass
class GrüneHorizonReport(StabilityReport):
    method: str = "Grüne Horizon Condition"
    gamma_estimate: float = float("nan")
    alpha_N_estimate: float = float("nan")
    required_horizon: float = float("nan")

@dataclass
class AlphaViolationStats:
    min_alpha: Optional[float] = None
    max_violation: Optional[float] = None
    min_residual: Optional[float] = None
    n_used: int = 0