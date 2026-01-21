from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class StabilityReport:
    method: str = ""
    is_stable: bool = False
    applicability: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""