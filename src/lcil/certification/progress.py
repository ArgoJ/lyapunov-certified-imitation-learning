from __future__ import annotations

import logging
from enum import Enum
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
)

__logger__ = logging.getLogger(__name__)

class ProgressLevel(Enum):
    NONE = 0
    TOP = 1
    ALL = 2

class CertificationProgress(Progress):
    """Manages rich.Progress instances and formatting for certification tasks."""
    def __init__(self, level: int | ProgressLevel):
        try:
            self.level = ProgressLevel(level)
        except ValueError:
            valid_values = [e.value for e in ProgressLevel]
            raise ValueError(
                f"progress_level must be a PROGRESS_LEVEL enum or one of its integer values: {valid_values}."
            )

        columns = (
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[detail]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        super().__init__(
            *columns,
            transient=True,
            disable=not self._show_progress(ProgressLevel.TOP),
        )

        self._recursive_task: TaskID | None = None
        self._certify_task: TaskID | None = None

        # State tracking for recursive progress
        self._rec_resolved: int = 0
        self._rec_unresolved: int = 0
        self._rec_irrelevant: int = 0
        self._rec_pending: int = 0

        # State tracking for certify progress
        self._cert_verified: int = 0
        self._cert_unknown: int = 0
        self._cert_cex: int = 0

        self._entered_depth = 0

    @staticmethod
    def _format_counts(**counts: int) -> str:
        color_map = {
            "safe": "green",
            "unknown": "yellow",
            "outside": "dim",
            "counterexample": "red",
            "pending": "red",
        }
        parts = [
            f"[{color_map.get(k, 'white')}]{k}: {v}[/{color_map.get(k, 'white')}]"
            for k, v in counts.items()
            if v is not None
        ]
        return " " + ", ".join(parts) + " " if parts else ""

    def _rec_detail(self) -> str:
        return self._format_counts(
            safe=self._rec_resolved,
            unknown=self._rec_unresolved,
            outside=self._rec_irrelevant,
            pending=self._rec_pending,
        )

    def _cert_detail(self) -> str:
        return self._format_counts(
            safe=self._cert_verified,
            unknown=self._cert_unknown,
            counterexample=self._cert_cex,
        )

    def _show_progress(self, required_level: ProgressLevel) -> bool:
        return self.level.value >= required_level.value

    def __enter__(self) -> CertificationProgress:
        if self._entered_depth == 0:
            super().__enter__()
        self._entered_depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._entered_depth > 0:
            self._entered_depth -= 1
            if self._entered_depth == 0:
                super().__exit__(exc_type, exc_val, exc_tb)

    def log_initialization(self, class_name: str) -> None:
        """Log that progress bars are enabled for the given class."""
        if self._show_progress(ProgressLevel.TOP):
            color = "cyan" if self.level == ProgressLevel.TOP else "green"
            __logger__.info("Enabling progress bars for %s: [%s]%s", class_name, color, self.level)

    # ==========================================
    # RECURSIVE
    # ==========================================
    def start_recursive(self, description: str, max_depth: int, force_display: bool = False) -> None:
        if force_display or self._show_progress(ProgressLevel.ALL):
            self._recursive_task = self.add_task(description, total=float(max_depth + 1), detail="")
            self._rec_resolved = 0
            self._rec_unresolved = 0
            self._rec_irrelevant = 0
            self._rec_pending = 0

    def stop_recursive(self) -> None:
        if self._recursive_task is not None:
            self.remove_task(self._recursive_task)
            self._recursive_task = None

    def update_recursive(
        self, 
        advance: int | None = None, 
        is_completed: bool = False, 
        max_depth: int = 0,
        n_resolved: int | None = None, 
        n_unresolved: int | None = None, 
        n_irrelevant: int | None = None, 
        n_pending: int | None = None
    ) -> None:
        if n_resolved is not None:
            self._rec_resolved = n_resolved
        if n_unresolved is not None:
            self._rec_unresolved = n_unresolved
        if n_irrelevant is not None:
            self._rec_irrelevant = n_irrelevant
        if n_pending is not None:
            self._rec_pending = max(0, n_pending)

        if self._recursive_task is not None:
            if is_completed:
                self.update(self._recursive_task, completed=max_depth + 1)
            else:
                self.update(self._recursive_task, advance=advance, detail=self._rec_detail())
            self.refresh()

    def add_recursive_counts(
        self,
        resolved: int = 0,
        unresolved: int = 0,
        irrelevant: int = 0,
        pending: int = 0
    ) -> None:
        if resolved:
            self._rec_resolved += resolved
        if unresolved:
            self._rec_unresolved += unresolved
        if irrelevant:
            self._rec_irrelevant += irrelevant
        if pending:
            self._rec_pending = max(0, self._rec_pending + pending)

        if self._recursive_task is not None:
            self.update(self._recursive_task, detail=self._rec_detail())
            self.refresh()

    # ==========================================
    # CERTIFY REGIONS
    # ==========================================
    def start_certify(self, description: str = "Certify Regions", total: int = 0) -> None:
        if self._show_progress(ProgressLevel.ALL):
            self._certify_task = self.add_task(description, total=float(total), detail="")
            self._cert_verified = 0
            self._cert_unknown = 0
            self._cert_cex = 0
            self.update(self._certify_task, detail=self._cert_detail())
            self.refresh()

    def stop_certify(self) -> None:
        if self._certify_task is not None:
            self.remove_task(self._certify_task)
            self._certify_task = None

    def step_certify(self, verified: bool, counterexample: bool = False) -> None:
        """Advance the certify progress bar by one region outcome."""
        if verified:
            self._cert_verified += 1
        elif counterexample:
            self._cert_cex += 1
        else:
            self._cert_unknown += 1

        if self._certify_task is not None:
            self.update(self._certify_task, advance=1, detail=self._cert_detail())
            self.refresh()

    def update_certify(
        self,
        advance: int | None = None,
        verified_count: int | None = None,
        unknown_count: int | None = None,
        cex_count: int | None = None,
    ) -> None:
        if verified_count is not None:
            self._cert_verified = verified_count
        if unknown_count is not None:
            self._cert_unknown = unknown_count
        if cex_count is not None:
            self._cert_cex = cex_count

        if self._certify_task is not None:
            self.update(self._certify_task, advance=advance, detail=self._cert_detail())
            self.refresh()