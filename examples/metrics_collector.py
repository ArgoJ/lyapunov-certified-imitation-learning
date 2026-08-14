from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

import torch as th

from lcil.certification import (
    LevelSetEstimate,
    RegionCertificationResult,
    estimate_level_set_measure,
)
from .constants import LEVEL_SET_FILENAME

__logger__ = logging.getLogger("lcil.examples.metrics_collector")


def add_entry(entry: str, output_root: str | Path, summary_name: str) -> None:
    """Appends a string to a file in the specified directory."""
    summary_path = Path(output_root) / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "a", encoding="utf-8") as f:
        if not entry.endswith("\n"):
            entry += "\n"
        f.write(entry)


class LevelSetMetricsWriter:
    """Context manager and helper to compute, save, and aggregate level set estimates into a CSV summary."""

    def __init__(
        self,
        output_root: Path | str,
        summary_name: str | None = None,
        method: Literal["ray_shooting", "monte_carlo"] = "ray_shooting",
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.method = method
        self.summary_name = summary_name or f"level_set_estimates_{method}.csv"
        self.summary_path = self.output_root / self.summary_name
        self.count = 0

    def __enter__(self) -> Self:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.summary_path.exists() or self.summary_path.stat().st_size == 0:
            with open(self.summary_path, "w", encoding="utf-8") as f:
                f.write("model,measure\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.count > 0:
            __logger__.info("Saved %d level set estimates to %s", self.count, self.summary_path)

    def add_entry(self, entry: str) -> None:
        """Appends a custom string entry to the summary file."""
        add_entry(entry, self.output_root, self.summary_name)
        self.count += 1

    def record(
        self,
        *,
        cert_dir: Path,
        cert_result: RegionCertificationResult,
        lyapunov_fn: Callable[[th.Tensor], th.Tensor],
        state_dim: int,
        device: th.device,
        method: Literal["ray_shooting", "monte_carlo"] | None = None,
    ) -> LevelSetEstimate:
        use_method = method or self.method
        metrics_path = cert_dir / LEVEL_SET_FILENAME
        estimate = estimate_level_set_measure(
            lyapunov_fn=lyapunov_fn,
            rho=float(cert_result.rho),
            num_states=int(state_dim),
            device=device,
            method=use_method,
        )
        estimate.save(metrics_path)

        add_entry(
            f"{cert_dir.parent.name},{estimate.measure:.6f}",
            output_root=self.output_root,
            summary_name=self.summary_name,
        )
        self.count += 1
        return estimate


LevelSetMetricsCollector = LevelSetMetricsWriter


def save_level_set_metrics(
    *,
    cert_dir: Path,
    cert_result: RegionCertificationResult,
    lyapunov_fn: Callable[[th.Tensor], th.Tensor],
    state_dim: int,
    device: th.device,
    method: Literal["ray_shooting", "monte_carlo"] = "ray_shooting",
    output_root: Path | str | None = None,
    summary_name: str | None = None,
) -> LevelSetEstimate:
    """Convenience function to compute and save level set estimate and append to summary CSV."""
    root = output_root if output_root is not None else cert_dir.parents[1]
    with LevelSetMetricsWriter(output_root=root, summary_name=summary_name, method=method) as writer:
        return writer.record(
            cert_dir=cert_dir,
            cert_result=cert_result,
            lyapunov_fn=lyapunov_fn,
            state_dim=state_dim,
            device=device,
            method=method,
        )
