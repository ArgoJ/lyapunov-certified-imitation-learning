from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go

from lcil.utils.constants import (
    TRAINING_CONFIG_FILENAME,
    TRAINING_METRICS_FILENAME,
)

__logger__ = logging.getLogger("lcil.examples.plot_training_metrics")

# Curated qualitative palette for clean multi-run comparisons
DISTINCT_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
    "#3366cc",  # Royal Blue
    "#dc3912",  # Crimson
    "#ff9900",  # Amber
    "#109618",  # Forest
    "#990099",  # Violet
]


@dataclass
class MetricSeries:
    """Represents a single 1D numerical metric trajectory over time/iterations."""

    name: str
    values: NDArray
    x_values: NDArray
    x_label: str = "Epoch"
    is_validation: bool = False

    @property
    def valid_count(self) -> int:
        return len(self.values)

    @property
    def min_val(self) -> float | None:
        return float(np.nanmin(self.values)) if len(self.values) > 0 else None

    @property
    def max_val(self) -> float | None:
        return float(np.nanmax(self.values)) if len(self.values) > 0 else None

    @property
    def final_val(self) -> float | None:
        return float(self.values[-1]) if len(self.values) > 0 else None

    @property
    def initial_val(self) -> float | None:
        return float(self.values[0]) if len(self.values) > 0 else None


@dataclass
class MetricRun:
    """Holds all training and validation metrics for a single experiment run."""

    name: str
    path: Path
    train_metrics: dict[str, MetricSeries] = field(default_factory=dict)
    val_metrics: dict[str, MetricSeries] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def get_all_metric_names(self) -> list[str]:
        return sorted(set(self.train_metrics.keys()).union(self.val_metrics.keys()))


def exponential_moving_average(scalars: NDArray, weight: float = 0.0) -> NDArray:
    """Compute exponential moving average (EMA) of a 1D sequence."""
    if len(scalars) == 0 or weight <= 0.0:
        return scalars.copy()

    smoothed = np.empty_like(scalars, dtype=np.float64)
    last = scalars[0]
    for i, val in enumerate(scalars):
        if np.isnan(val):
            smoothed[i] = np.nan
            continue
        if np.isnan(last):
            last = val
        last = last * weight + (1.0 - weight) * val
        debias_factor = 1.0 - (weight ** (i + 1))
        smoothed[i] = last / debias_factor if debias_factor > 0 else last
    return smoothed


def _extract_metric_series(
    raw_dict: dict[str, Any] | np.lib.npyio.NpzFile,
    is_validation: bool = False,
) -> tuple[dict[str, MetricSeries], dict[str, Any]]:
    """Extract 1D metric trajectories and scalar metadata from loaded npz dictionary."""
    metrics: dict[str, MetricSeries] = {}
    scalars: dict[str, Any] = {}

    for k in raw_dict.files if hasattr(raw_dict, "files") else raw_dict.keys():
        val = raw_dict[k]
        if np.ndim(val) == 0:
            scalars[k] = val.item() if hasattr(val, "item") else val

    epochs_completed = scalars.get("epochs_completed")
    outer_completed = scalars.get("outer_iterations_completed")
    inner_completed = scalars.get("inner_iterations_completed")

    for k in raw_dict.files if hasattr(raw_dict, "files") else raw_dict.keys():
        arr = np.asarray(raw_dict[k])
        if arr.ndim != 1 or arr.size == 0:
            continue

        if epochs_completed is not None and len(arr) >= int(epochs_completed):
            valid_len = int(epochs_completed)
        elif inner_completed is not None and len(arr) >= int(inner_completed) and "inner" in k:
            valid_len = int(inner_completed)
        elif outer_completed is not None and len(arr) >= int(outer_completed) and ("outer" in k or "rho" in k or "buffer" in k):
            valid_len = int(outer_completed)
        else:
            non_nan_indices = np.where(~np.isnan(arr))[0]
            valid_len = int(non_nan_indices[-1] + 1) if len(non_nan_indices) > 0 else 0

        clean_arr = arr[:valid_len].astype(np.float64)
        if len(clean_arr) == 0:
            continue

        if "rho" in k or "outer" in k or k in ("buffer_size", "num_mined_counterexamples"):
            x_label = "Outer Iteration"
        elif inner_completed is not None and len(arr) >= int(inner_completed or 0):
            x_label = "Inner Step"
        else:
            x_label = "Epoch"

        x_values = np.arange(1, len(clean_arr) + 1, dtype=np.int64)
        metrics[k] = MetricSeries(
            name=k,
            values=clean_arr,
            x_values=x_values,
            x_label=x_label,
            is_validation=is_validation,
        )

    return metrics, scalars


def _derive_run_name(path: Path) -> str:
    """Generate a clean human-readable name from a run directory or file path."""
    if path.is_file():
        parent = path.parent
        if parent.name in ("lyapunov", "certification", "data"):
            return f"{parent.parent.name}/{parent.name}"
        return parent.name
    return path.name


def load_metric_run(
    path: Path | str,
    include_val: bool = True,
) -> MetricRun:
    """Load training and optional validation metrics from a file or folder."""
    target_path = Path(path).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Path does not exist: {target_path}")

    run_dir = target_path if target_path.is_dir() else target_path.parent
    run_name = _derive_run_name(target_path)

    train_npz_path: Path | None = None
    val_npz_path: Path | None = None

    if target_path.is_file() and target_path.suffix == ".npz":
        if target_path.name.startswith("val_"):
            val_npz_path = target_path
            potential_train = target_path.parent / target_path.name[4:]
            if potential_train.exists():
                train_npz_path = potential_train
        else:
            train_npz_path = target_path
            potential_val = target_path.parent / f"val_{target_path.name}"
            if include_val and potential_val.exists():
                val_npz_path = potential_val
    else:
        potential_train = run_dir / TRAINING_METRICS_FILENAME
        if potential_train.exists():
            train_npz_path = potential_train
        else:
            npz_files = list(run_dir.glob("*.npz"))
            train_candidates = [f for f in npz_files if not f.name.startswith("val_")]
            if train_candidates:
                train_npz_path = train_candidates[0]

        if include_val:
            potential_val = run_dir / f"val_{TRAINING_METRICS_FILENAME}"
            if potential_val.exists():
                val_npz_path = potential_val

    if train_npz_path is None and val_npz_path is None:
        raise FileNotFoundError(f"No metric .npz files found under {target_path}")

    train_metrics: dict[str, MetricSeries] = {}
    val_metrics: dict[str, MetricSeries] = {}
    scalars: dict[str, Any] = {}

    if train_npz_path is not None and train_npz_path.exists():
        with np.load(train_npz_path, allow_pickle=True) as data:
            train_metrics, scalars = _extract_metric_series(data, is_validation=False)

    if val_npz_path is not None and val_npz_path.exists():
        with np.load(val_npz_path, allow_pickle=True) as data:
            val_metrics, val_scalars = _extract_metric_series(data, is_validation=True)
            for k, v in val_scalars.items():
                if k not in scalars:
                    scalars[k] = v

    config_dict: dict[str, Any] = {}
    config_file = run_dir / TRAINING_CONFIG_FILENAME
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        except Exception as e:
            __logger__.warning("Could not parse config file %s: %s", config_file, e)

    return MetricRun(
        name=run_name,
        path=target_path,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        scalars=scalars,
        config=config_dict,
    )


def discover_metric_runs(
    inputs: Sequence[str | Path],
    include_val: bool = True,
) -> list[MetricRun]:
    """Discover and load metric runs from a list of paths or glob patterns."""
    resolved_paths: list[Path] = []

    for item in inputs:
        item_str = str(item)
        if any(char in item_str for char in ("*", "?", "[")):
            matches = [Path(p) for p in sorted(glob.glob(item_str, recursive=True))]
            resolved_paths.extend(matches)
        else:
            p = Path(item).resolve()
            if p.exists():
                resolved_paths.append(p)
            else:
                __logger__.warning("Path does not exist: %s", item)

    unique_paths: list[Path] = []
    seen = set()
    for p in resolved_paths:
        canonical = p.resolve()
        if canonical not in seen:
            seen.add(canonical)
            unique_paths.append(p)

    runs: list[MetricRun] = []
    for p in unique_paths:
        try:
            run = load_metric_run(p, include_val=include_val)
            runs.append(run)
        except Exception as e:
            __logger__.warning("Failed to load metrics from %s: %s", p, e)

    return runs


def evaluate_metrics_summary(runs: list[MetricRun]) -> str:
    """Generate a clean ASCII summary table of all metrics across runs."""
    if not runs:
        return "No runs loaded."

    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(f"{'TRAINING METRICS EVALUATION SUMMARY':^90}")
    lines.append("=" * 90)

    for run in runs:
        lines.append(f"\n▶ RUN: {run.name}")
        lines.append(f"  Path: {run.path}")
        if run.scalars:
            scalar_str = ", ".join(f"{k}={v}" for k, v in run.scalars.items())
            lines.append(f"  Counters: {scalar_str}")

        all_names = run.get_all_metric_names()
        if not all_names:
            lines.append("  (No metric series found)")
            continue

        header = f"  {'Metric':<28} | {'Type':<6} | {'Steps':<6} | {'Initial':<11} | {'Final':<11} | {'Min (Best)':<11} | {'Max':<11}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for name in all_names:
            if name in run.train_metrics:
                s = run.train_metrics[name]
                init_s = f"{s.initial_val:.4e}" if s.initial_val is not None else "N/A"
                final_s = f"{s.final_val:.4e}" if s.final_val is not None else "N/A"
                min_s = f"{s.min_val:.4e}" if s.min_val is not None else "N/A"
                max_s = f"{s.max_val:.4e}" if s.max_val is not None else "N/A"
                lines.append(
                    f"  {name:<28} | {'Train':<6} | {s.valid_count:<6} | {init_s:<11} | {final_s:<11} | {min_s:<11} | {max_s:<11}"
                )

            if name in run.val_metrics:
                s = run.val_metrics[name]
                init_s = f"{s.initial_val:.4e}" if s.initial_val is not None else "N/A"
                final_s = f"{s.final_val:.4e}" if s.final_val is not None else "N/A"
                min_s = f"{s.min_val:.4e}" if s.min_val is not None else "N/A"
                max_s = f"{s.max_val:.4e}" if s.max_val is not None else "N/A"
                lines.append(
                    f"  {name:<28} | {'Val':<6} | {s.valid_count:<6} | {init_s:<11} | {final_s:<11} | {min_s:<11} | {max_s:<11}"
                )

    lines.append("\n" + "=" * 90)
    return "\n".join(lines)


def filter_metrics(
    metric_names: list[str],
    include_patterns: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> list[str]:
    """Filter metric names based on include/exclude patterns or wildcards."""
    result = list(metric_names)

    if include_patterns:
        filtered = []
        for name in result:
            for pattern in include_patterns:
                regex = re.compile("^" + re.escape(pattern).replace("\\*", ".*").replace("\\?", ".") + "$", re.IGNORECASE)
                if regex.match(name) or pattern.lower() in name.lower():
                    filtered.append(name)
                    break
        result = filtered

    if exclude_patterns:
        filtered = []
        for name in result:
            excluded = False
            for pattern in exclude_patterns:
                regex = re.compile("^" + re.escape(pattern).replace("\\*", ".*").replace("\\?", ".") + "$", re.IGNORECASE)
                if regex.match(name) or pattern.lower() in name.lower():
                    excluded = True
                    break
            if not excluded:
                filtered.append(name)
        result = filtered

    return sorted(list(set(result)))


def _format_metric_title(name: str) -> str:
    """Format raw metric key into a clean publication-ready title."""
    return name.replace("_", " ").title()


def plot_metric_plotly(
    runs: list[MetricRun],
    metric_name: str,
    output_path: Path | None = None,
    log_y: bool = False,
    smooth_weight: float = 0.0,
    title_prefix: str | None = None,
) -> tuple[go.Figure, Path | None]:
    """Generate an individual interactive Plotly HTML figure for a single metric comparing all runs."""
    fig = go.Figure()
    x_label = "Epoch"
    has_data = False

    for i, run in enumerate(runs):
        color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]

        # Training series
        if metric_name in run.train_metrics:
            series = run.train_metrics[metric_name]
            x_label = series.x_label
            x = series.x_values
            y = series.values

            if len(y) > 0:
                has_data = True
                name = f"{run.name} (train)" if run.val_metrics.get(metric_name) is not None else run.name

                if smooth_weight > 0.0:
                    y_smooth = exponential_moving_average(y, weight=smooth_weight)
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            line=dict(color=color, width=1.0),
                            opacity=0.3,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y_smooth,
                            mode="lines",
                            line=dict(color=color, width=2.5),
                            name=name,
                            legendgroup=run.name,
                            hovertemplate=f"<b>{name}</b><br>{x_label}: %{{x}}<br>{metric_name}: %{{y:.6e}}<extra></extra>",
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            line=dict(color=color, width=2.2),
                            name=name,
                            legendgroup=run.name,
                            hovertemplate=f"<b>{name}</b><br>{x_label}: %{{x}}<br>{metric_name}: %{{y:.6e}}<extra></extra>",
                        )
                    )

        # Validation series
        if metric_name in run.val_metrics:
            series = run.val_metrics[metric_name]
            x_label = series.x_label
            x = series.x_values
            y = series.values

            if len(y) > 0:
                has_data = True
                name = f"{run.name} (val)"

                if smooth_weight > 0.0:
                    y_smooth = exponential_moving_average(y, weight=smooth_weight)
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            line=dict(color=color, width=1.0, dash="dot"),
                            opacity=0.3,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y_smooth,
                            mode="lines",
                            line=dict(color=color, width=2.5, dash="dash"),
                            name=name,
                            legendgroup=f"{run.name}_val",
                            hovertemplate=f"<b>{name}</b><br>{x_label}: %{{x}}<br>{metric_name}: %{{y:.6e}}<extra></extra>",
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            line=dict(color=color, width=2.2, dash="dash"),
                            name=name,
                            legendgroup=f"{run.name}_val",
                            hovertemplate=f"<b>{name}</b><br>{x_label}: %{{x}}<br>{metric_name}: %{{y:.6e}}<extra></extra>",
                        )
                    )

    if not has_data:
        return fig, None

    title_text = _format_metric_title(metric_name)
    if title_prefix:
        title_text = f"{title_prefix} - {title_text}"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18, color="#222")),
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=13)),
            showgrid=True,
            gridcolor="#eaeaea",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=metric_name, font=dict(size=13)),
            showgrid=True,
            gridcolor="#eaeaea",
            type="log" if log_y else "linear",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e0e0e0",
            borderwidth=1,
        ),
        margin=dict(l=60, r=40, t=80, b=50),
    )

    saved_file: Path | None = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        saved_file = output_path
        __logger__.info("Saved Plotly HTML plot: %s", output_path)

    return fig, saved_file


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Plotly training metrics evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate and plot interactive Plotly training metrics from single or multiple policy / Lyapunov training runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=str,
        help="One or more directories or .npz metric file paths (supports glob patterns). Defaults to latest results if omitted.",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        type=str,
        default=None,
        help="Filter specific metrics to plot (e.g. 'loss' 'base' 'rho*' 'roa'). If omitted, plots all shared metrics.",
    )
    parser.add_argument(
        "--exclude-metrics",
        nargs="+",
        type=str,
        default=None,
        help="Metrics or patterns to exclude from plotting.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save HTML plots. Defaults to '<first_run_dir>/plots' or 'results/plots/training_metrics'.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Apply logarithmic scale to y-axis for positive loss/error metrics.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Exponential moving average smoothing weight in [0, 1). Default is 0.0 (no smoothing).",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Disable auto-loading paired validation metrics.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title prefix for generated figures.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal summary output table.",
    )
    return parser.parse_args()


def plot_training_metrics_main(args: argparse.Namespace) -> list[Path]:
    """Main execution function for generating Plotly metric plots.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    list[Path]
        List of generated and saved HTML plot files.
    """
    input_paths = list(args.paths)
    if not input_paths:
        results_dir = REPO_ROOT / "results"
        __logger__.info("No paths specified. Searching for recent training runs under %s...", results_dir)
        candidates = list(results_dir.glob("**/training_metrics.npz"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            input_paths = [str(candidates[0].parent)]
            __logger__.info("Discovered latest training run: %s", input_paths[0])
        else:
            raise FileNotFoundError(f"No metric files found under {results_dir}")

    # Load runs
    runs = discover_metric_runs(
        inputs=input_paths,
        include_val=not bool(args.no_val),
    )

    if not runs:
        __logger__.error("No valid metric runs could be loaded from: %s", input_paths)
        return []

    # Print summary table
    if not args.quiet:
        summary_text = evaluate_metrics_summary(runs)
        print(summary_text)

    # Determine target metric names
    all_metrics = set()
    for run in runs:
        all_metrics.update(run.get_all_metric_names())

    target_metrics = filter_metrics(
        metric_names=sorted(list(all_metrics)),
        include_patterns=args.metrics,
        exclude_patterns=args.exclude_metrics,
    )

    if not target_metrics:
        __logger__.warning("No matching metrics found after filtering. Available metrics: %s", sorted(all_metrics))
        return []

    # Determine output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()
    else:
        first_path = runs[0].path
        run_dir = first_path if first_path.is_dir() else first_path.parent
        output_dir = run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    __logger__.info("Output directory set to: %s", output_dir)

    # Generate individual Plotly HTML plot for each metric
    saved_files: list[Path] = []
    for m in target_metrics:
        out_file = output_dir / f"{m}.html"
        _, saved = plot_metric_plotly(
            runs=runs,
            metric_name=m,
            output_path=out_file,
            log_y=bool(args.log_y),
            smooth_weight=float(args.smooth),
            title_prefix=args.title,
        )
        if saved is not None:
            saved_files.append(saved)

    return saved_files


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    plot_training_metrics_main(args)


if __name__ == "__main__":
    main()
