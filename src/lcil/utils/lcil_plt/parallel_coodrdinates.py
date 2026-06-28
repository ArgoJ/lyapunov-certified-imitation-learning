from __future__ import annotations

import numpy as np
import logging
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from pathlib import Path
from numpy.typing import NDArray
from typing import Sequence

from mpc_datagen.plots import (
    _resolve_labels,
    _to_latex,
    _handle_figure_output,
)
from .styles import *

__logger__ = logging.getLogger(__name__)


def _normalize_states(x: NDArray, state_bounds: NDArray) -> NDArray:
    """Normalisiert die Daten auf einen Bereich von ca. -1.0 bis 1.0."""
    bounds = np.asarray(state_bounds, dtype=np.float32)
    if bounds.shape != (2, x.shape[1]):
        raise ValueError(f"state_bounds must have shape (2, {x.shape[1]}).")
        
    lb, ub = bounds
    center = 0.5 * (lb + ub)
    half = 0.5 * (ub - lb)
    half = np.where(half > 1e-8, half, 1.0)
    
    return (x - center[None, :]) / half[None, :]


def _prepare_parallel_data(
    states: NDArray,
    state_bounds: NDArray,
    state_labels: Sequence[str] | None,
    state_order: Sequence[int] | None,
    origin_exclusion: Sequence[float] | None,
    max_lines: int | None,
) -> tuple[NDArray, list[str], int, int, int]:
    """Zentrale Funktion zur Vorbereitung (Validierung, Downsampling, Normalisierung) der Daten."""
    x = np.asarray(states, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("states must have shape (N, nx) with N > 0.")

    n_total, d = x.shape

    # Downsampling
    if max_lines is not None and n_total > max_lines:
        idx = np.linspace(0, n_total - 1, num=max_lines, dtype=int)
        x = x[idx]
    n = x.shape[0]

    # resolve labels
    labels = _resolve_labels(state_labels, d)
    
    # State sorting
    bounds_ordered = state_bounds
    if state_order is not None:
        order = np.asarray(state_order, dtype=np.int64)
        if order.shape != (d,) or np.unique(order).shape[0] != d:
            raise ValueError("state_order must be a valid permutation of state indices.")
        
        x = x[:, order]
        labels = [labels[i] for i in order]
        if state_bounds is not None:
            bounds_ordered = np.asarray(state_bounds, dtype=np.float32)[:, order]

    x_norm = _normalize_states(x, bounds_ordered)
    
    origin_exc_bounds = None
    if origin_exclusion is not None:
        eps = np.asarray(origin_exclusion, dtype=np.float32)
        if eps.ndim == 0:
            eps = np.full(d, eps)
        
        upper_norm = _normalize_states(np.array([eps]), bounds_ordered)[0]
        lower_norm = _normalize_states(np.array([-eps]), bounds_ordered)[0]
        origin_exc_bounds = (lower_norm, upper_norm)
        
    return x_norm, origin_exc_bounds, labels, n_total, n, d


# --- MATPLOTLIB IMPLEMENTIERUNG ---
def parallel_coordinates_matplot(
    states: NDArray,
    state_bounds: NDArray | None = None,
    state_labels: Sequence[str] | None = None,
    state_order: Sequence[int] | None = None,
    origin_exclusion: float | Sequence[float] | None = None,
    max_lines: int = 64,
):
    x_norm, origin_exc, labels, _, n, d = _prepare_parallel_data(
        states, state_bounds, state_labels, state_order, origin_exclusion, max_lines
    )

    xs = np.arange(d)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * d), 3.0), constrained_layout=True)

    for i in range(n):
        ax.plot(xs, x_norm[i], color="tab:red", alpha=0.15, linewidth=0.8)

    for xi in xs:
        ax.axvline(xi, color="0.85", linewidth=0.8, zorder=0)

    if origin_exc is not None:
        lower_norm, upper_norm = origin_exc
        for i, xi in enumerate(xs):
            ax.plot(
                [xi, xi], 
                [lower_norm[i], upper_norm[i]], 
                color="tab:gray", 
                linewidth=8, 
                alpha=0.6, 
                zorder=2, 
                solid_capstyle="butt"
            )

    ax.axhline(-1.0, color="0.6", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="0.2", linestyle="-", linewidth=0.8)
    ax.axhline(1.0, color="0.6", linestyle="--", linewidth=0.8)

    ax.set_xticks(xs, labels=[_to_latex(lbl) for lbl in labels])
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("normalized state")
    ax.set_title(_to_latex(f"Counterexamples (n={n})"))

    return fig


# --- PLOTLY IMPLEMENTIERUNG ---
def parallel_coordinates_plotly(
    states: NDArray,
    state_bounds: NDArray,
    state_labels: Sequence[str] | None = None,
    state_order: Sequence[int] | None = None,
    max_lines: int | None = None,
    line_color: str = STYLE_UNCERTIFIED,
    title: str | None = None,
    annotation_text: str | None = None,
    origin_exclusion: float | Sequence[float] | None = None,
    html_path: Path | None = None,
) -> go.Figure | None:

    x_norm, origin_exc, labels, n_total, n, d = _prepare_parallel_data(
        states, state_bounds, state_labels, state_order, origin_exclusion, max_lines
    )

    dimensions = []
    for i in range(d):
        dimensions.append(dict(
            range=[-1.1, 1.1],
            label=_to_latex(labels[i]),
            values=x_norm[:, i],
            tickvals=[-1.0, 0.0, 1.0],
            ticktext=[_to_latex("-1"), _to_latex("0"), _to_latex("+1")],
        ))

    # Text annotation
    title = title or f"Counterexamples (n={n})"
    if annotation_text is None:
        sampled_note = f"Showing {n}/{n_total} lines" if n < n_total else f"Showing all {n} lines"
        lines = ["Parallel coordinates", sampled_note]
        latex_annotation = "<br>".join(_to_latex(line) for line in lines)
    else:
        latex_annotation = "<br>".join(_to_latex(line) for line in annotation_text.split("<br>"))

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=line_color),
        dimensions=dimensions,
        labelangle=-45,
    ))

    fig.update_layout(
        title=_to_latex(title),
        template="plotly_white",
        width=max(700, 170 * d),
        height=340,
        margin=dict(l=60, r=20, t=60, b=70),
    )

    fig.add_annotation(
        x=0.01, y=0.99,
        xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        align="left", showarrow=False,
        bgcolor=STYLE_ANNOTATION_BG,
        bordercolor=STYLE_ANNOTATION_BORDER,
        borderwidth=1,
        text=latex_annotation,
    )

    if origin_exc is not None:
        lower_norm, upper_norm = origin_exc
        y_min, y_max = -1.1, 1.1
        y_range = y_max - y_min
        
        for i in range(d):
            y0_norm = (lower_norm[i] - y_min) / y_range
            y1_norm = (upper_norm[i] - y_min) / y_range
            x_pos = i / (d - 1) if d > 1 else 0.5
            
            fig.add_shape(
                type="line",
                xref="paper", 
                yref="paper",
                x0=x_pos, y0=y0_norm,
                x1=x_pos, y1=y1_norm,
                line=dict(
                    color="gray",
                    width=8, # Entspricht der Matplotlib linewidth=8
                ),
                opacity=0.6,
                layer="below" # Hinter den Parcoord-Linien rendern
            )

    return _handle_figure_output(fig, html_path, kind="Parallel Coordinates")