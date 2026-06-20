from __future__ import annotations

import numpy as np
import logging
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from pathlib import Path
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Sequence

from mpc_datagen.plots import (
    PairPlotResult,
    _save_pair_figures,
    _resolve_indices,
    _resolve_labels,
    _state_index_pairs,
    _to_latex,
)

from .utils import (
    add_regions,
    collapse_projected_regions,
    limits_from_regions,
    partition_annotation,
    regions_to_np,
)

if TYPE_CHECKING:
    from ...certification.bisect_certifier import RegionCertificationResult

__logger__ = logging.getLogger(__name__)


def certified_regions_2d(
    certification_result: "RegionCertificationResult",
    state_indices: list[int] | None = None,
    state_labels: list[str] | None = None,
    bounds: list[tuple[float, float]] | None = None,
    html_path: Path | str | None = None,
):
    """Plot certified, failed and outside-sublevel regions as 2D overlays.

    If more than two state indices are selected, one figure per 2D state pair
    is generated.

    Parameters
    ----------
    certification_result : RegionCertificationResult
        Full certification output containing certified, failed and outside-sublevel
        region boxes. Outside-sublevel boxes are rendered as a gray overlay.
    state_indices : list[int], optional
        State indices to consider. If None, all states are used and all
        pairwise combinations are plotted.
    state_labels : list of str, optional
        Labels for all state dimensions.
    bounds : list of tuples, optional
        ((min_x, max_x), (min_y, max_y)). If provided, applied to every pair.
    html_path : Path | str, optional
        If provided, saves the plot to the specified HTML file.
    """
    cert_regs_np = regions_to_np(certification_result.certified_sublevel_regions)
    uncert_regs_np = regions_to_np(certification_result.uncertified_regions)
    ctex_regs_np = regions_to_np(certification_result.outside_sublevel_regions)

    if cert_regs_np.shape[0] == 0 and uncert_regs_np.shape[0] == 0 and ctex_regs_np.shape[0] == 0:
        __logger__.warning("No regions provided for plotting.")
        return

    region_dims = []
    if cert_regs_np.shape[0] > 0:
        region_dims.append(int(cert_regs_np.shape[2]))
    if uncert_regs_np.shape[0] > 0:
        region_dims.append(int(uncert_regs_np.shape[2]))
    if ctex_regs_np.shape[0] > 0:
        region_dims.append(int(ctex_regs_np.shape[2]))
    if len(region_dims) == 0:
        raise ValueError("Could not infer state dimension from empty regions.")

    max_state_idx = max(state_indices) if state_indices is not None else -1
    num_states = max(region_dims + [max_state_idx + 1])

    state_indices = _resolve_indices(state_indices, num_states)

    if state_labels is not None and len(state_labels) == len(state_indices):
        labels_full = [f"State {i}" for i in range(num_states)]
        for label_idx, state_idx in enumerate(state_indices):
            labels_full[state_idx] = state_labels[label_idx]
    else:
        labels_full = _resolve_labels(state_labels, num_states)

    pair_indices = _state_index_pairs(state_indices)

    figures: list[PairPlotResult] = []
    for pair in pair_indices:
        labels_pair = (labels_full[pair[0]], labels_full[pair[1]])
        cert_pair, uncert_pair, ctex_pair = collapse_projected_regions(
            cert_regs_np,
            uncert_regs_np,
            ctex_regs_np,
            indices=list(pair),
        )

        if cert_pair.shape[0] == 0 and uncert_pair.shape[0] == 0 and ctex_pair.shape[0] == 0:
            continue

        pair_bounds = bounds
        if pair_bounds is None:
            pair_bounds = limits_from_regions(
                cert_pair,
                uncert_pair,
                ctex_pair,
                pad_ratio=0.0,
                default_pad=0.0,
            )
        if pair_bounds is None:
            pair_bounds = [(-1.0, 1.0), (-1.0, 1.0)]

        fig = go.Figure()

        add_regions(
            cert_pair,
            fig,
            "#2ca02c",
            _to_latex("Certified"),
            fill=True,
        )
        add_regions(
            uncert_pair,
            fig,
            "#d62728",
            _to_latex("Uncertified"),
            fill=True,
        )
        add_regions(
            ctex_pair,
            fig,
            "#808080",
            _to_latex("Outside Sublevel"),
            fill=True,
            alpha=0.5,
        )

        fig.update_layout(
            title=_to_latex(
                f"Certification Partition ({labels_pair[0]} vs {labels_pair[1]})"
            ),
            xaxis_title=_to_latex(labels_pair[0]),
            yaxis_title=_to_latex(labels_pair[1]),
            xaxis=dict(range=[pair_bounds[0][0], pair_bounds[0][1]]),
            yaxis=dict(range=[pair_bounds[1][0], pair_bounds[1][1]]),
        )
        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor="rgba(0,0,0,0.20)",
            borderwidth=1,
            text=_to_latex(partition_annotation(cert_pair, uncert_pair, ctex_pair)),
        )
        figures.append(PairPlotResult(
            idx_x=pair[0],
            idx_y=pair[1],
            label_x=labels_pair[0],
            label_y=labels_pair[1],
            figure=fig,
        ))

    if len(figures) == 0:
        __logger__.warning("No projectable regions found for selected state indices.")
        return

    if html_path is not None:
        _save_pair_figures(figures, html_path, kind="Certified region")
        return

    return figures


def parallel_coordinates(
    states: NDArray,
    state_bounds: NDArray | None = None,
    state_labels: Sequence[str] | None = None,
    state_order: Sequence[int] | None = None,
    max_lines: int = 64,
):
    x = np.asarray(states, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("states must have shape (N, nx) with N > 0.")

    n, d = x.shape

    if state_order is None:
        order = np.arange(d)
    else:
        order = np.asarray(state_order, dtype=np.int64)
        if order.shape != (d,):
            raise ValueError(f"state_order must have shape ({d},), got {order.shape}.")
        if np.unique(order).shape[0] != d:
            raise ValueError("state_order must be a permutation of state indices.")

    x = x[:, order]

    if n > max_lines:
        idx = np.linspace(0, n - 1, num=max_lines, dtype=int)
        x = x[idx]
        n = x.shape[0]

    if state_labels is None:
        labels = [f"x{i}" for i in range(d)]
    else:
        labels = list(state_labels)
        if len(labels) != d:
            raise ValueError(f"state_labels must have length {d}, got {len(labels)}.")
    labels = [labels[i] for i in order]

    if state_bounds is not None:
        bounds = np.asarray(state_bounds, dtype=np.float32)
        if bounds.shape != (2, d):
            raise ValueError(f"state_bounds must have shape (2, {d}).")
        bounds = bounds[:, order]
        lb, ub = bounds
        center = 0.5 * (lb + ub)
        half = 0.5 * (ub - lb)
        half = np.where(half > 1e-8, half, 1.0)
        x = (x - center[None, :]) / half[None, :]
        ylim = (-1.1, 1.1)
    else:
        mins = x.min(axis=0)
        maxs = x.max(axis=0)
        span = np.where((maxs - mins) > 1e-8, maxs - mins, 1.0)
        x = 2.0 * (x - mins[None, :]) / span[None, :] - 1.0
        ylim = (-1.1, 1.1)

    xs = np.arange(d)

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * d), 3.0), constrained_layout=True)

    for i in range(n):
        ax.plot(xs, x[i], color="tab:red", alpha=0.15, linewidth=0.8)

    for xi in xs:
        ax.axvline(xi, color="0.85", linewidth=0.8, zorder=0)

    ax.axhline(-1.0, color="0.6", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="0.2", linestyle="-", linewidth=0.8)
    ax.axhline(1.0, color="0.6", linestyle="--", linewidth=0.8)

    ax.set_xticks(xs, labels=labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel("normalized state")
    ax.set_title(f"Counterexamples (n={n})")

    return fig