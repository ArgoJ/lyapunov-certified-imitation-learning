from __future__ import annotations

import numpy as np
import logging
import plotly.graph_objects as go

from numpy.typing import NDArray
from typing import Callable, TYPE_CHECKING

from mpc_datagen.mpc_data import MPCDataset
from mpc_datagen.plots import (
    lyapunov, 
    _save_pair_figures,
    _resolve_indices,
    _resolve_labels,
    _state_index_pairs,
    _to_latex,
)

if TYPE_CHECKING:
    from ..certification.bisect_certifier import RegionCertificationResult

__logger__ = logging.getLogger(__name__)


def _regions_to_np(regs: NDArray | None) -> NDArray:
    if regs is None or regs.shape[0] == 0:
        return np.empty((0, 2, 2))
    return np.asarray(regs, dtype=np.float64)


def _plotly_multiline(x: NDArray, axis: int=0) -> NDArray:
    if axis == 0:
        return np.hstack([x, np.full((x.shape[0], 1), np.nan)]).flatten()
    elif axis == 1:
        return np.vstack([x, np.full((x.shape[1], 1), np.nan)]).flatten()


def _project_if_needed(regs: NDArray, indices: list[int]) -> NDArray:
    if regs is None or regs.shape[0] == 0:
        return regs
    if regs.shape[2] > 2:
        return regs[:, :, [indices[0], indices[1]]]
    return regs


def _dedupe_regions(regs: NDArray, decimals: int = 12) -> NDArray:
    regs = np.asarray(regs, dtype=np.float64)
    if regs.size == 0:
        return np.empty((0, 2, 2), dtype=np.float64)

    rounded_flat = np.round(regs, decimals=decimals).reshape(regs.shape[0], -1)
    _, unique_indices = np.unique(rounded_flat, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return regs[unique_indices]


def _collapse_projected_regions(
    cert_regs: NDArray,
    uncert_regs: NDArray,
    ctex_regs: NDArray,
    indices: list[int] = [0, 1],
    decimals: int = 12,
) -> tuple[NDArray, NDArray, NDArray]:
    cert_regs_np = _regions_to_np(cert_regs)
    uncert_regs_np = _regions_to_np(uncert_regs)
    ctex_regs_np = _regions_to_np(ctex_regs)

    cert_projected = _project_if_needed(cert_regs_np, indices)
    uncert_projected = _project_if_needed(uncert_regs_np, indices)
    ctex_projected = _project_if_needed(ctex_regs_np, indices)

    cert_unique = _dedupe_regions(cert_projected, decimals)
    uncert_unique = _dedupe_regions(uncert_projected, decimals)
    ctex_unique = _dedupe_regions(ctex_projected, decimals)

    has_2d_projection = (
        (cert_unique.shape[0] == 0 or cert_unique.shape[2] == 2)
        and (uncert_unique.shape[0] == 0 or uncert_unique.shape[2] == 2)
        and (ctex_unique.shape[0] == 0 or ctex_unique.shape[2] == 2)
    )
    if has_2d_projection:
        cert_unique, uncert_unique, ctex_unique = _make_regions_disjoint_3way_2d(
            cert_unique,
            uncert_unique,
            ctex_unique,
            decimals=decimals,
        )

    return cert_unique, uncert_unique, ctex_unique


def _limits_from_regions(
    cert_regs: NDArray,
    uncert_regs: NDArray,
    ctex_regs: NDArray | None = None,
    pad_ratio: float = 0.0,
    default_pad: float = 1.0,
) -> list[tuple[float, float]] | None:
    cert_regs_np = _regions_to_np(cert_regs)
    uncert_regs_np = _regions_to_np(uncert_regs)
    ctex_regs_np = _regions_to_np(ctex_regs)

    if cert_regs_np.shape[0] + uncert_regs_np.shape[0] + ctex_regs_np.shape[0] == 0:
        return None

    lb_parts = [arr[:, 0, :] for arr in (cert_regs_np, uncert_regs_np, ctex_regs_np) if arr.shape[0] > 0]
    ub_parts = [arr[:, 1, :] for arr in (cert_regs_np, uncert_regs_np, ctex_regs_np) if arr.shape[0] > 0]

    all_lbs = np.vstack(lb_parts)
    all_ubs = np.vstack(ub_parts)

    min_x = all_lbs[:, 0].min()
    max_x = all_ubs[:, 0].max()
    min_y = all_lbs[:, 1].min()
    max_y = all_ubs[:, 1].max()

    span_x = max_x - min_x
    span_y = max_y - min_y
    pad_x = span_x * pad_ratio if span_x != 0 else default_pad
    pad_y = span_y * pad_ratio if span_y != 0 else default_pad

    return [
        (min_x - pad_x, max_x + pad_x),
        (min_y - pad_y, max_y + pad_y),
    ]


def _make_regions_disjoint_2d(
    cert_regs: NDArray,
    uncert_regs: NDArray,
    decimals: int = 12,
) -> tuple[NDArray, NDArray]:
    """Make two 2D rectangle sets disjoint via grid-cell partitioning.

    The returned sets are non-overlapping in 2D. If both classes cover the
    same projected cell, the cell is assigned to ``uncertified``.
    """
    cert_regs = _regions_to_np(cert_regs)
    uncert_regs = _regions_to_np(uncert_regs)

    if cert_regs.shape[0] == 0 and uncert_regs.shape[0] == 0:
        return cert_regs, uncert_regs

    if cert_regs.shape[0] > 0 and cert_regs.shape[2] != 2:
        return cert_regs, uncert_regs
    if uncert_regs.shape[0] > 0 and uncert_regs.shape[2] != 2:
        return cert_regs, uncert_regs

    def _boundaries(regs_a: NDArray, regs_b: NDArray, axis: int) -> NDArray:
        vals = []
        if regs_a.shape[0] > 0:
            vals.extend([regs_a[:, 0, axis], regs_a[:, 1, axis]])
        if regs_b.shape[0] > 0:
            vals.extend([regs_b[:, 0, axis], regs_b[:, 1, axis]])
        if len(vals) == 0:
            return np.empty((0,), dtype=np.float64)
        merged = np.concatenate(vals)
        return np.unique(np.round(merged, decimals=decimals))

    x_edges = _boundaries(cert_regs, uncert_regs, axis=0)
    y_edges = _boundaries(cert_regs, uncert_regs, axis=1)

    if x_edges.shape[0] < 2 or y_edges.shape[0] < 2:
        return cert_regs, uncert_regs

    x_l = x_edges[:-1]
    x_u = x_edges[1:]
    y_l = y_edges[:-1]
    y_u = y_edges[1:]

    def _coverage(regs: NDArray) -> NDArray:
        nx, ny = len(x_l), len(y_l)
        if regs.shape[0] == 0:
            return np.zeros((nx, ny), dtype=bool)

        lbs = np.round(regs[:, 0, :], decimals=decimals)
        ubs = np.round(regs[:, 1, :], decimals=decimals)

        ix_min = np.searchsorted(x_edges, lbs[:, 0])
        ix_max = np.searchsorted(x_edges, ubs[:, 0])
        iy_min = np.searchsorted(y_edges, lbs[:, 1])
        iy_max = np.searchsorted(y_edges, ubs[:, 1])

        diff = np.zeros((nx + 1, ny + 1), dtype=np.int32)
        
        np.add.at(diff, (ix_min, iy_min), 1)
        np.add.at(diff, (ix_max, iy_max), 1)
        np.add.at(diff, (ix_min, iy_max), -1)
        np.add.at(diff, (ix_max, iy_min), -1)

        coverage = np.cumsum(np.cumsum(diff, axis=0), axis=1)[:-1, :-1] > 0
        return coverage

    cert_cover = _coverage(cert_regs)
    uncert_cover = _coverage(uncert_regs)

    cert_only = cert_cover & ~uncert_cover
    uncert_final = uncert_cover

    def _cells_to_regions(mask: NDArray) -> NDArray:
        if not np.any(mask):
            return np.empty((0, 2, 2), dtype=np.float64)
        x_idx, y_idx = np.nonzero(mask)
        lbs = np.column_stack([x_l[x_idx], y_l[y_idx]])
        ubs = np.column_stack([x_u[x_idx], y_u[y_idx]])
        return np.stack([lbs, ubs], axis=1)

    cert_disjoint = _dedupe_regions(_cells_to_regions(cert_only), decimals=decimals)
    uncert_disjoint = _dedupe_regions(_cells_to_regions(uncert_final), decimals=decimals)

    return cert_disjoint, uncert_disjoint


def _make_regions_disjoint_3way_2d(
    cert_regs: NDArray,
    uncert_regs: NDArray,
    ctex_regs: NDArray,
    decimals: int = 12,
) -> tuple[NDArray, NDArray, NDArray]:
    """Make certified, uncertified and outside-sublevel regions disjoint in 2D.

    Priority for overlapping projected cells is:
    1) uncertified
    2) outside-sublevel (IBP-pruned)
    3) certified
    """
    cert_regs = _regions_to_np(cert_regs)
    uncert_regs = _regions_to_np(uncert_regs)
    ctex_regs = _regions_to_np(ctex_regs)

    if cert_regs.shape[0] == 0 and uncert_regs.shape[0] == 0 and ctex_regs.shape[0] == 0:
        return cert_regs, uncert_regs, ctex_regs

    if cert_regs.shape[0] > 0 and cert_regs.shape[2] != 2:
        return cert_regs, uncert_regs, ctex_regs
    if uncert_regs.shape[0] > 0 and uncert_regs.shape[2] != 2:
        return cert_regs, uncert_regs, ctex_regs
    if ctex_regs.shape[0] > 0 and ctex_regs.shape[2] != 2:
        return cert_regs, uncert_regs, ctex_regs

    def _boundaries(regs_a: NDArray, regs_b: NDArray, regs_c: NDArray, axis: int) -> NDArray:
        vals = []
        for regs in (regs_a, regs_b, regs_c):
            if regs.shape[0] > 0:
                vals.extend([regs[:, 0, axis], regs[:, 1, axis]])
        if len(vals) == 0:
            return np.empty((0,), dtype=np.float64)
        merged = np.concatenate(vals)
        return np.unique(np.round(merged, decimals=decimals))

    x_edges = _boundaries(cert_regs, uncert_regs, ctex_regs, axis=0)
    y_edges = _boundaries(cert_regs, uncert_regs, ctex_regs, axis=1)

    if x_edges.shape[0] < 2 or y_edges.shape[0] < 2:
        return cert_regs, uncert_regs, ctex_regs

    x_l = x_edges[:-1]
    x_u = x_edges[1:]
    y_l = y_edges[:-1]
    y_u = y_edges[1:]

    def _coverage(regs: NDArray) -> NDArray:
        nx, ny = len(x_l), len(y_l)
        if regs.shape[0] == 0:
            return np.zeros((nx, ny), dtype=bool)

        lbs = np.round(regs[:, 0, :], decimals=decimals)
        ubs = np.round(regs[:, 1, :], decimals=decimals)

        ix_min = np.searchsorted(x_edges, lbs[:, 0])
        ix_max = np.searchsorted(x_edges, ubs[:, 0])
        iy_min = np.searchsorted(y_edges, lbs[:, 1])
        iy_max = np.searchsorted(y_edges, ubs[:, 1])

        diff = np.zeros((nx + 1, ny + 1), dtype=np.int32)

        np.add.at(diff, (ix_min, iy_min), 1)
        np.add.at(diff, (ix_max, iy_max), 1)
        np.add.at(diff, (ix_min, iy_max), -1)
        np.add.at(diff, (ix_max, iy_min), -1)

        return np.cumsum(np.cumsum(diff, axis=0), axis=1)[:-1, :-1] > 0

    cert_cover = _coverage(cert_regs)
    uncert_cover = _coverage(uncert_regs)
    ctex_cover = _coverage(ctex_regs)

    uncert_mask = uncert_cover
    ctex_mask = ctex_cover & ~uncert_mask
    cert_mask = cert_cover & ~uncert_mask & ~ctex_mask

    def _cells_to_regions(mask: NDArray) -> NDArray:
        if not np.any(mask):
            return np.empty((0, 2, 2), dtype=np.float64)
        x_idx, y_idx = np.nonzero(mask)
        lbs = np.column_stack([x_l[x_idx], y_l[y_idx]])
        ubs = np.column_stack([x_u[x_idx], y_u[y_idx]])
        return np.stack([lbs, ubs], axis=1)

    cert_disjoint = _dedupe_regions(_cells_to_regions(cert_mask), decimals=decimals)
    uncert_disjoint = _dedupe_regions(_cells_to_regions(uncert_mask), decimals=decimals)
    ctex_disjoint = _dedupe_regions(_cells_to_regions(ctex_mask), decimals=decimals)

    return cert_disjoint, uncert_disjoint, ctex_disjoint


def _regions_area_2d(regions: NDArray) -> float:
    regs = _regions_to_np(regions)
    if regs.shape[0] == 0:
        return 0.0
    widths = np.maximum(0.0, regs[:, 1, 0] - regs[:, 0, 0])
    heights = np.maximum(0.0, regs[:, 1, 1] - regs[:, 0, 1])
    return float(np.sum(widths * heights))


def _partition_annotation(cert_regs: NDArray, uncert_regs: NDArray, ctex_regs: NDArray) -> str:
    cert_area = _regions_area_2d(cert_regs)
    uncert_area = _regions_area_2d(uncert_regs)
    ctex_area = _regions_area_2d(ctex_regs)
    total = cert_area + uncert_area + ctex_area

    if total <= 0.0:
        cert_ratio = uncert_ratio = ctex_ratio = 0.0
    else:
        cert_ratio = cert_area / total
        uncert_ratio = uncert_area / total
        ctex_ratio = ctex_area / total

    return (
        "Projected partition: "
        f"certified={cert_ratio:.1%}, "
        f"failed={uncert_ratio:.1%}, "
        f"outside V<=rho={ctex_ratio:.1%}<br>"
        f"Box count: cert={cert_regs.shape[0]}, "
        f"failed={uncert_regs.shape[0]}, "
        f"outside={ctex_regs.shape[0]}"
    )


def _build_X_Y(regions: NDArray) -> tuple[NDArray, NDArray]:
    lbs = regions[:, 0, :]
    ubs = regions[:, 1, :]

    X = np.column_stack([lbs[:, 0], ubs[:, 0], ubs[:, 0], lbs[:, 0], lbs[:, 0]])
    Y = np.column_stack([lbs[:, 1], lbs[:, 1], ubs[:, 1], ubs[:, 1], lbs[:, 1]])
    return X, Y


def _add_regions(
    regions: NDArray | None,
    fig: go.Figure,
    color: str,
    name: str,
    z_level: float | None = None,
    dash: str = "solid",
    fill: bool = False,
    alpha: float | None = None,
) -> None:
    if regions is None or regions.shape[0] == 0:
        return

    X, Y = _build_X_Y(regions)

    x_coords = _plotly_multiline(X, axis=0)
    y_coords = _plotly_multiline(Y, axis=0)

    if z_level is not None:
        Z = np.full(X.shape, z_level)
        z_coords = _plotly_multiline(Z, axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="lines",
                line=dict(color=color, width=4, dash=dash),
                opacity=alpha,
                name=name,
                showlegend=True,
            )
        )
    else:
        fig.add_trace(
            go.Scattergl(
                x=x_coords,
                y=y_coords,
                mode="lines",
                fill="toself" if fill else None,
                fillcolor=color if fill else None,
                opacity=alpha if alpha is not None else (0.3 if fill else None),
                line=dict(color=color, width=2, dash=dash),
                name=name,
                showlegend=True,
            )
        )

def lyapunov_cert_regions(
    lyapunov_func: Callable[[NDArray], NDArray],
    certification_result: "RegionCertificationResult",
    dataset: MPCDataset | None = None,
    state_indices: list[int] | None = None,
    state_labels: list[str] | None = None,
    limits: list = None,
    resolution: int = 100,
    plot_3d: bool = False,
    html_path: str = None,
):
    """Plot Lyapunov landscape, trajectories, and optional certified regions in 2D/3D.
    If more than two state indices are provided, one figure per 2D state pair
    is generated.

    Parameters
    ----------
    lyapunov_func : Callable[[NDArray], NDArray]
        A function that takes a state vector and returns the Lyapunov value.
    certification_result : RegionCertificationResult
        Full certification output containing certified, failed and outside-sublevel
        region boxes. The outside-sublevel boxes are rendered as a gray overlay.
    dataset : MPCDataset, optional
        The dataset containing trajectories to plot. If None, only the
        Lyapunov landscape and optional regions are shown. Default is None.
    state_indices : list[int], optional
        State indices to consider. If None, all states are used and all pairwise
        combinations are plotted.
    state_labels : list[str], optional
        Labels for the plotted state dimensions. Defaults to ["State i", "State j"].
    limits : list of tuples, optional
        ((min_x, max_x), (min_y, max_y)). If None, inferred from data with padding.
    resolution : int, optional
        Grid resolution for the Lyapunov function contour plot.
    plot_3d : bool, optional
        If True, plot a 3D surface and 3D trajectories. Default is False.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.
    """
    cert_regs_np = _regions_to_np(certification_result.certified_regions)
    uncert_regs_np = _regions_to_np(certification_result.failed_regions)
    outside_sublevel_regs_np = _regions_to_np(certification_result.outside_sublevel_regions)

    has_dataset = dataset is not None and len(dataset) > 0
    if has_dataset:
        num_states = int(dataset[0].trajectory.states.shape[1])
    else:
        region_dims = []
        if cert_regs_np.shape[0] > 0:
            region_dims.append(int(cert_regs_np.shape[2]))
        if uncert_regs_np.shape[0] > 0:
            region_dims.append(int(uncert_regs_np.shape[2]))
        if outside_sublevel_regs_np.shape[0] > 0:
            region_dims.append(int(outside_sublevel_regs_np.shape[2]))

        if len(region_dims) == 0 and state_indices is None:
            raise ValueError(
                "Without dataset or non-empty regions, state_indices must be provided."
            )

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

    regions_by_pair: dict[tuple[int, int], tuple[NDArray, NDArray, NDArray]] = {}
    pair_limits: dict[tuple[int, int], list[tuple[float, float]]] = {}

    for pair in pair_indices:
        cert_pair, uncert_pair, ctex_pair = _collapse_projected_regions(
            cert_regs_np,
            uncert_regs_np,
            outside_sublevel_regs_np,
            indices=list(pair),
        )
        regions_by_pair[pair] = (cert_pair, uncert_pair, ctex_pair)

        if limits is not None:
            pair_limits[pair] = limits
            continue

        inferred_limits = _limits_from_regions(
            cert_pair,
            uncert_pair,
            ctex_pair,
            pad_ratio=0.1,
            default_pad=1.0,
        )
        if inferred_limits is None:
            inferred_limits = [(-1.0, 1.0), (-1.0, 1.0)]

        pair_limits[pair] = inferred_limits

    z_overlay = 0.0 if plot_3d else None

    figures: dict[tuple[int, int], go.Figure] = {}
    for pair in pair_indices:
        fig_pair = lyapunov(
            lyapunov_func=lyapunov_func,
            dataset=dataset,
            state_indices=list(pair),
            state_labels=labels_full,
            limits=pair_limits[pair],
            resolution=resolution,
            plot_3d=plot_3d,
            use_dataset_v=False,
        )

        if isinstance(fig_pair, dict):
            fig = fig_pair[pair]
        else:
            fig = fig_pair

        cert_pair, uncert_pair, ctex_pair = regions_by_pair[pair]
        _add_regions(
            cert_pair,
            fig,
            color="#209209",
            z_level=z_overlay,
            name="Certified",
            fill=False,
        )
        _add_regions(
            uncert_pair,
            fig,
            color="#c53131",
            z_level=z_overlay,
            name="Uncertified",
            dash="dash",
            fill=False,
        )
        _add_regions(
            ctex_pair,
            fig,
            color="#808080",
            z_level=z_overlay,
            name="Outside Sublevel",
            fill=False,
            alpha=0.5,
        )

        if z_overlay is None:
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
                text=_partition_annotation(cert_pair, uncert_pair, ctex_pair),
            )
        figures[pair] = fig

    if html_path is not None:
        _save_pair_figures(figures, html_path, labels_full, kind="Lyapunov")
        return None

    if len(figures) == 1:
        return next(iter(figures.values()))
    return figures


def certified_regions_2d(
    certification_result: "RegionCertificationResult",
    state_indices: list[int] | None = None,
    state_labels: list[str] | None = None,
    bounds: list[tuple[float, float]] | None = None,
    html_path: str | None = None,
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
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.
    """
    cert_regs_np = _regions_to_np(certification_result.certified_regions)
    uncert_regs_np = _regions_to_np(certification_result.failed_regions)
    ctex_regs_np = _regions_to_np(certification_result.outside_sublevel_regions)

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

    figures: dict[tuple[int, int], go.Figure] = {}
    for pair in pair_indices:
        cert_pair, uncert_pair, ctex_pair = _collapse_projected_regions(
            cert_regs_np,
            uncert_regs_np,
            ctex_regs_np,
            indices=list(pair),
        )

        if cert_pair.shape[0] == 0 and uncert_pair.shape[0] == 0 and ctex_pair.shape[0] == 0:
            continue

        pair_bounds = bounds
        if pair_bounds is None:
            pair_bounds = _limits_from_regions(
                cert_pair,
                uncert_pair,
                ctex_pair,
                pad_ratio=0.0,
                default_pad=0.0,
            )
        if pair_bounds is None:
            pair_bounds = [(-1.0, 1.0), (-1.0, 1.0)]

        fig = go.Figure()

        _add_regions(
            cert_pair,
            fig,
            "#2ca02c",
            _to_latex("Certified"),
            fill=True,
        )
        _add_regions(
            uncert_pair,
            fig,
            "#d62728",
            _to_latex("Uncertified"),
            fill=True,
        )
        _add_regions(
            ctex_pair,
            fig,
            "#808080",
            _to_latex("Outside Sublevel"),
            fill=True,
            alpha=0.5,
        )

        fig.update_layout(
            title=_to_latex(
                f"Certification Partition ({labels_full[pair[0]]} vs {labels_full[pair[1]]})"
            ),
            xaxis_title=_to_latex(labels_full[pair[0]]),
            yaxis_title=_to_latex(labels_full[pair[1]]),
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
            text=_to_latex(_partition_annotation(cert_pair, uncert_pair, ctex_pair)),
        )
        figures[pair] = fig

    if len(figures) == 0:
        __logger__.warning("No projectable regions found for selected state indices.")
        return

    if html_path is not None:
        _save_pair_figures(figures, html_path, labels_full, kind="Certified region")
        return

    if len(figures) == 1:
        return next(iter(figures.values()))
    return figures