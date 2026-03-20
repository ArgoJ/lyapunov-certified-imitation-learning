import numpy as np
import os
import plotly.graph_objects as go

from numpy.typing import NDArray
from typing import Callable

from mpc_datagen.mpc_data import MPCDataset
from mpc_datagen.plots import lyapunov as _lyapunov
from pkg_logger import get_package_logger

__logger__ = get_package_logger(__name__)


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
    indices: list[int] = [0, 1],
    decimals: int = 12,
) -> tuple[NDArray, NDArray]:
    cert_regs_np = _regions_to_np(cert_regs)
    uncert_regs_np = _regions_to_np(uncert_regs)

    cert_projected = _project_if_needed(cert_regs_np, indices)
    uncert_projected = _project_if_needed(uncert_regs_np, indices)

    cert_unique = _dedupe_regions(cert_projected, decimals)
    uncert_unique = _dedupe_regions(uncert_projected, decimals)

    if cert_unique.shape[0] > 0 and cert_unique.shape[2] == 2 and \
       uncert_unique.shape[0] > 0 and uncert_unique.shape[2] == 2:
        return _make_regions_disjoint_2d(
            cert_unique,
            uncert_unique,
            decimals=decimals,
        )

    return cert_unique, uncert_unique


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
                name=name,
                showlegend=True,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                fill="toself" if fill else None,
                fillcolor=color if fill else None,
                opacity=0.3 if fill else None,
                line=dict(color=color, width=2, dash=dash),
                name=name,
                showlegend=True,
            )
        )

def lyapunov(
    lyapunov_func: Callable[[NDArray], NDArray],
    dataset: MPCDataset | None = None,
    state_indices: list = [0, 1],
    state_labels: list[str] | None = None,
    limits: list = None,
    resolution: int = 100,
    plot_3d: bool = False,
    certified_regions: NDArray | None = None,
    uncertified_regions: NDArray | None = None,
    html_path: str = None,
):
    """Plot Lyapunov landscape, trajectories, and optional certified regions in 2D/3D.
    Only two state dimensions can be visualized at once.

    Parameters
    ----------
    lyapunov_func : Callable[[NDArray], NDArray]
        A function that takes a state vector and returns the Lyapunov value.
    dataset : MPCDataset, optional
        The dataset containing trajectories to plot. If None, only the
        Lyapunov landscape and optional regions are shown. Default is None.
    state_indices : list, optional
        Indices of the two state variables to plot (x, y axes). Default is [0, 1].
    state_labels : list[str], optional
        Labels for the plotted state dimensions. Defaults to ["State i", "State j"].
    limits : list of tuples, optional
        ((min_x, max_x), (min_y, max_y)). If None, inferred from data with padding.
    resolution : int, optional
        Grid resolution for the Lyapunov function contour plot.
    plot_3d : bool, optional
        If True, plot a 3D surface and 3D trajectories. Default is False.
    certified_regions : list of (lb, ub), optional
        Certified boxes in state space overlaid in the same plot.
        Certified regions are drawn with red outlines.
    uncertified_regions : list of (lb, ub), optional
        Uncertified boxes in state space overlaid in the same plot.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.
    """
    if len(state_indices) != 2:
        raise ValueError("state_indices must contain exactly 2 indices.")

    if min(state_indices) < 0:
        raise ValueError("state_indices must be non-negative.")

    certified_regions, uncertified_regions = _collapse_projected_regions(
        certified_regions,
        uncertified_regions,
        indices=state_indices,
    )

    # Determine limits if not provided
    if limits is None:
        min_x = max_x = min_y = max_y = None

        if certified_regions.shape[0] + uncertified_regions.shape[0] > 0:
            all_lbs = np.vstack([certified_regions[:, 0, :], uncertified_regions[:, 0, :]])
            all_ubs = np.vstack([certified_regions[:, 1, :], uncertified_regions[:, 1, :]])

            if min_x is None:
                min_x = all_lbs[:, 0].min()
                max_x = all_ubs[:, 0].max()
                min_y = all_lbs[:, 1].min()
                max_y = all_ubs[:, 1].max()
            else:
                min_x = min(min_x, all_lbs[:, 0].min())
                max_x = max(max_x, all_ubs[:, 0].max())
                min_y = min(min_y, all_lbs[:, 1].min())
                max_y = max(max_y, all_ubs[:, 1].max())

        # Add some padding
        pad_x = (max_x - min_x) * 0.1 if max_x != min_x else 1.0
        pad_y = (max_y - min_y) * 0.1 if max_y != min_y else 1.0

        limits = [
            (min_x - pad_x, max_x + pad_x),
            (min_y - pad_y, max_y + pad_y)
        ]

    fig = _lyapunov(
        lyapunov_func=lyapunov_func,
        dataset=dataset,
        state_indices=state_indices,
        state_labels=state_labels,
        limits=limits,
        resolution=resolution,
        plot_3d=plot_3d,
        use_dataset_v=False
    )

    # === CERTIFIED/UNCERTIFIED REGIONS ===
    z_overlay = 0.0 if plot_3d else None
    _add_regions(
        certified_regions,
        fig,
        color="#209209",
        z_level=z_overlay,
        name="Certified",
        fill=False,
    )
    _add_regions(
        uncertified_regions,
        fig,
        color="#c53131",
        z_level=z_overlay,
        name="Uncertified",
        dash="dash",
        fill=False
    )

    if html_path is not None:
        dir_path = os.path.dirname(html_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        fig.write_html(html_path)
        __logger__.info(f"Trajectories plot saved to {html_path}.")
    else:   
        return fig


def certified_regions_2d(
    certified_regions: NDArray,
    uncertified_regions: NDArray,
    state_indices: list[int] = [0, 1],
    state_labels: list[str] | None = None,
    bounds: list[tuple[float, float]] | None = None,
    html_path: str | None = None,
):
    """Plot certified vs uncertified 2D regions as rectangles.

    Parameters
    ----------
    certified_regions : list of (lb, ub)
        Certified boxes in state space.
    uncertified_regions : list of (lb, ub)
        Uncertified boxes in state space.
    state_indices : list[int], optional
        Indices of the two state dimensions to visualize. If input regions
        contain more than two state dimensions, they are projected to these
        indices before plotting.
    state_labels : list of str, optional
        Axis labels for the two states. Defaults to ["State 0", "State 1"].
    bounds : list of tuples, optional
        ((min_x, max_x), (min_y, max_y)). If None, inferred from regions.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.
    """
    # Validate and possibly project regions to the requested state indices.
    if len(state_indices) != 2:
        raise ValueError("state_indices must contain exactly 2 indices.")
    if min(state_indices) < 0:
        raise ValueError("state_indices must be non-negative.")

    certified_regions, uncertified_regions = _collapse_projected_regions(
        certified_regions,
        uncertified_regions,
        indices=state_indices,
    )

    if certified_regions.shape[0] == 0 and uncertified_regions.shape[0] == 0:
        __logger__.warning("No regions provided for plotting.")
        return

    if state_labels is None:
        state_labels = ["State 0", "State 1"]

    if bounds is None:
        all_lbs = np.vstack([certified_regions[:, 0, :], uncertified_regions[:, 0, :]]) if (certified_regions.shape[0] + uncertified_regions.shape[0]) > 0 else np.empty((0,2))
        all_ubs = np.vstack([certified_regions[:, 1, :], uncertified_regions[:, 1, :]]) if (certified_regions.shape[0] + uncertified_regions.shape[0]) > 0 else np.empty((0,2))
        x_min = all_lbs[:, 0].min()
        x_max = all_ubs[:, 0].max()
        y_min = all_lbs[:, 1].min()
        y_max = all_ubs[:, 1].max()
        bounds = [(x_min, x_max), (y_min, y_max)]

    fig = go.Figure()

    _add_regions(
        certified_regions, 
        fig,
        "#2ca02c", 
        "Certified",
        fill=True,
    )
    _add_regions(
        uncertified_regions, 
        fig,
        "#d62728", 
        "Uncertified",
        fill=True,
    )

    fig.update_layout(
        title="Certified Regions",
        xaxis_title=state_labels[0],
        yaxis_title=state_labels[1],
        xaxis=dict(range=[bounds[0][0], bounds[0][1]]),
        yaxis=dict(range=[bounds[1][0], bounds[1][1]]),
    )

    if html_path is not None:
        dir_path = os.path.dirname(html_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        fig.write_html(html_path)
        __logger__.info("Certified region plot saved to %s.", html_path)
    else:
        fig.show()