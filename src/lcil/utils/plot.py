import numpy as np
import os
import plotly.graph_objects as go

from numpy.typing import NDArray
from typing import Callable

from mpc_datagen.mpc_data import MPCDataset
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

    if cert_unique.shape[0] == 0 or uncert_unique.shape[0] == 0:
        return cert_unique, uncert_unique

    cert_lbs = cert_unique[:, 0, :]
    cert_ubs = cert_unique[:, 1, :]
    uncert_lbs = uncert_unique[:, 0, :]
    uncert_ubs = uncert_unique[:, 1, :]

    # certified[i] contains uncertified[j] iff
    # cert_lb <= uncert_lb and cert_ub >= uncert_ub (component-wise).
    contains_matrix = (
        (cert_lbs[:, None, :] <= uncert_lbs[None, :, :])
        & (cert_ubs[:, None, :] >= uncert_ubs[None, :, :])
    ).all(axis=2)

    # Remove any certified region that contains at least one uncertified region.
    keep_cert_mask = ~contains_matrix.any(axis=1)
    cert_filtered = cert_unique[keep_cert_mask]

    return cert_filtered, uncert_unique


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

    has_dataset = dataset is not None and len(dataset) > 0

    # Infer state dimension from dataset if present, otherwise from indices.
    if has_dataset:
        first_traj = dataset[0].trajectory
        num_states = first_traj.states.shape[1]
    else:
        num_states = max(state_indices) + 1

    idx_x, idx_y = state_indices

    if idx_x >= num_states or idx_y >= num_states:
        raise ValueError(
            f"state_indices {state_indices} exceed inferred state dimension {num_states}."
        )

    if state_labels is None:
        state_labels = [f"State {idx_x}", f"State {idx_y}"]
    if len(state_labels) != 2:
        raise ValueError("state_labels must contain exactly 2 labels.")

    certified_regions, uncertified_regions = _collapse_projected_regions(
        certified_regions,
        uncertified_regions,
        indices=state_indices,
    )

    # Determine limits if not provided
    if limits is None:
        min_x = max_x = min_y = max_y = None

        if has_dataset:
            all_states = np.vstack([d.trajectory.states for d in dataset])
            min_x = all_states[:, idx_x].min()
            max_x = all_states[:, idx_x].max()
            min_y = all_states[:, idx_y].min()
            max_y = all_states[:, idx_y].max()

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

        if min_x is None:
            __logger__.warning(
                "Could not infer limits without dataset/regions. Falling back to [-1, 1]^2."
            )
            limits = [(-1.0, 1.0), (-1.0, 1.0)]
        else:
            # Add some padding
            pad_x = (max_x - min_x) * 0.1 if max_x != min_x else 1.0
            pad_y = (max_y - min_y) * 0.1 if max_y != min_y else 1.0

            limits = [
                (min_x - pad_x, max_x + pad_x),
                (min_y - pad_y, max_y + pad_y)
            ]

    # === LYAPUNOV FUNCTION PLOT ===
    # Create grid for Lyapunov function
    x_range = np.linspace(limits[0][0], limits[0][1], resolution)
    y_range = np.linspace(limits[1][0], limits[1][1], resolution)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Prepare grid points for evaluation
    grid_points = np.zeros((X.size, num_states))
    grid_points[:, idx_x] = X.flatten()
    grid_points[:, idx_y] = Y.flatten()
    
    # Evaluate Lyapunov function
    try:
        Z_flat = lyapunov_func(grid_points)
    except Exception:
        Z_flat = np.array([lyapunov_func(s) for s in grid_points])
        
    if hasattr(Z_flat, 'ndim') and Z_flat.ndim > 1:
        Z_flat = Z_flat.flatten()
    elif isinstance(Z_flat, list):
        Z_flat = np.array(Z_flat)
        
    Z = Z_flat.reshape(X.shape)

    fig = go.Figure()

    # Plot Lyapunov Landscape
    if plot_3d:
        fig.add_trace(
            go.Surface(
                z=Z,
                x=x_range,
                y=y_range,
                colorscale='Viridis',
                name='Lyapunov Function',
                opacity=0.8,
                showscale=True
            )
        )
    else:
        fig.add_trace(
            go.Contour(
                z=Z,
                x=x_range,
                y=y_range,
                colorscale='Viridis',
                name='Lyapunov Function',
                showscale=True,
                contours=dict(
                    coloring='heatmap',
                    showlabels=True,
                )
            )
        )

    # === CERTIFIED/UNCERTIFIED REGIONS ===
    z_overlay = float(np.nanmin(Z)) if plot_3d else None
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


    # === MPC TRAJECTORIES ===
    trajectory_indices = []
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    if has_dataset:
        for idx, entry in enumerate(dataset):
            traj = entry.trajectory
            color = colors[idx % len(colors)]

            if plot_3d:
                try:
                    v_traj = lyapunov_func(traj.states)
                except Exception:
                    v_traj = np.array([lyapunov_func(s) for s in traj.states])

                if hasattr(v_traj, 'ndim') and v_traj.ndim > 1:
                    v_traj = v_traj.flatten()
                elif isinstance(v_traj, list):
                    v_traj = np.array(v_traj)

                fig.add_trace(
                    go.Scatter3d(
                        x=traj.states[:, idx_x],
                        y=traj.states[:, idx_y],
                        z=v_traj,
                        mode='lines',
                        name=f'Run {idx+1}',
                        line=dict(color=color, width=4),
                        showlegend=False
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=traj.states[:, idx_x],
                        y=traj.states[:, idx_y],
                        mode='lines',
                        name=f'Run {idx+1}',
                        line=dict(color=color, width=2),
                        opacity=0.7,
                        showlegend=False
                    )
                )
            trajectory_indices.append(len(fig.data) - 1)


    # === CONFIGURE LAYOUT ===
    if plot_3d:
        fig.update_layout(
            title_text=(
                f"Lyapunov Landscape + Regions 3D "
                f"({state_labels[0]} vs {state_labels[1]})"
            ),
            scene=dict(
                xaxis_title=state_labels[0],
                yaxis_title=state_labels[1],
                zaxis_title="V(x)",
            ),
            width=1000,
            height=800,
            autosize=True
        )
    else:
        fig.update_layout(
            title_text=(
                f"Lyapunov Landscape + Regions "
                f"({state_labels[0]} vs {state_labels[1]})"
            ),
            xaxis_title=state_labels[0],
            yaxis_title=state_labels[1],
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        )
    
    # Toggle Button for Trajectories
    if trajectory_indices:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": True}, trajectory_indices],
                            args2=[{"visible": False}, trajectory_indices],
                            label="Trajectories",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=1.0,
                    xanchor="right",
                    y=-0.05,
                    yanchor="top"
                ),
            ]
        )

    if html_path is not None:
        dir_path = os.path.dirname(html_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        fig.write_html(html_path)
        __logger__.info(f"Trajectories plot saved to {html_path}.")
    else:   
        fig.show()


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
        yaxis=dict(range=[bounds[1][0], bounds[1][1]], scaleanchor="x", scaleratio=1),
    )

    if html_path is not None:
        dir_path = os.path.dirname(html_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        fig.write_html(html_path)
        __logger__.info("Certified region plot saved to %s.", html_path)
    else:
        fig.show()