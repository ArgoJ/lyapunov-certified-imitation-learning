import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Callable

from mpc_datagen.mpc_data import MPCDataset
from .package_logger import get_package_logger

__logger__ = get_package_logger(__name__)


def lyapunov(
    dataset: MPCDataset,
    lyapunov_func: Callable[[np.ndarray], np.ndarray],
    state_indices: list = [0, 1],
    state_labels: list[str] | None = None,
    limits: list = None,
    resolution: int = 100,
    plot_3d: bool = False,
    certified_regions: list[tuple[list[float], list[float]]] | None = None,
    uncertified_regions: list[tuple[list[float], list[float]]] | None = None,
    html_path: str = None,
):
    """Plot Lyapunov landscape, trajectories, and optional certified regions in 2D/3D.
    Only two state dimensions can be visualized at once.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories to plot.
    lyapunov_func : Callable[[np.ndarray], np.ndarray]
        A function that takes a state vector and returns the Lyapunov value.
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
    if len(dataset) == 0:
        __logger__.warning("Dataset is empty.")
        return

    # Infer dimensions
    first_traj = dataset[0].trajectory
    num_states = first_traj.states.shape[1]
    
    if len(state_indices) != 2:
        raise ValueError("state_indices must contain exactly 2 indices.")

    idx_x, idx_y = state_indices
    if state_labels is None:
        state_labels = [f"State {idx_x}", f"State {idx_y}"]
    if len(state_labels) != 2:
        raise ValueError("state_labels must contain exactly 2 labels.")

    certified_regions = certified_regions or []
    uncertified_regions = uncertified_regions or []

    # Determine limits if not provided
    if limits is None:
        all_states = np.vstack([d.trajectory.states for d in dataset])
        min_x, max_x = all_states[:, idx_x].min(), all_states[:, idx_x].max()
        min_y, max_y = all_states[:, idx_y].min(), all_states[:, idx_y].max()

        all_regions = certified_regions + uncertified_regions
        if all_regions:
            min_x = min(min_x, min(lb[0] for lb, _ in all_regions))
            max_x = max(max_x, max(ub[0] for _, ub in all_regions))
            min_y = min(min_y, min(lb[1] for lb, _ in all_regions))
            max_y = max(max_y, max(ub[1] for _, ub in all_regions))
        
        # Add some padding
        pad_x = (max_x - min_x) * 0.2 if max_x != min_x else 1.0
        pad_y = (max_y - min_y) * 0.2 if max_y != min_y else 1.0
        
        limits = [
            (min_x - pad_x, max_x + pad_x),
            (min_y - pad_y, max_y + pad_y)
        ]

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

    def _add_region_outlines_2d(
        regions: list[tuple[list[float], list[float]]],
        color: str,
        name: str,
        dash: str = "solid",
    ) -> None:
        if not regions:
            return
        for lb, ub in regions:
            fig.add_shape(
                type="rect",
                x0=lb[0],
                y0=lb[1],
                x1=ub[0],
                y1=ub[1],
                line=dict(color=color, width=2, dash=dash),
                fillcolor="rgba(0,0,0,0)",
            )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color, width=2, dash=dash),
                name=name,
                showlegend=True,
            )
        )

    def _add_region_outlines_3d(
        regions: list[tuple[list[float], list[float]]],
        color: str,
        name: str,
        z_level: float,
        dash: str = "solid",
    ) -> None:
        if not regions:
            return
        for lb, ub in regions:
            x_loop = [lb[0], ub[0], ub[0], lb[0], lb[0]]
            y_loop = [lb[1], lb[1], ub[1], ub[1], lb[1]]
            z_loop = [z_level] * 5
            fig.add_trace(
                go.Scatter3d(
                    x=x_loop,
                    y=y_loop,
                    z=z_loop,
                    mode="lines",
                    line=dict(color=color, width=4, dash=dash),
                    name=name,
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="lines",
                line=dict(color=color, width=4, dash=dash),
                name=name,
                showlegend=True,
            )
        )

    if plot_3d:
        z_overlay = float(np.nanmin(Z))
        _add_region_outlines_3d(
            certified_regions,
            color="#d62728",
            name="Certified (outline)",
            z_level=z_overlay,
        )
        _add_region_outlines_3d(
            uncertified_regions,
            color="#7f7f7f",
            name="Uncertified (outline)",
            z_level=z_overlay,
            dash="dash",
        )
    else:
        _add_region_outlines_2d(
            certified_regions,
            color="#d62728",
            name="Certified (outline)",
        )
        _add_region_outlines_2d(
            uncertified_regions,
            color="#7f7f7f",
            name="Uncertified (outline)",
            dash="dash",
        )

    trajectory_indices = []
    
    # Plot MPC Trajectories
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

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

    # Layout Configuration
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
    certified_regions: list[tuple[list[float], list[float]]],
    uncertified_regions: list[tuple[list[float], list[float]]],
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
    state_labels : list of str, optional
        Axis labels for the two states. Defaults to ["State 0", "State 1"].
    bounds : list of tuples, optional
        ((min_x, max_x), (min_y, max_y)). If None, inferred from regions.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.
    """
    if not certified_regions and not uncertified_regions:
        __logger__.warning("No regions provided for plotting.")
        return

    if state_labels is None:
        state_labels = ["State 0", "State 1"]

    all_regions = certified_regions + uncertified_regions
    if bounds is None:
        x_min = min(lb[0] for lb, _ in all_regions)
        x_max = max(ub[0] for _, ub in all_regions)
        y_min = min(lb[1] for lb, _ in all_regions)
        y_max = max(ub[1] for _, ub in all_regions)
        bounds = [(x_min, x_max), (y_min, y_max)]

    fig = go.Figure()

    def add_regions(regions: list[tuple[list[float], list[float]]], color: str, name: str):
        for lb, ub in regions:
            fig.add_shape(
                type="rect",
                x0=lb[0],
                y0=lb[1],
                x1=ub[0],
                y1=ub[1],
                line=dict(color=color, width=1),
                fillcolor=color,
                opacity=0.3,
            )
        # Legend entry
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=name,
                showlegend=True,
            )
        )

    if certified_regions:
        add_regions(certified_regions, "#2ca02c", "Certified")
    if uncertified_regions:
        add_regions(uncertified_regions, "#d62728", "Uncertified")

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