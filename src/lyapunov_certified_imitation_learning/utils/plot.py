import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Callable

from mpc_datagen.mpc_data import MPCDataset
from .package_logger import PackageLogger

__logger__ = PackageLogger.get_logger(__name__)


def mpc_trajectories(
    dataset: MPCDataset,
    state_labels: list,
    control_labels: list,
    plot_predictions: bool = False,
    html_path: str = None,
):
    """Plot MPC trajectories for states and controls using Plotly.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories to plot.
    state_labels : list
        List of labels for each state variable.
    control_labels : list
        List of labels for each control variable.
    plot_predictions : bool, optional
        If True, plot the OCP predictions at each step. Default is False.
    html_path : str, optional
        If provided, saves the plot to the specified HTML file.
    """
    if len(dataset) == 0:
        __logger__.warning("Dataset is empty.")
        return

    # Extract dimensions from the first trajectory
    first_traj = dataset[0].trajectory
    num_states = first_traj.states.shape[1]
    num_controls = first_traj.inputs.shape[1]

    # Create subplots
    fig = make_subplots(
        rows=num_states + num_controls, 
        cols=1, 
        shared_xaxes=True,
        subplot_titles=(state_labels + control_labels),
        vertical_spacing=0.05
    )

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    prediction_indices = []

    # Plot states
    for i in range(num_states):
        row = i + 1
        for idx in range(len(dataset)):
            traj = dataset[idx].trajectory
            color = colors[idx % len(colors)]
            
            # Main Trajectory
            fig.add_trace(
                go.Scatter(
                    x=traj.time, 
                    y=traj.states[:, i],
                    mode='lines',
                    name=f'Run {idx+1} - {state_labels[i]}',
                    line=dict(color=color),
                    legendgroup=f'Run {idx+1}',
                    showlegend=(i == 0)
                ),
                row=row, col=1
            )
            
            if plot_predictions and traj.solved_states is not None and not np.all(np.isnan(traj.solved_states)):
                dt = traj.time[1] - traj.time[0] if len(traj.time) > 1 else 0.1
                
                # Consolidate prediction lines into one trace with None gaps for performance
                x_lines = []
                y_lines = []
                
                for k in range(traj.solved_states.shape[0]):
                    pred_state = traj.solved_states[k, :, i]
                    if np.isnan(pred_state).all():
                        continue
                    
                    t_start = traj.time[k]
                    t_pred = t_start + np.arange(len(pred_state)) * dt
                    
                    x_lines.extend(t_pred)
                    x_lines.append(None)
                    y_lines.extend(pred_state)
                    y_lines.append(None)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_lines,
                        y=y_lines,
                        mode='lines',
                        line=dict(color=color, width=1),
                        opacity=0.3,
                        showlegend=False,
                        legendgroup=f'Run {idx+1}',
                        hoverinfo='skip'
                    ),
                    row=row, col=1
                )
                prediction_indices.append(len(fig.data) - 1)

    # Plot controls
    for i in range(num_controls):
        plot_idx = num_states + i
        row = plot_idx + 1
        for idx in range(len(dataset)):
            traj = dataset[idx].trajectory
            color = colors[idx % len(colors)]
            
            # Controls (Step plot)
            fig.add_trace(
                go.Scatter(
                    x=traj.time[:-1],
                    y=traj.inputs[:, i],
                    mode='lines',
                    line=dict(color=color, shape='hv'), # 'hv' for step-after behavior
                    name=f'Run {idx+1} - {control_labels[i]}',
                    legendgroup=f'Run {idx+1}',
                    showlegend=False
                ),
                row=row, col=1
            )
            
            if plot_predictions and traj.solved_inputs is not None and not np.all(np.isnan(traj.solved_inputs)):
                dt = traj.time[1] - traj.time[0] if len(traj.time) > 1 else 0.1
                
                x_lines = []
                y_lines = []
                
                for k in range(traj.solved_inputs.shape[0]):
                    pred_input = traj.solved_inputs[k, :, i]
                    if np.isnan(pred_input).all():
                        continue
                    
                    t_start = traj.time[k]
                    t_pred = t_start + np.arange(len(pred_input)) * dt
                    
                    x_lines.extend(t_pred)
                    x_lines.append(None)
                    y_lines.extend(pred_input)
                    y_lines.append(None)

                fig.add_trace(
                    go.Scatter(
                        x=x_lines,
                        y=y_lines,
                        mode='lines',
                        line=dict(color=color, width=1, shape='hv'),
                        opacity=0.3,
                        showlegend=False,
                        legendgroup=f'Run {idx+1}',
                        hoverinfo='skip'
                    ),
                    row=row, col=1
                )
                prediction_indices.append(len(fig.data) - 1)

    fig.update_layout(
        height=300 * (num_states + num_controls), 
        title_text="MPC Trajectories",
        hovermode="x unified"
    )
    
    if plot_predictions and prediction_indices:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": True}, prediction_indices],
                            args2=[{"visible": False}, prediction_indices],
                            label="Predictions",
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


def lyapunov(
    dataset: MPCDataset,
    lyapunov_func: Callable[[np.ndarray], np.ndarray],
    state_indices: list = [0, 1],
    limits: list = None,
    resolution: int = 100,
    plot_3d: bool = False,
    html_path: str = None,
):
    """Plot the Lyapunov function landscape and MPC trajectories in 2D or 3D.
    Only two state dimensions can be visualized at once.

    Parameters
    ----------
    dataset : MPCDataset
        The dataset containing trajectories to plot.
    lyapunov_func : Callable[[np.ndarray], np.ndarray]
        A function that takes a state vector and returns the Lyapunov value.
    state_indices : list, optional
        Indices of the two state variables to plot (x, y axes). Default is [0, 1].
    limits : list of tuples, optional
        ((min_x, max_x), (min_y, max_y)). If None, inferred from data with padding.
    resolution : int, optional
        Grid resolution for the Lyapunov function contour plot.
    plot_3d : bool, optional
        If True, plot a 3D surface and 3D trajectories. Default is False.
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

    # Determine limits if not provided
    if limits is None:
        all_states = np.vstack([d.trajectory.states for d in dataset])
        min_x, max_x = all_states[:, idx_x].min(), all_states[:, idx_x].max()
        min_y, max_y = all_states[:, idx_y].min(), all_states[:, idx_y].max()
        
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

    trajectory_indices = []
    
    # Plot MPC Trajectories
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    for idx in range(len(dataset)):
        traj = dataset[idx].trajectory
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
            title_text=f"Lyapunov Landscape 3D (States {idx_x} vs {idx_y})",
            scene=dict(
                xaxis_title=f"State {idx_x}",
                yaxis_title=f"State {idx_y}",
                zaxis_title="V(x)",
            ),
            width=1000,
            height=800,
            autosize=True
        )
    else:
        fig.update_layout(
            title_text=f"Lyapunov Landscape (States {idx_x} vs {idx_y})",
            xaxis_title=f"State {idx_x}",
            yaxis_title=f"State {idx_y}",
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