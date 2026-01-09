import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data_generation.mpc_data import MPCDataset


def plot_mpc_trajectories(
    dataset: MPCDataset,
    state_labels: list,
    control_labels: list,
    plot_predictions: bool = False,
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
    """
    if len(dataset) == 0:
        print("Dataset is empty.")
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
                    showlegend=(i == 0) # Only show first occurrence in legend
                ),
                row=row, col=1
            )
            
            if plot_predictions:
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
            
            if plot_predictions:
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
                            args=[{"visible": False}, prediction_indices],
                            args2=[{"visible": True}, prediction_indices],
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
    
    output_file = "mpc_trajectories.html"
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}. Open this file in VS Code or your browser to view.")
    
    # fig.show()