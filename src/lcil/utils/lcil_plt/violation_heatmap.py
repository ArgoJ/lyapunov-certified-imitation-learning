"""Empirical violation heatmap for Lyapunov certification diagnostics.

Evaluates the Lyapunov decrease condition on a dense grid and produces
2D heatmaps showing where violations occur, projected onto all pairs
of state dimensions.
"""
from __future__ import annotations

import logging
import numpy as np
import torch as th
import torch.nn as nn
import plotly.graph_objects as go

from pathlib import Path
from typing import Sequence
from numpy.typing import NDArray

from mpc_datagen.plots import (
    _handle_figure_output,
    _to_latex,
)

__logger__ = logging.getLogger(__name__)


def _build_grid_2d(
    bounds: NDArray,
    dim_i: int,
    dim_j: int,
    resolution: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Build a 2D grid for dimensions (dim_i, dim_j), fixing all others at zero.

    Returns
    -------
    xi, xj : NDArray
        Meshgrid arrays for the two selected dimensions.
    grid_states : NDArray
        Full state-space grid of shape (resolution^2, state_dim).
    """
    state_dim = bounds.shape[1]
    lb, ub = bounds[0], bounds[1]

    xi = np.linspace(lb[dim_i], ub[dim_i], resolution)
    xj = np.linspace(lb[dim_j], ub[dim_j], resolution)
    Xi, Xj = np.meshgrid(xi, xj, indexing="ij")

    grid_states = np.zeros((resolution * resolution, state_dim), dtype=np.float32)
    grid_states[:, dim_i] = Xi.ravel()
    grid_states[:, dim_j] = Xj.ravel()
    return Xi, Xj, grid_states


@th.no_grad()
def compute_violation_grid(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    grid_states: NDArray,
    kappa: float,
    device: th.device,
    batch_size: int = 4096,
) -> NDArray:
    """Evaluate the Lyapunov decrease violation on a grid of states.

    Returns ``V(x_next) - (1-kappa) * V(x)`` for each state.
    Positive values indicate violations.
    """
    n = grid_states.shape[0]
    violations = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = th.as_tensor(grid_states[start:end], dtype=th.float32, device=device)
        v_curr = lyap_model(x)
        u = policy_model(x)
        x_next = dyn_model(x, u)
        v_next = lyap_model(x_next)
        decrease_violation = v_next - (1.0 - kappa) * v_curr
        violations[start:end] = decrease_violation.squeeze(-1).cpu().numpy()

    return violations


def empirical_violation_heatmap(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    state_bounds: NDArray,
    kappa: float,
    device: th.device | str = "cpu",
    state_labels: Sequence[str] | None = None,
    resolution: int = 80,
    html_path: Path | str | None = None,
) -> list[go.Figure] | None:
    """Generate 2D heatmaps of the Lyapunov decrease violation for all state-dimension pairs.

    Parameters
    ----------
    policy_model : nn.Module
        The control policy network.
    lyap_model : nn.Module
        The Lyapunov candidate network.
    dyn_model : nn.Module
        The dynamics model.
    state_bounds : NDArray
        Bounds of shape ``(2, state_dim)``.
    kappa : float
        Decrease rate in the Lyapunov condition.
    device : th.device or str
        Torch device.
    state_labels : list of str, optional
        Labels for each state dimension.
    resolution : int
        Grid resolution per dimension.
    html_path : Path or str, optional
        If provided, save the combined figure to this HTML file.

    Returns
    -------
    list of go.Figure or None
        One figure per state-dimension pair, or None if saved to file.
    """
    device = th.device(device)
    bounds = np.asarray(state_bounds, dtype=np.float32)
    state_dim = bounds.shape[1]

    if state_labels is None:
        state_labels = [f"x{i}" for i in range(state_dim)]

    policy_model.eval()
    lyap_model.eval()
    dyn_model.eval()

    pairs = [(i, j) for i in range(state_dim) for j in range(i + 1, state_dim)]
    figures = []

    for dim_i, dim_j in pairs:
        Xi, Xj, grid_states = _build_grid_2d(bounds, dim_i, dim_j, resolution)

        violations = compute_violation_grid(
            policy_model, lyap_model, dyn_model,
            grid_states, kappa, device,
        )
        V = violations.reshape(resolution, resolution)

        # Clip for better visualization: show range around zero
        vmax = max(float(np.abs(V).max()), 1e-6)

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=Xi[0, :] if Xi.shape[0] > 0 else [],
            y=Xj[:, 0] if Xj.shape[0] > 0 else [],
            z=V.T,
            colorscale=[
                [0.0, "rgb(8, 48, 107)"],       # deep blue (strong satisfaction)
                [0.45, "rgb(107, 174, 214)"],    # light blue
                [0.5, "rgb(255, 255, 255)"],     # white (boundary)
                [0.55, "rgb(252, 174, 145)"],    # light red
                [1.0, "rgb(165, 15, 21)"],       # deep red (strong violation)
            ],
            zmid=0.0,
            zmin=-vmax,
            zmax=vmax,
            colorbar=dict(
                title="V(x') - (1-κ)V(x)",
                titleside="right",
            ),
        ))

        # Add zero contour
        fig.add_trace(go.Contour(
            x=Xi[0, :] if Xi.shape[0] > 0 else [],
            y=Xj[:, 0] if Xj.shape[0] > 0 else [],
            z=V.T,
            contours=dict(
                start=0, end=0, size=1,
                coloring="none",
            ),
            line=dict(color="black", width=2),
            showscale=False,
            name="V(x')=(1-κ)V(x)",
        ))

        label_i = _to_latex(state_labels[dim_i])
        label_j = _to_latex(state_labels[dim_j])
        fig.update_layout(
            title=f"Decrease Violation: {label_i} vs {label_j} (other dims=0)",
            xaxis_title=label_i,
            yaxis_title=label_j,
            width=600,
            height=500,
            template="plotly_white",
        )
        figures.append(fig)

    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)

        if len(figures) == 1:
            figures[0].write_html(str(html_path))
        else:
            # Combine all figures into a single HTML file
            combined_html = "<html><head><title>Violation Heatmaps</title></head><body>\n"
            for i, fig in enumerate(figures):
                combined_html += fig.to_html(full_html=False, include_plotlyjs=(i == 0))
                combined_html += "<hr>\n"
            combined_html += "</body></html>"
            html_path.write_text(combined_html, encoding="utf-8")

        __logger__.info(
            "%d violation heatmap(s) saved to %s",
            len(figures), html_path,
        )
        return None

    return figures
