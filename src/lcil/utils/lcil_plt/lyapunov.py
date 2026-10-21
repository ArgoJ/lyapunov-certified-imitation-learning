from __future__ import annotations

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from pathlib import Path
from numpy.typing import NDArray
from typing import Callable, TYPE_CHECKING, Sequence

from mpc_datagen import MPCDataset, mdg_plt

from .utils import (
    add_certification_overlays,
    add_regions,
    build_grid_2d,
    collapse_projected_regions,
    create_origin_exclusion_region,
    extract_certification_regions,
    infer_num_states_from_regions,
    limits_from_regions,
    resolve_state_pairs_and_labels,
)

if TYPE_CHECKING:
    from ...certification.bisect_certifier import RegionCertificationResult

__logger__ = logging.getLogger(__name__)

def lyapunov_cert_regions(
    lyapunov_func: Callable[[NDArray], NDArray],
    certification_result: "RegionCertificationResult",
    dataset: MPCDataset | None = None,
    state_indices: list[int] | None = None,
    state_labels: list[str] | None = None,
    limits: list = None,
    resolution: int = 100,
    plot_3d: bool = False,
    html_path: Path | str = None,
) -> list[mdg_plt.PairPlotResult] | None:
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
    html_path : Path | str, optional
        If provided, saves the plot to the specified HTML file.
    """
    cert_regs_np, uncert_regs_np, outside_sublevel_regs_np = (
        extract_certification_regions(certification_result)
    )

    has_dataset = dataset is not None and len(dataset) > 0
    if has_dataset:
        num_states = int(dataset[0].trajectory.states.shape[1])
    else:
        num_states = infer_num_states_from_regions(
            cert_regs_np,
            uncert_regs_np,
            outside_sublevel_regs_np,
            state_indices=state_indices,
        )

    state_indices, labels_full, pair_indices = resolve_state_pairs_and_labels(
        num_states, state_indices, state_labels
    )

    z_overlay = 0.0 if plot_3d else None

    figures: list[mdg_plt.PairPlotResult] = []
    for pair in pair_indices:
        cert_pair, uncert_pair, ctex_pair = collapse_projected_regions(
            cert_regs_np,
            uncert_regs_np,
            outside_sublevel_regs_np,
            indices=list(pair),
        )

        pair_limits = limits
        if pair_limits is None:
            pair_limits = limits_from_regions(
                cert_pair,
                uncert_pair,
                ctex_pair,
                pad_ratio=0.1,
                default_pad=1.0,
            ) or [(-1.0, 1.0), (-1.0, 1.0)]

        fig_pair = mdg_plt.lyapunov(
            lyapunov_func=lyapunov_func,
            dataset=dataset,
            state_indices=list(pair),
            state_labels=labels_full,
            limits=pair_limits,
            resolution=resolution,
            plot_3d=plot_3d,
            use_dataset_v=has_dataset,
        )

        if fig_pair is NotImplementedError:
            raise ValueError("Expected lyapunov to return figures.")

        add_certification_overlays(
            fig_pair[0].figure,
            cert_pair,
            uncert_pair,
            ctex_pair,
            z_level=z_overlay,
            fill=False,
            uncert_dash="dash",
        )
        figures.append(fig_pair[0])

    if html_path is not None:
        mdg_plt.utils._save_pair_figures(figures, html_path, kind="Lyapunov")
        return None

    return figures


def lyapunov_with_exclusion(
    lyapunov_func: Callable[[NDArray], NDArray],
    dataset: MPCDataset | None = None,
    roa_level: float | None = None,
    origin_exclusion: float | Sequence[float] | None = None,
    state_indices: list[int] | None = None,
    state_labels: list[str] | None = None,
    num_states: int | None = None,
    limits: list = None,
    resolution: int = 100,
    plot_3d: bool = False,
    html_path: Path | str | None = None,
):
    """Plot Lyapunov landscape, trajectories, and optional certified regions in 2D/3D.
    If more than two state indices are provided, one figure per 2D state pair
    is generated.

    Parameters
    ----------
    lyapunov_func : Callable[[NDArray], NDArray]
        A function that takes a state vector and returns the Lyapunov value.
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
    html_path : Path | str, optional
        If provided, saves the plot to the specified HTML file.
    """
    z_overlay = 0.0 if plot_3d else None

    results = mdg_plt.lyapunov(
        lyapunov_func=lyapunov_func,
        dataset=dataset,
        roa_level=roa_level,
        state_indices=state_indices,
        state_labels=state_labels,
        num_states=num_states,
        limits=limits,
        resolution=resolution,
        plot_3d=plot_3d,
        html_path=None,
    )

    num_states = mdg_plt.utils._resolve_num_states(num_states, dataset, state_labels, state_indices)

    for result in results:
        if origin_exclusion is not None:
            exclusion_box = create_origin_exclusion_region(
                origin_exclusion, 
                idx_x=result.idx_x, 
                idx_y=result.idx_y, 
                num_states=num_states
            )
            
            if exclusion_box.shape[0] > 0:
                add_regions(
                    exclusion_box,
                    result.figure,
                    color="#FFD700",
                    z_level=z_overlay,
                    fill=True,
                    alpha=0.4,
                    dash="dot"
                )

    if html_path is not None:
        mdg_plt.utils._save_pair_figures(results, html_path, kind="Lyapunov")
        return None

    return results


@th.no_grad()
def lyapunov_contour_matplot(
    lyap_model: th.nn.Module | Callable[[th.Tensor], th.Tensor],
    state_bounds: NDArray,
    rho_estimate: float | None = None,
    state_indices: list[int] | None = None,
    state_labels: Sequence[str] | None = None,
    resolution: int = 40,
    device: th.device | str | None = None,
) -> plt.Figure:
    """Plot 2D contour slices of an n-dimensional Lyapunov function and its ROA level set.

    For systems with more than two state dimensions, one subplot per 2D state pair
    is generated while setting all remaining state dimensions to zero.

    Parameters
    ----------
    lyap_model : th.nn.Module or Callable[[th.Tensor], th.Tensor]
        Lyapunov model mapping state tensors of shape ``(N, nx)`` to values of shape
        ``(N, 1)`` or ``(N,)``.
    state_bounds : NDArray
        State bounds of shape ``(2, nx)`` with ``[lower_bounds, upper_bounds]``.
    rho_estimate : float, optional
        Current ROA sublevel estimate ``rho``. If provided and within the plotted
        value range, the contour ``V(x) = rho`` is drawn.
    state_indices : list[int], optional
        State indices to consider. If ``None``, all state dimensions are used and
        all pairwise combinations are plotted.
    state_labels : Sequence[str], optional
        Labels for the state dimensions.
    resolution : int, optional
        Grid resolution per axis. Default is 40.
    device : th.device or str, optional
        Device on which to evaluate ``lyap_model``. Inferred from model parameters
        or buffers if ``None``.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the 2D contour plot(s).
    """
    bounds = np.asarray(state_bounds, dtype=np.float32)
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError(f"Expected state_bounds with shape (2, nx), got {bounds.shape}.")

    num_states = int(bounds.shape[1])
    state_indices, labels_full, pair_indices = resolve_state_pairs_and_labels(
        num_states, state_indices, state_labels
    )

    if device is None:
        if isinstance(lyap_model, th.nn.Module):
            param = next(lyap_model.parameters(), None)
            buf = next(lyap_model.buffers(), None)
            device = (
                param.device
                if param is not None
                else (buf.device if buf is not None else th.device("cpu"))
            )
        else:
            device = th.device("cpu")

    n_pairs = len(pair_indices)
    ncols = min(3, max(1, n_pairs))
    nrows = max(1, math.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.0 * nrows),
        squeeze=False,
        dpi=100,
    )
    axes_flat = axes.ravel()

    lb, ub = bounds[0], bounds[1]

    for pair_idx, (dim_i, dim_j) in enumerate(pair_indices):
        ax = axes_flat[pair_idx]
        Xi, Xj, grid_states = build_grid_2d(bounds, dim_i, dim_j, resolution)

        x_tensor = th.as_tensor(grid_states, dtype=th.float32, device=device)
        v_vals = lyap_model(x_tensor).squeeze(-1).detach().cpu().numpy()
        V = v_vals.reshape(resolution, resolution)

        cf = ax.contourf(Xi, Xj, V, levels=20, cmap="viridis")
        ax.contour(Xi, Xj, V, levels=20, colors="white", linewidths=0.4, alpha=0.4)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=r"$V(x)$")

        if rho_estimate is not None and np.isfinite(rho_estimate):
            v_min, v_max = float(np.min(V)), float(np.max(V))
            if v_min < float(rho_estimate) < v_max:
                ax.contour(
                    Xi,
                    Xj,
                    V,
                    levels=[float(rho_estimate)],
                    colors="crimson",
                    linewidths=2.0,
                    linestyles="--",
                )

        ax.plot(0.0, 0.0, marker="+", color="white", markersize=7, markeredgewidth=1.5)
        ax.set_xlim(float(lb[dim_i]), float(ub[dim_i]))
        ax.set_ylim(float(lb[dim_j]), float(ub[dim_j]))
        ax.set_xlabel(mdg_plt.utils._to_latex(labels_full[dim_i]))
        ax.set_ylabel(mdg_plt.utils._to_latex(labels_full[dim_j]))
        title = f"({labels_full[dim_i]}, {labels_full[dim_j]})"
        if rho_estimate is not None and np.isfinite(rho_estimate):
            title += f" | $\\rho = {float(rho_estimate):.3f}$"
        ax.set_title(title)

    for unused_idx in range(n_pairs, len(axes_flat)):
        axes_flat[unused_idx].set_visible(False)

    fig.tight_layout()
    return fig

