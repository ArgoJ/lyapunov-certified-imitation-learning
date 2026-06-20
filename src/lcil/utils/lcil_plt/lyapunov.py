from __future__ import annotations

import logging

from pathlib import Path
from numpy.typing import NDArray
from typing import Callable, TYPE_CHECKING, Sequence

from mpc_datagen.mpc_data import MPCDataset
from mpc_datagen.plots import (
    PairPlotResult,
    lyapunov,
    _save_pair_figures,
    _resolve_num_states,
    _resolve_indices,
    _resolve_labels,
    _state_index_pairs,
)

from .utils import (
    add_regions,
    collapse_projected_regions,
    limits_from_regions,
    partition_annotation,
    regions_to_np,
    create_origin_exclusion_region,
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
) -> list[PairPlotResult] | None:
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
    cert_regs_np = regions_to_np(certification_result.certified_sublevel_regions)
    uncert_regs_np = regions_to_np(certification_result.uncertified_regions)
    outside_sublevel_regs_np = regions_to_np(certification_result.outside_sublevel_regions)

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
        cert_pair, uncert_pair, ctex_pair = collapse_projected_regions(
            cert_regs_np,
            uncert_regs_np,
            outside_sublevel_regs_np,
            indices=list(pair),
        )
        regions_by_pair[pair] = (cert_pair, uncert_pair, ctex_pair)

        if limits is not None:
            pair_limits[pair] = limits
            continue

        inferred_limits = limits_from_regions(
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

    figures: list[PairPlotResult] = {}
    for pair in pair_indices:
        fig_pair = lyapunov(
            lyapunov_func=lyapunov_func,
            dataset=dataset,
            state_indices=list(pair),
            state_labels=labels_full,
            limits=pair_limits[pair],
            resolution=resolution,
            plot_3d=plot_3d,
            use_dataset_v=has_dataset,
        )

        if fig_pair is NotImplementedError:
            raise ValueError("Expected lyapunov to return figures.")
        
        fig = fig_pair[0].figure
        cert_pair, uncert_pair, ctex_pair = regions_by_pair[pair]
        add_regions(
            cert_pair,
            fig,
            color="#209209",
            z_level=z_overlay,
            name="Certified",
            fill=False,
        )
        add_regions(
            uncert_pair,
            fig,
            color="#c53131",
            z_level=z_overlay,
            name="Uncertified",
            dash="dash",
            fill=False,
        )
        add_regions(
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
                text=partition_annotation(cert_pair, uncert_pair, ctex_pair),
            )
        figures.append(fig)

    if html_path is not None:
        _save_pair_figures(figures, html_path, kind="Lyapunov")
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

    results = lyapunov(
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

    num_states = _resolve_num_states(num_states, dataset, state_labels, state_indices)

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
        _save_pair_figures(results, html_path, kind="Lyapunov")
        return None

    return results
