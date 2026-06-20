from __future__ import annotations

import numpy as np
import logging
import plotly.graph_objects as go

from numpy.typing import NDArray
from typing import Sequence


__logger__ = logging.getLogger(__name__)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {hex_color!r}.")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def regions_to_np(regs: NDArray | None) -> NDArray:
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
    cert_regs = regions_to_np(cert_regs)
    uncert_regs = regions_to_np(uncert_regs)
    ctex_regs = regions_to_np(ctex_regs)

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


def collapse_projected_regions(
    cert_regs: NDArray,
    uncert_regs: NDArray,
    ctex_regs: NDArray,
    indices: list[int] = [0, 1],
    decimals: int = 12,
) -> tuple[NDArray, NDArray, NDArray]:
    cert_regs_np = regions_to_np(cert_regs)
    uncert_regs_np = regions_to_np(uncert_regs)
    ctex_regs_np = regions_to_np(ctex_regs)

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


def limits_from_regions(
    cert_regs: NDArray,
    uncert_regs: NDArray,
    ctex_regs: NDArray | None = None,
    pad_ratio: float = 0.0,
    default_pad: float = 1.0,
) -> list[tuple[float, float]] | None:
    cert_regs_np = regions_to_np(cert_regs)
    uncert_regs_np = regions_to_np(uncert_regs)
    ctex_regs_np = regions_to_np(ctex_regs)

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


def regions_area_2d(regions: NDArray) -> float:
    regs = regions_to_np(regions)
    if regs.shape[0] == 0:
        return 0.0
    widths = np.maximum(0.0, regs[:, 1, 0] - regs[:, 0, 0])
    heights = np.maximum(0.0, regs[:, 1, 1] - regs[:, 0, 1])
    return float(np.sum(widths * heights))


def partition_annotation(cert_regs: NDArray, uncert_regs: NDArray, ctex_regs: NDArray) -> str:
    cert_area = regions_area_2d(cert_regs)
    uncert_area = regions_area_2d(uncert_regs)
    ctex_area = regions_area_2d(ctex_regs)
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


def create_origin_exclusion_region(
    origin_exclusion: float | Sequence[float],
    idx_x: int,
    idx_y: int,
    num_states: int,
) -> NDArray:
    """Create a single 2D bounding box representing the origin exclusion."""
    # 1. Normalisiere die Input-Werte
    if isinstance(origin_exclusion, (int, float)):
        ex_x = float(origin_exclusion)
        ex_y = float(origin_exclusion)
    elif isinstance(origin_exclusion, Sequence):
        if len(origin_exclusion) != num_states:
            raise ValueError("Sequence length must match num_states.")
        ex_x = float(origin_exclusion[idx_x])
        ex_y = float(origin_exclusion[idx_y])
    else:
        raise ValueError("origin_exclusion must be a float or a sequence of floats.")

    if ex_x <= 0.0 and ex_y <= 0.0:
        return np.empty((0, 2, 2), dtype=np.float64)

    box = np.array([
        [
            [-ex_x, -ex_y], # Lower bounds [lb_x, lb_y]
            [ ex_x,  ex_y]  # Upper bounds [ub_x, ub_y]
        ]
    ], dtype=np.float64)
    return box


def _build_X_Y(regions: NDArray) -> tuple[NDArray, NDArray]:
    lbs = regions[:, 0, :]
    ubs = regions[:, 1, :]

    X = np.column_stack([lbs[:, 0], ubs[:, 0], ubs[:, 0], lbs[:, 0], lbs[:, 0]])
    Y = np.column_stack([lbs[:, 1], lbs[:, 1], ubs[:, 1], ubs[:, 1], lbs[:, 1]])
    return X, Y


def add_regions(
    regions: NDArray | None,
    fig: go.Figure,
    color: str,
    name: str | None = None,
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
                showlegend=name is not None,
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
                showlegend=name is not None,
            )
        )