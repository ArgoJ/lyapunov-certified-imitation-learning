from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch as th

from lcil.utils.region_builder import RegionBuilder


def _normalize_scalar_or_vector(values: list[float], state_dim: int, name: str) -> float | tuple[float, ...]:
    if len(values) == 1:
        return float(values[0])
    if len(values) != state_dim:
        raise ValueError(f"{name} must be a scalar or provide exactly {state_dim} values.")
    return tuple(float(value) for value in values)


def _normalize_bins(values: list[int], state_dim: int) -> int | tuple[int, ...]:
    if len(values) == 1:
        return int(values[0])
    if len(values) != state_dim:
        raise ValueError(f"bins must be a scalar or provide exactly {state_dim} values.")
    return tuple(int(value) for value in values)


def _unique_projected_regions(regions: th.Tensor, dims: tuple[int, int]) -> th.Tensor:
    projected = regions[:, :, list(dims)].detach().cpu()
    if len(projected) == 0:
        return projected

    unique_regions: list[th.Tensor] = []
    seen: set[tuple[float, float, float, float]] = set()
    for region in projected:
        key = tuple(float(value) for value in region.reshape(-1).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique_regions.append(region)

    return th.stack(unique_regions, dim=0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RegionBuilder root regions and the carved origin hole.")
    parser.add_argument(
        "--bounds-lower",
        type=float,
        nargs="+",
        default=[-1.0, -1.0],
        help="Lower certification bounds. Default: -1 -1",
    )
    parser.add_argument(
        "--bounds-upper",
        type=float,
        nargs="+",
        default=[1.0, 1.0],
        help="Upper certification bounds. Default: 1 1",
    )
    parser.add_argument(
        "--bins",
        type=int,
        nargs="+",
        default=[2],
        help="Bins per dimension. Provide one value or one per dimension.",
    )
    parser.add_argument(
        "--center-refinement",
        type=float,
        nargs="+",
        default=[1.0],
        help="Center refinement factor. Provide one value or one per dimension.",
    )
    parser.add_argument(
        "--origin-exclusion",
        type=float,
        nargs="+",
        default=[0.15],
        help="Origin exclusion radius. Provide one value or one per dimension.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs=2,
        default=[0, 1],
        metavar=("DIM_X", "DIM_Y"),
        help="State dimensions to project onto for plotting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("region_builder_plot.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--print-regions",
        action="store_true",
        help="Print the packed regions to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    lower = [float(value) for value in args.bounds_lower]
    upper = [float(value) for value in args.bounds_upper]
    if len(lower) != len(upper):
        raise ValueError("bounds-lower and bounds-upper must have the same length.")

    state_dim = len(lower)
    dims = tuple(int(dim) for dim in args.dims)
    if dims[0] == dims[1]:
        raise ValueError("dims must refer to two distinct state dimensions.")
    if any(dim < 0 or dim >= state_dim for dim in dims):
        raise ValueError(f"dims must be in [0, {state_dim - 1}].")

    bounds = th.tensor([lower, upper], dtype=th.float32)
    bins = _normalize_bins(list(args.bins), state_dim)
    center_refinement = _normalize_scalar_or_vector(list(args.center_refinement), state_dim, "center-refinement")
    origin_exclusion = _normalize_scalar_or_vector(list(args.origin_exclusion), state_dim, "origin-exclusion")

    builder = RegionBuilder(
        bounds=bounds,
        bins_per_dim=bins,
        center_refinement_factor=center_refinement,
        origin_exclusion=origin_exclusion,
    )
    regions = builder.build_regions()
    projected_regions = _unique_projected_regions(regions, dims)

    fig, ax = plt.subplots(figsize=(7, 7))
    for region in projected_regions:
        lb = region[0]
        ub = region[1]
        rect = Rectangle(
            (float(lb[0]), float(lb[1])),
            float(ub[0] - lb[0]),
            float(ub[1] - lb[1]),
            facecolor="#7db3d7",
            edgecolor="#1f5a7a",
            linewidth=1.2,
            alpha=0.35,
        )
        ax.add_patch(rect)

    exclusion_tensor = builder.origin_exclusion.detach().cpu()
    exclusion_x = float(exclusion_tensor[dims[0]])
    exclusion_y = float(exclusion_tensor[dims[1]])
    if exclusion_x > 0.0 and exclusion_y > 0.0:
        hole = Rectangle(
            (-exclusion_x, -exclusion_y),
            2.0 * exclusion_x,
            2.0 * exclusion_y,
            facecolor="none",
            edgecolor="#b22222",
            linewidth=2.0,
            linestyle="--",
        )
        ax.add_patch(hole)

    ax.set_xlim(float(bounds[0, dims[0]]), float(bounds[1, dims[0]]))
    ax.set_ylim(float(bounds[0, dims[1]]), float(bounds[1, dims[1]]))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(f"x[{dims[0]}]")
    ax.set_ylabel(f"x[{dims[1]}]")
    ax.set_title(
        f"RegionBuilder projection ({len(regions)} regions, {len(projected_regions)} unique in 2D)"
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    total_volume = float(((regions[:, 1] - regions[:, 0]).prod(dim=1)).sum().item()) if len(regions) > 0 else 0.0
    print(f"Wrote plot to {args.output.resolve()}")
    print(f"Built {len(regions)} regions; plotted {len(projected_regions)} unique projected rectangles.")
    print(f"Total covered volume outside exclusion: {total_volume:.6f}")
    if args.print_regions:
        print(regions)


if __name__ == "__main__":
    main()