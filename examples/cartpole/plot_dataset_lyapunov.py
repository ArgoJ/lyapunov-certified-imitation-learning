from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mpc_datagen import MPCDataset, mdg_plt
from lcil.utils import ArgumentParserConfig, config_field

from . import DATA_DIR, default_dataset_path, resolve_dataset_path

__logger__ = logging.getLogger("lcil.examples.cartpole.plot_dataset_lyapunov")


@dataclass(frozen=True)
class PlotDatasetLyapunovConfig(ArgumentParserConfig):
    """Configuration for plotting the interpolated Lyapunov value function from a cartpole MPC dataset."""

    dataset_path: str | None = config_field(
        default=None,
        help="Path to the MPC dataset .hdf5 file or directory. Defaults to latest dataset under results/cartpole/data.",
    )
    method: Literal["linear", "nearest", "cubic"] = config_field(
        default="linear",
        help="Interpolation method for creating the Lyapunov function from dataset ('linear', 'nearest', 'cubic').",
    )
    n_trajectories: int = config_field(
        default=50,
        help="Number of trajectories from the dataset to overlay on the plot (0 to disable).",
    )
    plot_3d: bool = config_field(
        default=False,
        help="Whether to render a 3D surface plot instead of 2D contours.",
    )
    scatter_points: bool = config_field(
        default=False,
        help="If True and 3D is active, adds 3D scatter points of all dataset (x, y, V_N) samples.",
    )
    resolution: int = config_field(
        default=100,
        help="Grid resolution for the Lyapunov landscape.",
    )
    output_path: str | None = config_field(
        default=None,
        help="Custom HTML output path. Defaults to <dataset_dir>/lyapunov_from_dataset_<method>[_3d].html.",
    )


def _build_script_defaults() -> PlotDatasetLyapunovConfig:
    default_ds = default_dataset_path()
    __logger__.info("Discovered latest dataset at %s for default configuration.", default_ds)
    return PlotDatasetLyapunovConfig(
        dataset_path=str(default_ds),
    )


def parse_args() -> PlotDatasetLyapunovConfig:
    """Parse CLI arguments into a PlotDatasetLyapunovConfig instance."""
    script_defaults = _build_script_defaults()
    parser = argparse.ArgumentParser(
        description="Interpolate and plot Lyapunov value function V_N from a cartpole MPC dataset using mdg_plt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    script_defaults.add_to_argparse(parser, suppress_defaults=True)
    args = parser.parse_args()
    return script_defaults.from_namespace(args)


def plot_dataset_lyapunov(config: PlotDatasetLyapunovConfig) -> Path:
    """Load dataset, interpolate Lyapunov value function V_N, and save HTML plot.

    Parameters
    ----------
    config : PlotDatasetLyapunovConfig
        Configuration containing dataset path, interpolation method, and plot options.

    Returns
    -------
    Path
        Path to the saved HTML plot.
    """
    dataset_path = resolve_dataset_path(config.dataset_path)
    __logger__.info("Loading MPC dataset from %s", dataset_path)
    dataset = MPCDataset.load(dataset_path)

    __logger__.info("Interpolating Lyapunov function from dataset (method='%s')...", config.method)
    lyapunov_func = mdg_plt.create_lyapunov_from_dataset(dataset, method=config.method)

    if config.output_path is not None:
        html_path = Path(config.output_path).resolve()
    else:
        suffix = "_3d.html" if config.plot_3d else ".html"
        html_path = dataset_path.parent / f"lyapunov_from_dataset_{config.method}{suffix}"

    __logger__.info("Generating Lyapunov plot with mdg_plt.lyapunov...")
    mdg_plt.lyapunov(
        lyapunov_func=lyapunov_func,
        dataset=dataset[: config.n_trajectories] if config.n_trajectories > 0 else None,
        state_labels=[r"$x$", r"$v$", r"$\theta$", r"$\dot{\theta}$"],
        resolution=int(config.resolution),
        plot_3d=bool(config.plot_3d),
        scatter_points=bool(config.scatter_points),
        html_path=html_path,
    )
    __logger__.info("Saved Lyapunov plot to %s", html_path)
    return html_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = parse_args()
    plot_dataset_lyapunov(config)


if __name__ == "__main__":
    main()
