from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from mpc_datagen import MPCDataset, mdg_plt
from lcil.utils import ArgumentParserConfig, config_field

from . import default_dataset_path, resolve_dataset_path

__logger__ = logging.getLogger("lcil.examples.double_integrator.plot_dataset_lyapunov")


@dataclass(frozen=True)
class PlotDatasetLyapunovConfig(ArgumentParserConfig):
    """Configuration for plotting the Lyapunov value function directly from an MPC dataset."""

    dataset_path: str | None = config_field(
        default=None,
        help="Path to the MPC dataset .hdf5 file or directory. Defaults to latest dataset under results/double_integrator/data.",
    )
    n_trajectories: int = config_field(
        default=50,
        help="Number of trajectories from the dataset to overlay on the plot (0 to disable).",
    )
    plot_3d: bool = config_field(
        default=False,
        help="Whether to render a 3D surface plot instead of 2D contours.",
    )
    use_dataset_v: bool = config_field(
        default=True,
        help="If True, uses the dataset's solved MPC cost V_N for trajectory values directly.",
    )
    scatter_points: bool = config_field(
        default=True,
        help="If True and 3D is active, adds 3D scatter points of all dataset (x, y, V_N) samples.",
    )
    output_path: str | None = config_field(
        default=None,
        help="Custom HTML output path. Defaults to <dataset_dir>/dataset_lyapunov[_3d].html.",
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
        description="Plot Lyapunov optimal value function V_N directly from an MPC dataset using mdg_plt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    script_defaults.add_to_argparse(parser, suppress_defaults=True)
    args = parser.parse_args()
    return script_defaults.from_namespace(args)


def plot_dataset_lyapunov(config: PlotDatasetLyapunovConfig) -> Path:
    """Load dataset and plot Lyapunov values V_N directly using mdg_plt.

    Parameters
    ----------
    config : PlotDatasetLyapunovConfig
        Configuration containing dataset path and plot options.

    Returns
    -------
    Path
        Path to the saved HTML plot.
    """
    dataset_path = resolve_dataset_path(config.dataset_path)
    __logger__.info("Loading MPC dataset from %s", dataset_path)
    dataset = MPCDataset.load(dataset_path)

    if config.output_path is not None:
        html_path = Path(config.output_path).resolve()
    else:
        suffix = "_3d.html" if config.plot_3d else ".html"
        html_path = dataset_path.parent / f"dataset_lyapunov{suffix}"

    __logger__.info("Generating Lyapunov plot directly from dataset with mdg_plt.lyapunov...")
    mdg_plt.lyapunov(
        lyapunov_func=None,
        dataset=dataset[: config.n_trajectories] if config.n_trajectories > 0 else dataset,
        state_labels=[r"$x$", r"$v$"],
        plot_3d=bool(config.plot_3d),
        use_dataset_v=bool(config.use_dataset_v),
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
