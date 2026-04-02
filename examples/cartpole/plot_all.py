import argparse
import mpc_datagen.plots as mdg_plots

from pathlib import Path
from mpc_datagen import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot trajectories from the generated dataset for the inverted pendulum on cart."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="results/inverted_pendulum_on_cart/data",
        help="Base output path for generated datasets and plots.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=150,
        help="Maximal Nunumber of trajectories to plot.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    data_path = Path(args.path)
    dataset = MPCDataset.load(data_path)

    mdg_plots.mpc_trajectories(
        dataset=dataset[:min(args.max_samples, len(dataset))],
        state_labels=["x", "v", "theta", "theta_dot"],
        control_labels=["a"],
        plot_predictions=False,
        html_path=data_path.parent / data_path.name.replace("data.hdf5", "plot_trajectories.html"),
    )



if __name__ == "__main__":
    args = parse_args()
    main(args)