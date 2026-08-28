from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from mpc_datagen import EmpiricalROAEstimator, MPCDataset

from lcil.certification import LevelSetEstimate, estimate_mpc_dataset_level_set_measure
from lcil.utils import ArgumentParserConfig, config_field
from lcil.utils.constants import LEVEL_SET_FILENAME

from .constants import RESULTS_DIR
from .metrics_collector import add_entry

__logger__ = logging.getLogger("lcil.examples.estimate_mpc_dataset_measure")


@dataclass(frozen=True)
class EstimateMPCDatasetScriptConfig(ArgumentParserConfig):
    """Configuration for evaluating and saving MPC dataset level-set measure."""

    dataset_path: str | None = config_field(
        default=None,
        help="Path to an MPC dataset .hdf5 file or directory. If omitted, discovers the latest dataset under results/.",
    )
    rho: float | None = config_field(
        default=None,
        help="Sublevel set level rho to estimate V_N(x) <= rho. If omitted, computes empirical ROA (c_empirical) via EmpiricalROAEstimator.",
    )
    output_dir: str | None = config_field(
        default=None,
        help="Directory to save the resulting JSON estimate. Defaults to the dataset directory.",
    )
    summary_name: str = config_field(
        default="mpc_level_set_estimates.csv",
        help="Filename for appending CSV summary entries in the results root.",
    )


def _discover_latest_dataset() -> Path:
    """Find the most recently modified .hdf5 dataset under the results directory."""
    candidates = list(RESULTS_DIR.glob("**/*.hdf5"))
    if not candidates:
        raise FileNotFoundError(f"No .hdf5 datasets found under {RESULTS_DIR}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_args() -> EstimateMPCDatasetScriptConfig:
    """Parse CLI arguments into a script configuration dataclass."""
    parser = argparse.ArgumentParser(
        description="Estimate and save the nD Monte-Carlo level set measure of an MPC dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = EstimateMPCDatasetScriptConfig()
    defaults.add_to_argparse(parser, suppress_defaults=True)
    args = parser.parse_args()
    return defaults.from_namespace(args)


def estimate_and_save_mpc_dataset_measure(
    config: EstimateMPCDatasetScriptConfig,
) -> tuple[LevelSetEstimate, Path]:
    """Load MPC dataset, determine rho, compute level set measure, and save results.

    Parameters
    ----------
    config : EstimateMPCDatasetScriptConfig
        Parsed CLI script configuration.

    Returns
    -------
    tuple[LevelSetEstimate, Path]
        The computed LevelSetEstimate object and the path to the saved JSON file.
    """
    if config.dataset_path is not None:
        ds_path = Path(config.dataset_path).resolve()
        if ds_path.is_dir():
            hdf5_files = list(ds_path.glob("*.hdf5"))
            if not hdf5_files:
                raise FileNotFoundError(f"No .hdf5 files found in directory {ds_path}")
            hdf5_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            ds_path = hdf5_files[0]
    else:
        ds_path = _discover_latest_dataset()
        __logger__.info("Discovered latest MPC dataset: %s", ds_path)

    if not ds_path.is_file():
        raise FileNotFoundError(f"Dataset file not found at {ds_path}")

    __logger__.info("Loading MPC dataset from %s", ds_path)
    dataset = MPCDataset.load(ds_path)

    # 1. Determine rho (manual override or empirical ROA)
    if config.rho is not None:
        rho = float(config.rho)
        __logger__.info("Using user-provided rho = %.6f", rho)
    else:
        __logger__.info("Estimating empirical ROA level via EmpiricalROAEstimator...")
        report = EmpiricalROAEstimator(dataset).estimate(show_progress=True)
        if report.c_empirical is None:
            raise ValueError("EmpiricalROAEstimator did not find a valid c_empirical for this dataset.")
        rho = float(report.c_empirical)
        __logger__.info("Inferred empirical ROA level: rho = c_empirical = %.6f", rho)

    # 2. Compute LevelSetEstimate directly from dataset
    estimate = estimate_mpc_dataset_level_set_measure(dataset=dataset, rho=rho)

    # 3. Determine output directory and save JSON
    save_dir = Path(config.output_dir).resolve() if config.output_dir is not None else ds_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = save_dir / f"mpc_{LEVEL_SET_FILENAME}"
    estimate.save(out_json_path)
    __logger__.info("Saved MPC level set estimate to %s", out_json_path)

    # 4. Append to CSV summary
    output_root = save_dir.parent if save_dir.parent != save_dir else save_dir
    summary_entry = f"{ds_path.parent.name},{rho:.6f},{estimate.measure:.6f},{estimate.details.inside_fraction:.6f}"
    add_entry(summary_entry, output_root=output_root, summary_name=config.summary_name)
    __logger__.info("Appended entry to summary CSV at %s/%s", output_root, config.summary_name)

    return estimate, out_json_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = parse_args()
    estimate_and_save_mpc_dataset_measure(config)


if __name__ == "__main__":
    main()
