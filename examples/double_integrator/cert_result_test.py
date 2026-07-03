from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch as th
from mpc_datagen import MPCDataset, mdg_plt

from lcil.certification import (
    CertificationResultTester,
    LyapunovCertificationConfig,
    estimate_level_set_measure,
)
from lcil.certification import RegionCertificationResult
from lcil.utils import ArgumentParserConfig, config_field, lcil_plt, add_entry
from lcil.rollouts import build_rollout_dataset

from . import (
    DoubleIntegratorDynamics,
    load_lyapunov_model,
    load_mpc_config,
    load_policy_model,
    build_lyapunov_func,
    find_all_lyapunov_dirs,
    discover_latest_cert_lyapunov_path,
    require_file,
    require_dir,
)
from ..constants import *

__logger__ = logging.getLogger("lcil.examples.double_integrator.cert_result_test")


@dataclass(frozen=True)
class RetestCertResultScriptConfig(ArgumentParserConfig):
    lyapunov_dir: str | None = config_field(
        help="Defaults to the latest saved result under results/double_integrator.",
    )
    apply_all_lyapunov_dirs: bool = config_field(
        default=False,
        help="Whether to apply the tester to all lyapunov models found under the parent directory of the specified lyapunov_dir (if False, only the specified lyapunov_dir is tested)."
    )
    device: str = config_field(
        default="cpu", 
        help="Torch device string (for example cpu or cuda).")
    rollout_steps: int = config_field(
        default=50,
        help="Closed-loop rollout steps per region center during empirical result testing.",
    )
    test_sample_size: int = config_field(
        default=256,
        help="Number of states sampled near the rho-sublevel boundary for empirical testing.",
    )


def _build_script_defaults() -> RetestCertResultScriptConfig:
    default_lyapunov_dir = discover_latest_cert_lyapunov_path()
    __logger__.info(f"Discovered latest lyapunov directory at {default_lyapunov_dir} for default script configuration.")
    return RetestCertResultScriptConfig(
        lyapunov_dir=str(default_lyapunov_dir),
    )


def parse_args() -> list[RetestCertResultScriptConfig]:
    script_defaults = _build_script_defaults()
    parser = argparse.ArgumentParser(
        description="Load a saved certification result and rerun empirical center-rollout testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    script_defaults.add_to_argparse(parser, suppress_defaults=True)
    args = parser.parse_args()
    __logger__.info("Parsed command-line arguments: %s", args)
    script_config = script_defaults.from_namespace(args)

    configs = []
    if script_config.apply_all_lyapunov_dirs:
        search_dir = Path(script_config.lyapunov_dir).parent.parent
        lyap_dirs = find_all_lyapunov_dirs(search_dir)
        __logger__.info("Found %d lyapunov model directories under %s for certification", len(lyap_dirs), search_dir)
        for lyap_dir in lyap_dirs:
            script_config = replace(
                script_config,
                lyapunov_dir=str(lyap_dir)
            )
            configs.append(script_config)
    else:
        configs.append(script_config)

    return configs


def _save_lyapunov_plot(
    *,
    cert_dir: Path,
    certification_config: LyapunovCertificationConfig,
    lyapunov_func: Callable[[np.ndarray], np.ndarray],
    roa_level: float | None = None,
    rollout_dataset: MPCDataset | None = None,
) -> None:
    cert_bounds = np.asarray(certification_config.cert_bounds, dtype=float)
    plot_path = cert_dir / "certification_tester_lyapunov_plot.html"
    lcil_plt.lyapunov_with_exclusion(
        lyapunov_func=lyapunov_func,
        dataset=rollout_dataset,
        roa_level=roa_level,
        origin_exclusion=certification_config.origin_exclusion,
        state_indices=[0, 1],
        state_labels=["$x$", "$v$"],
        limits=cert_bounds.T.tolist(),
        plot_3d=False,
        html_path=plot_path,
    )


def _save_level_set_metrics(
    *,
    cert_dir: Path,
    cert_result: RegionCertificationResult,
    lyapunov_fn: Callable[[th.Tensor], th.Tensor],
    state_dim: int,
    device: th.device,
) -> None:
    metrics_path = cert_dir / LEVEL_SET_FILENAME
    level_set_estimate = estimate_level_set_measure(
        lyapunov_fn=lyapunov_fn,
        rho=float(cert_result.rho),
        num_states=int(state_dim),
        device=device,
    )
    level_set_estimate.save(metrics_path)

    add_entry(
        f"{cert_dir.parent.name}, {level_set_estimate.measure:.6f}",
        output_root=cert_dir.parents[1],
        summary_name="level_set_estimates.csv",
    )


def main() -> None:
    script_configs = parse_args()
    for script_config in script_configs:
        device = th.device(script_config.device)

        lyapunov_dir = Path(script_config.lyapunov_dir).resolve()

        lyapunov_path = require_file(lyapunov_dir / LYAPUNOV_MODEL_FILENAME, name="Lyapunov checkpoint")
        policy_path = require_file(lyapunov_dir / POLICY_MODEL_FILENAME, name="Policy checkpoint")

        try:
            cert_path = require_dir(lyapunov_dir / CERTIFICATION_DIRNAME, name="Certification directory")
            certification_config_path = require_file(cert_path / CERTIFICATION_CONFIG_FILENAME, name="Certification config")
            cert_result_path = require_file(cert_path / CERTIFICATION_DETAILS_FILENAME, name="Certification result details")
        except FileNotFoundError as e:
            __logger__.error("Skipping due to missing file:\n%s", e)
            continue

        certification_config = LyapunovCertificationConfig.load(certification_config_path)
        cert_result = RegionCertificationResult.load(cert_result_path)
        
        if not cert_result.global_success:
            __logger__.warning(
                "Certification result at rho=%.6f is not globally proven "
                "(partial_success=%s, certified_sublevel=%d, uncertified=%d, outside_sublevel=%d). "
                "The empirical tester will still sample the whole V(x) <= rho sublevel set, "
                "so violations may come from uncertified regions.",
                float(cert_result.rho),
                cert_result.partial_success,
                len(cert_result.certified_sublevel_regions),
                len(cert_result.uncertified_regions),
                len(cert_result.outside_sublevel_regions),
            )

        policy_model = load_policy_model(policy_path, device)
        mpc_cfg = load_mpc_config(policy_path)
        lyap_model = load_lyapunov_model(lyapunov_path, device)
        lyapunov_func = build_lyapunov_func(lyap_model, device)

        dyn_model = DoubleIntegratorDynamics(
            dt=mpc_cfg.dt,
            abcrown_compatible_ops=True,
        ).to(device)
        dyn_model.eval()

        cert_tester = CertificationResultTester(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            config=certification_config,
            device=device,
        )
        test_results = cert_tester.test_result(
            rho=float(cert_result.rho),
            sample_size=int(script_config.test_sample_size),
            rollout_steps=int(script_config.rollout_steps),
        )
        test_results.save(cert_path)

        if test_results.rho_boundary.max_violation > 0:
            __logger__.warning("Skipping rollout dataset generation and metric/plot saving for this result.")
            continue

        rollout_dataset = build_rollout_dataset(
            initial_states=test_results.rho_boundary.sampled_states,
            policy_model=policy_model,
            mpc_config=mpc_cfg,
            dyn_model=dyn_model,
            lyap_model=lyap_model,
            rollout_steps=int(script_config.rollout_steps),
            device=device,
        )
        _save_level_set_metrics(
            cert_dir=cert_path,
            cert_result=cert_result,
            lyapunov_fn=lyap_model,
            state_dim=int(mpc_cfg.nx),
            device=device,
        )
        _save_lyapunov_plot(
            cert_dir=cert_path,
            certification_config=certification_config,
            lyapunov_func=lyapunov_func,
            roa_level=float(cert_result.rho),
            rollout_dataset=rollout_dataset,
        )


if __name__ == "__main__":
    main()
