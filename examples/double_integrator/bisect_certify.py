from __future__ import annotations

import argparse
import logging
import torch as th
import gc

from dataclasses import dataclass, replace
from pathlib import Path

from mpc_datagen import mdg_plt
from lcil.certification import (
    BisectCertifier,
    LyapunovCertificationConfig,
)
from lcil.lyapunov_learning import LyapunovTrainingConfig, LyapunovTrainingResult
from lcil.utils import ArgumentParserConfig, config_field, lcil_plt, IntegrationMethod

from . import (
    DoubleIntegratorDynamics,
    discover_latest_lyapunov_dir,
    build_lyapunov_func,
    load_lyapunov_model,
    load_policy_model,
    find_all_lyapunov_dirs,
)
from ..constants import *

__logger__ = logging.getLogger("lcil.examples.double_integrator.bisect_certify")


@dataclass(frozen=True)
class BisectCertifyScriptConfig(ArgumentParserConfig):
    lyapunov_dir: str = config_field(help="Lyapunov run directory containing lyapunov_model.pt.")
    certify_all_lyapunov_models: bool = config_field(default=False, help="Whether to certify all saved lyapunov models in the lyapunov_dir (if False, only the best model is certified).")
    rho_multiplicator: float = config_field(default=0.4, help="Optional multiplicative factor for the initial rho estimate in the bisection search.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    test_rollout_steps: int = config_field(default=50, help="Closed-loop rollout steps per region center during empirical result testing.")
    save_dir: str | None = config_field(default=None, help="Optional directory where certification details and tester results are written.")


def _build_script_defaults() -> BisectCertifyScriptConfig | list[BisectCertifyScriptConfig]:
    default_lyapunov_dir = discover_latest_lyapunov_dir()
    __logger__.info(f"Discovered latest lyapunov directory at {default_lyapunov_dir} for default script configuration.")
    return BisectCertifyScriptConfig(
        lyapunov_dir=str(default_lyapunov_dir),
    )


def _build_certification_defaults(
    lyapunov_dir: Path,
) -> LyapunovCertificationConfig:
    training_config = LyapunovTrainingConfig.load(lyapunov_dir)
    return LyapunovCertificationConfig.from_training_config(
        training_config,
        cert_bounds=training_config.state_bounds, # TODO: apply percentage of these bounds
        bins_per_dim=2,
        center_refinement_factor=1.0,
        origin_exclusion=0.0,
        cert_method="crown",
        condition_margin=float(training_config.condition_margin),
        suppress_native_output=True,
        batch_size=32,
        abcrown_timeout=60.0,
    )


def _build_parser(
    script_defaults: BisectCertifyScriptConfig,
    certification_defaults: LyapunovCertificationConfig,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bisection-based double-integrator certification analysis with follow-up empirical rollout testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    script_defaults.add_to_argparse(parser)
    certification_defaults.add_to_argparse(
        parser,
        exclude_fields={"state_dim", "cert_bounds"},
        suppress_defaults=True,
    )
    return parser


def parse_args() -> list[tuple[BisectCertifyScriptConfig, LyapunovCertificationConfig]]:
    script_defaults = _build_script_defaults()
    certification_defaults = _build_certification_defaults(Path(script_defaults.lyapunov_dir))
    parser = _build_parser(script_defaults, certification_defaults)
    args = parser.parse_args()
    __logger__.debug("Parsed command-line arguments: %s", args)
    script_config = script_defaults.from_namespace(args)

    configs = []
    if script_config.certify_all_lyapunov_models:
        search_dir = Path(script_config.lyapunov_dir).parent
        lyap_dirs = find_all_lyapunov_dirs(search_dir)
        __logger__.info("Found %d lyapunov model directories under %s for certification", len(lyap_dirs), search_dir)
        for lyap_dir in lyap_dirs:
            script_config = replace(
                script_config,
                lyapunov_dir=str(lyap_dir)
            )
            dir_certification_defaults = _build_certification_defaults(lyap_dir)
            certification_config = dir_certification_defaults.from_namespace(args)
            configs.append((script_config, certification_config))
    else:
        configs.append((
            script_config,
            certification_defaults.from_namespace(args)
        ))

    return configs


def main() -> None:
    configs = parse_args()
    for script_config, certification_config in configs:
        device = th.device(script_config.device)
        lyapunov_dir = Path(script_config.lyapunov_dir).resolve()
        policy_model = load_policy_model(lyapunov_dir, device, model_name=POLICY_MODEL_FILENAME)
        lyap_model = load_lyapunov_model(lyapunov_dir, device, model_name=LYAPUNOV_MODEL_FILENAME)

        dyn_model = DoubleIntegratorDynamics(
            dt=policy_model.global_config.dt,
            abcrown_compatible_ops=True,
        ).to(device)
        dyn_model.eval()

        results_path = lyapunov_dir / TRAINING_RESULTS_FILENAME
        training_results = LyapunovTrainingResult.load(results_path)
        rho_estimate = training_results.rho_estimate * script_config.rho_multiplicator

        save_dir = (
            Path(script_config.save_dir).resolve()
            if script_config.save_dir is not None
            else (lyapunov_dir / CERTIFICATION_DIRNAME).resolve()
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        __logger__.info("Using rho estimate %.6f for Lyapunov dir %s", rho_estimate, lyapunov_dir)

        certifier = BisectCertifier(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            config=certification_config,
            device=device,
            # progress_level=2,
        )

        cert_results = certifier.certify(rho_estimate)
        certifier.save(save_dir)

        if cert_results is None:
            continue

        cert_bounds = certification_config.cert_bounds
        plot_bounds = [
            (float(cert_bounds[0, i]), float(cert_bounds[1, i]))
            for i in range(cert_bounds.shape[1])
        ]
        lcil_plt.certified_regions_2d(
            certification_result=cert_results,
            state_labels=["$x$", "$v$"],
            bounds=plot_bounds,
            html_path=save_dir / "certification_regions_plot.html",
        )
        mdg_plt.lyapunov(
            lyapunov_func=build_lyapunov_func(lyap_model, device),
            roa_level=cert_results.rho,
            state_labels=["$x$", "$v$"],
            num_states=2,
            limits=plot_bounds,
            plot_3d=False,
            html_path=save_dir / "certification_lyapunov_plot.html",
        )

        del policy_model
        del lyap_model
        del dyn_model
        del certifier
        del cert_results
        gc.collect()
        if device.type == "cuda":
            th.cuda.empty_cache()

if __name__ == "__main__":
    main()