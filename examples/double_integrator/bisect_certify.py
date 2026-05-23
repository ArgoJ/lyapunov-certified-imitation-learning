from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from numpy.typing import NDArray
from typing import Callable

import torch as th

from mpc_datagen import mdg_plt
from lcil.certification import (
    BisectCertifier,
    LyapunovCertificationConfig,
)
from lcil.lyapunov_learning import LyapunovTrainingConfig, LyapunovTrainingResult
from lcil.utils import ArgumentParserConfig, config_field, lcil_plt

from . import (
    DoubleIntegratorDynamics,
    discover_latest_lyapunov_dir,
    discover_latest_policy_dir,
    load_lyapunov_model,
    load_policy_model,
)

__logger__ = logging.getLogger("lcil.examples.double_integrator.certify")

_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "double_integrator"
_DEFAULT_CERT_BOUND_SCALES = (0.15, 0.15)


@dataclass(frozen=True)
class BisectCertifyScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(help="Policy run directory containing model.pt.")
    lyapunov_dir: str = config_field(help="Lyapunov run directory containing lyapunov_model.pt.")
    certify_all_lyapunov_models: bool = config_field(default=False, help="Whether to certify all saved lyapunov models in the lyapunov_dir (if False, only the best model is certified).")
    rho_estimate: float | None = config_field(default=None, help="Optional initial rho estimate for the bisection search.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    test_rollout_steps: int = config_field(default=50, help="Closed-loop rollout steps per region center during empirical result testing.")
    cert_bound_scales: list[float] = config_field(
        default_factory=lambda: list(_DEFAULT_CERT_BOUND_SCALES),
        help="Per-dimension scaling applied to policy state bounds to define certification bounds.",
    )
    save_dir: str | None = config_field(default=None, help="Optional directory where certification details and tester results are written.")


def _find_all_lyapunov_dirs(lyapunov_root: Path) -> list[Path]:
    """
    Find all subdirectories in lyapunov_root that contain 'lyapunov_model.pt'.
    """
    lyap_dirs = []
    for subdir in lyapunov_root.glob("**/"):
        if (subdir / "lyapunov_model.pt").is_file():
            lyap_dirs.append(subdir)
    return lyap_dirs


def _build_script_defaults() -> BisectCertifyScriptConfig | list[BisectCertifyScriptConfig]:
    default_policy_dir = discover_latest_policy_dir(_DEFAULT_RESULTS_ROOT)
    default_lyapunov_dir = discover_latest_lyapunov_dir(default_policy_dir)
    return BisectCertifyScriptConfig(
        policy_dir=str(default_policy_dir),
        lyapunov_dir=str(default_lyapunov_dir),
    )


def _build_certification_defaults(
    lyapunov_dir: Path,
) -> LyapunovCertificationConfig:
    training_config = LyapunovTrainingConfig.load(lyapunov_dir / "training_config.json")
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


def _build_lyapunov_func(
    lyap_model,
    device: th.device,
) -> Callable[[NDArray], NDArray]:
    def lyapunov_func(states: NDArray) -> NDArray:
        x = th.as_tensor(states, dtype=th.float32, device=device)
        with th.no_grad():
            values = lyap_model(x)
        return values.detach().cpu().numpy().reshape(-1)

    return lyapunov_func


def parse_args() -> list[tuple[BisectCertifyScriptConfig, LyapunovCertificationConfig]]:
    script_defaults = _build_script_defaults()
    certification_defaults = _build_certification_defaults(Path(script_defaults.lyapunov_dir))
    parser = _build_parser(script_defaults, certification_defaults)
    args = parser.parse_args()
    __logger__.info("Parsed command-line arguments: %s", args)
    script_config = script_defaults.from_namespace(args)

    if script_config.certify_all_lyapunov_models:
        lyapunov_root = Path(script_config.lyapunov_dir)
        lyap_dirs = _find_all_lyapunov_dirs(lyapunov_root)
        configs = []
        for lyap_dir in lyap_dirs:
            script_config = replace(
                script_config,
                lyapunov_dir=str(lyap_dir)
            )
            dir_certification_defaults = _build_certification_defaults(lyap_dir)
            certification_config = dir_certification_defaults.from_namespace(args)
            configs.append((script_config, certification_config))
    else:
        configs = [(
            script_config,
            certification_defaults.from_namespace(args)
        )]

    return configs


def main() -> None:
    configs = parse_args()
    for script_config, certification_config in configs:
        device = th.device(script_config.device)

        policy_dir = Path(script_config.policy_dir).resolve()
        lyapunov_dir = Path(script_config.lyapunov_dir).resolve()

        policy_model = load_policy_model(policy_dir, device)
        lyap_model = load_lyapunov_model(lyapunov_dir, device)

        dyn_model = DoubleIntegratorDynamics(
            dt=policy_model.global_config.dt,
            abcrown_compatible_ops=True,
        ).to(device)
        dyn_model.eval()

        results_path = lyapunov_dir / "training_result.json"
        if not results_path.is_file():
            raise FileNotFoundError(f"Training results file not found at {results_path}. Cannot extract rho estimate. Please provide an initial rho estimate via the --rho_estimate argument or ensure the training results file exists.")
        training_results = LyapunovTrainingResult.load(results_path)
        train_rho_estimate = training_results.rho_estimate
        rho_estimate = script_config.rho_estimate if script_config.rho_estimate is not None else train_rho_estimate * 0.1

        save_dir = (
            Path(script_config.save_dir).resolve()
            if script_config.save_dir is not None
            else (lyapunov_dir / "bisect_certification").resolve()
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        __logger__.info(f"Using rho estimate %.6f for Lyapunov dir %s", rho_estimate, lyapunov_dir)

        certifier = BisectCertifier(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            config=certification_config,
            device=device,
        )

        cert_results = certifier.certify(rho_estimate)
        certifier.save(save_dir)

        cert_bounds = certification_config.cert_bounds
        plot_bounds = [
            (float(cert_bounds[0, i]), float(cert_bounds[1, i]))
            for i in range(cert_bounds.shape[1])
        ]
        plot_path = save_dir / "certification_regions_plot.html"
        lcil_plt.certified_regions_2d(
            certification_result=cert_results,
            state_labels=["$x$", "$v$"],
            bounds=plot_bounds,
            html_path=plot_path,
        )
        mdg_plt.lyapunov(
            lyapunov_func=_build_lyapunov_func(lyap_model, device),
            roa_level=cert_results.rho,
            state_labels=["$x$", "$v$"],
            num_states=2,
            limits=plot_bounds,
            plot_3d=False,
            html_path=plot_path,
        )
        __logger__.info("Saved certification region plot to %s", plot_path)

if __name__ == "__main__":
    main()