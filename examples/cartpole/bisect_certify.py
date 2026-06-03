from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import torch as th
import numpy as np

from lcil.certification import (
    BisectCertifier,
    CertificationResultTester,
    LyapunovCertificationConfig,
)
from lcil.lyapunov_learning import LyapunovTrainingConfig, LyapunovTrainingResult
from lcil.utils import GridSearchHelper
from lcil.utils.base_config import ArgumentParserConfig, config_field

from . import (
    CartpoleDynamics,
    discover_latest_lyapunov_dir,
    discover_latest_policy_dir,
    load_lyapunov_model,
    load_policy_model,
)
from ..constants import TRAINING_RESULTS_FILENAME

__logger__ = logging.getLogger("lcil.examples.cartpole.certify")

_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "cartpole"
_DEFAULT_CERT_BOUND_SCALES = (0.15, 0.15, 0.05, 0.15)


@dataclass(frozen=True)
class BisectCertifyScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(help="Policy run directory containing policy_model.pt.")
    lyapunov_dir: str = config_field(help="Lyapunov run directory containing lyapunov_model.pt.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    rho_estimate: float | None = config_field(default=None, help="Optional initial rho estimate for the bisection search.")
    test_rollout_steps: int = config_field(default=50, help="Closed-loop rollout steps per region center during empirical result testing.")
    test_sample_size: int = config_field(
        default=256,
        help="Number of states sampled near the rho-sublevel boundary for empirical testing.",
    )
    cert_bound_scales: list[float] = config_field(
        default_factory=lambda: list(_DEFAULT_CERT_BOUND_SCALES),
        help="Per-dimension scaling applied to policy state bounds to define certification bounds.",
    )
    save_dir: str | None = config_field(default=None, help="Optional directory where certification details and tester results are written.")


def _build_script_defaults() -> BisectCertifyScriptConfig:
    default_policy_dir = discover_latest_policy_dir(_DEFAULT_RESULTS_ROOT)
    default_lyapunov_dir = discover_latest_lyapunov_dir(default_policy_dir)
    return BisectCertifyScriptConfig(
        policy_dir=str(default_policy_dir),
        lyapunov_dir=str(default_lyapunov_dir),
    )


def _build_certification_defaults(
    lyapunov_dir: Path,
    cert_bound_scales: list[float],
) -> LyapunovCertificationConfig:
    training_config = LyapunovTrainingConfig.load(lyapunov_dir / "training_config.json")
    cert_bounds = _scale_cert_bounds(training_config.state_bounds, cert_bound_scales)
    return LyapunovCertificationConfig.from_training_config(
        training_config,
        cert_bounds=cert_bounds,
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
        description="Bisection-based cartpole certification analysis with follow-up empirical rollout testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    script_defaults.add_to_argparse(parser)
    certification_defaults.add_to_argparse(
        parser,
        exclude_fields={"state_dim", "cert_bounds"},
        suppress_defaults=True,
    )
    return parser


def _scale_cert_bounds(state_bounds: np.ndarray, cert_bound_scales: list[float]) -> np.ndarray:
    scales = np.asarray(cert_bound_scales, dtype=float)
    if scales.shape != (state_bounds.shape[1],):
        raise ValueError(
            "cert_bound_scales must contain exactly one scale per state dimension. "
            f"Expected {state_bounds.shape[1]}, got {scales.shape[0]}."
        )
    return np.asarray(state_bounds, dtype=float) * scales.reshape(1, -1)


def _resolve_script_config(
    script_config: BisectCertifyScriptConfig,
    script_defaults: BisectCertifyScriptConfig,
) -> BisectCertifyScriptConfig:
    policy_dir = Path(script_config.policy_dir).resolve()
    lyapunov_dir = Path(script_config.lyapunov_dir).resolve()
    default_policy_dir = Path(script_defaults.policy_dir).resolve()
    default_lyapunov_dir = Path(script_defaults.lyapunov_dir).resolve()
    if policy_dir != default_policy_dir and lyapunov_dir == default_lyapunov_dir:
        lyapunov_dir = discover_latest_lyapunov_dir(policy_dir)
    return replace(
        script_config,
        policy_dir=str(policy_dir),
        lyapunov_dir=str(lyapunov_dir),
    )


def parse_args() -> GridSearchHelper[tuple[BisectCertifyScriptConfig, LyapunovCertificationConfig]]:
    script_defaults = _build_script_defaults()

    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    script_defaults.add_to_argparse(
        bootstrap_parser,
        include_fields={"policy_dir", "lyapunov_dir"},
    )
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    bootstrap_config = script_defaults.from_namespace(bootstrap_args)

    bootstrap_policy_dir = Path(bootstrap_config.policy_dir).resolve()
    bootstrap_lyapunov_dir = Path(bootstrap_config.lyapunov_dir).resolve()
    default_policy_dir = Path(script_defaults.policy_dir).resolve()
    default_lyapunov_dir = Path(script_defaults.lyapunov_dir).resolve()
    if bootstrap_policy_dir != default_policy_dir and bootstrap_lyapunov_dir == default_lyapunov_dir:
        bootstrap_lyapunov_dir = discover_latest_lyapunov_dir(bootstrap_policy_dir)

    script_defaults = replace(
        script_defaults,
        policy_dir=str(bootstrap_policy_dir),
        lyapunov_dir=str(bootstrap_lyapunov_dir),
    )
    certification_defaults = _build_certification_defaults(
        bootstrap_lyapunov_dir,
        script_defaults.cert_bound_scales,
    )

    parser = _build_parser(script_defaults, certification_defaults)
    args = parser.parse_args()
    configs: list[tuple[BisectCertifyScriptConfig, LyapunovCertificationConfig]] = []
    for script_config in script_defaults.iter_from_namespace(args):
        resolved_script_config = _resolve_script_config(script_config, script_defaults)
        dir_certification_defaults = _build_certification_defaults(
            Path(resolved_script_config.lyapunov_dir),
            resolved_script_config.cert_bound_scales,
        )
        for certification_config in dir_certification_defaults.iter_from_namespace(args):
            configs.append((resolved_script_config, certification_config))
    return GridSearchHelper(configs)

def main() -> None:
    sweep = parse_args()
    total_runs = len(sweep.configs)
    for run_idx, (script_config, certification_config) in enumerate(sweep.configs, start=1):
        __logger__.info("[%d/%d] Certifying %s", run_idx, total_runs, script_config.lyapunov_dir)
        device = th.device(script_config.device)

        policy_dir = Path(script_config.policy_dir).resolve()
        lyapunov_dir = Path(script_config.lyapunov_dir).resolve()

        policy_model = load_policy_model(policy_dir, device)
        lyap_model = load_lyapunov_model(lyapunov_dir, device)

        dyn_model = CartpoleDynamics(dt=policy_model.net.global_config.dt).to(device)
        dyn_model.eval()

        results_path = lyapunov_dir / TRAINING_RESULTS_FILENAME
        if not results_path.is_file():
            raise FileNotFoundError(
                f"Training results file not found at {results_path}. Cannot extract rho estimate. "
                "Please provide an initial rho estimate via --rho-estimate or ensure the training result exists."
            )
        training_results = LyapunovTrainingResult.load(results_path)
        train_rho_estimate = training_results.rho_estimate
        rho_estimate = script_config.rho_estimate if script_config.rho_estimate is not None else train_rho_estimate

        save_dir = (
            Path(script_config.save_dir).resolve()
            if script_config.save_dir is not None
            else (lyapunov_dir / "bisect_certification").resolve()
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        __logger__.info("Using rho estimate %.6f", rho_estimate)

        certifier = BisectCertifier(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            config=certification_config,
            device=device,
        )

        cert_results = certifier.certify(rho_estimate)
        certifier.save(save_dir)

        cert_tester = CertificationResultTester(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            config=certification_config,
            device=device,
        )
        test_results = cert_tester.test_result(
            rho=float(cert_results.rho),
            sample_size=int(script_config.test_sample_size),
            rollout_steps=int(script_config.test_rollout_steps),
        )
        test_results.save(save_dir)
        __logger__.info("Saved certification tester results to %s", save_dir)

    

if __name__ == "__main__":
    main()