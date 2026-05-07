from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch as th

from lcil.certification import (
    BisectCertifier,
    CertificationResultTester,
    LyapunovCertificationConfig,
)
from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.models import NeuralLyapunovCandidate
from lcil.utils.base_config import ArgumentParserConfig, config_field

from . import (
    CartpoleDynamics,
    CartpoleAngleWrapper,
    discover_latest_lyapunov_dir,
    discover_latest_policy_dir,
    load_policy_model,
)

__logger__ = logging.getLogger("lcil.examples.cartpole.certify")

_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "cartpole"
_DEFAULT_CERT_BOUND_SCALES = (0.15, 0.15, 0.05, 0.15)


@dataclass(frozen=True)
class BisectCertifyScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(help="Policy run directory containing model.pt.")
    lyapunov_dir: str = config_field(help="Lyapunov run directory containing lyapunov_model.pt.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    rho_estimate: float | None = config_field(default=None, help="Optional initial rho estimate for the bisection search.")
    test_rollout_steps: int = config_field(default=50, help="Closed-loop rollout steps per region center during empirical result testing.")
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
        description="Bisection-based cartpole certification analysis with follow-up empirical rollout testing."
    )
    script_defaults.add_to_argparse(parser)
    certification_defaults.add_to_argparse(
        parser,
        exclude_fields={"state_dim", "cert_bounds"},
    )
    return parser


def parse_args() -> tuple[BisectCertifyScriptConfig, LyapunovCertificationConfig]:
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
    certification_defaults = _build_certification_defaults(bootstrap_lyapunov_dir)

    parser = _build_parser(script_defaults, certification_defaults)
    args = parser.parse_args()
    return (
        script_defaults.from_namespace(args),
        certification_defaults.from_namespace(args),
    )



def _load_lyapunov_model(
    lyapunov_dir: Path,
    device: th.device,
) -> NeuralLyapunovCandidate:
    checkpoint_path = lyapunov_dir / "lyapunov_model.pt"
    try:
        lyap_model = NeuralLyapunovCandidate.load(
            checkpoint_path,
            map_location=device,
        ).to(device)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Lyapunov checkpoint '{checkpoint_path}' is not compatible with the new model save/load format. "
            "Re-save the model with NeuralLyapunovCandidate.save before using this script."
        ) from exc
    lyap_model.eval()
    return lyap_model


def main() -> None:
    script_config, certification_config = parse_args()
    device = th.device(script_config.device)

    policy_dir = Path(script_config.policy_dir).resolve()
    lyapunov_dir = Path(script_config.lyapunov_dir).resolve()

    policy_model = load_policy_model(policy_dir, device)
    lyap_model = _load_lyapunov_model(lyapunov_dir, device=device)

    dyn_model = CartpoleDynamics(dt=policy_model.net.global_config.dt).to(device)
    dyn_model.eval()

    rho_estimate = script_config.rho_estimate

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
        cert_result=cert_results,
        rollout_steps=int(script_config.test_rollout_steps),
    )
    tester_results_path = save_dir / "certification_tester_results.json"
    test_results.save(tester_results_path)
    __logger__.info("Saved certification tester results to %s", tester_results_path)

if __name__ == "__main__":
    main()