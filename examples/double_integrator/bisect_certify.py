from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch as th

from lcil.certification import (
    BisectCertifier,
    LyapunovCertificationConfig,
)
from lcil.lyapunov_learning import LyapunovTrainingConfig
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
    rho_estimate: float = config_field(help="Optional initial rho estimate for the bisection search.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
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
        rho_estimate=1.0,
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
    )
    return parser


def parse_args() -> tuple[BisectCertifyScriptConfig, LyapunovCertificationConfig]:
    script_defaults = _build_script_defaults()
    certification_defaults = _build_certification_defaults(Path(script_defaults.lyapunov_dir))

    parser = _build_parser(script_defaults, certification_defaults)
    args = parser.parse_args()
    return (
        script_defaults.from_namespace(args),
        certification_defaults.from_namespace(args),
    )

def main() -> None:
    script_config, certification_config = parse_args()
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

    cert_bounds = certification_config.cert_bounds
    plot_path = save_dir / "certification_regions_plot.html"
    lcil_plt.certified_regions_2d(
        certification_result=cert_results,
        state_indices=[0, 1],
        state_labels=["$x$", "$v$"],
        bounds=[
            (float(cert_bounds[0][0]), float(cert_bounds[1][0])),
            (float(cert_bounds[0][1]), float(cert_bounds[1][1])),
        ],
        html_path=plot_path,
    )
    __logger__.info("Saved certification region plot to %s", plot_path)

if __name__ == "__main__":
    main()