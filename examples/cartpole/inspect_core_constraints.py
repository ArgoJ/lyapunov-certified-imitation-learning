from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import torch as th
import numpy as np

from lcil.certification import (
    CoreConstraintInspector,
    LyapunovCertificationConfig,
)
from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.utils import GridSearchHelper
from lcil.utils.base_config import ArgumentParserConfig, config_field

from . import (
    CartpoleDynamics,
    discover_latest_lyapunov_dir,
    discover_latest_policy_dir,
    load_mpc_config,
    load_policy_model,
    load_lyapunov_model,
)

__logger__ = logging.getLogger("lcil.examples.cartpole.inspect_core_constraints")

_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "cartpole"
_DEFAULT_CERT_BOUND_SCALES = (0.15, 0.15, 0.05, 0.15)


@dataclass(frozen=True)
class CoreInspectScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(help="Policy run directory containing policy_model.pt.")
    lyapunov_dir: str = config_field(help="Lyapunov run directory containing lyapunov_model.pt.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    show_progress: bool = config_field(default=False, help="Show per-constraint ABCrown progress bars.")
    cert_bound_scales: list[float] = config_field(
        default_factory=lambda: list(_DEFAULT_CERT_BOUND_SCALES),
        help="Per-dimension scaling applied to Lyapunov training bounds before inspection.",
    )


def _build_script_defaults() -> CoreInspectScriptConfig:
    default_policy_dir = discover_latest_policy_dir(_DEFAULT_RESULTS_ROOT)
    default_lyapunov_dir = discover_latest_lyapunov_dir(default_policy_dir)
    return CoreInspectScriptConfig(
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
        lirpa_method="crown",
        condition_margin=float(training_config.condition_margin),
        suppress_native_output=True,
        batch_size=32,
        abcrown_timeout=60.0,
    )


def _build_parser(
    script_defaults: CoreInspectScriptConfig,
    certification_defaults: LyapunovCertificationConfig,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect cartpole Lyapunov core constraints with separate ABCrown checks.",
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
    script_config: CoreInspectScriptConfig,
    script_defaults: CoreInspectScriptConfig,
) -> CoreInspectScriptConfig:
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


def parse_args() -> GridSearchHelper[tuple[CoreInspectScriptConfig, LyapunovCertificationConfig]]:
    script_defaults = _build_script_defaults()
    certification_defaults = _build_certification_defaults(
        Path(script_defaults.lyapunov_dir).resolve(),
        script_defaults.cert_bound_scales,
    )
    parser = _build_parser(script_defaults, certification_defaults)
    args = parser.parse_args()

    configs: list[tuple[CoreInspectScriptConfig, LyapunovCertificationConfig]] = []
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
        __logger__.info("[%d/%d] Inspecting %s", run_idx, total_runs, script_config.lyapunov_dir)
        device = th.device(script_config.device)

        policy_dir = Path(script_config.policy_dir).resolve()
        lyapunov_dir = Path(script_config.lyapunov_dir).resolve()

        policy_model = load_policy_model(policy_dir, device)
        lyap_model = load_lyapunov_model(lyapunov_dir, device)
        mpc_cfg = load_mpc_config(policy_dir)

        dyn_model = CartpoleDynamics(dt=mpc_cfg.dt).to(device)
        dyn_model.eval()

        inspector = CoreConstraintInspector(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            config=certification_config,
            device=device,
        )

        __logger__.info(
            "Running core constraint inspection with bins=%s, center_refinement=%s, origin_exclusion=%s.",
            certification_config.bins_per_dim,
            certification_config.center_refinement_factor,
            certification_config.origin_exclusion,
        )

        inspector.inspect(show_progress=bool(script_config.show_progress))


if __name__ == "__main__":
    main()