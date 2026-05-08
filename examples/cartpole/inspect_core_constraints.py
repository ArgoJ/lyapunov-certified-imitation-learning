from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import torch as th

from lcil.certification import (
    CoreConstraintInspector,
    LyapunovCertificationConfig,
)
from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.utils.base_config import ArgumentParserConfig, config_field

from . import (
    CartpoleDynamics,
    discover_latest_lyapunov_dir,
    discover_latest_policy_dir,
    load_policy_model,
    load_lyapunov_model,
)

__logger__ = logging.getLogger("lcil.examples.cartpole.inspect_core_constraints")

_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "cartpole"


@dataclass(frozen=True)
class CoreInspectScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(
        default=str(discover_latest_policy_dir(_DEFAULT_RESULTS_ROOT)),
        help="Policy run directory containing model.pt."
    )
    lyapunov_dir: str = config_field(
        default=str(discover_latest_lyapunov_dir(_DEFAULT_RESULTS_ROOT)),
        help="Lyapunov run directory containing lyapunov_model.pt."
    )
    device: str = config_field(
        default="cpu",
        help="Torch device string (for example cpu or cuda)."
    )
    show_progress: bool = config_field(
        default=False, 
        help="Show per-constraint ABCrown progress bars."
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
) -> LyapunovCertificationConfig:
    training_config = LyapunovTrainingConfig.load(lyapunov_dir / "training_config.json")
    return LyapunovCertificationConfig.from_training_config(
        training_config,
        cert_bounds=training_config.state_bounds,
        bins_per_dim=2,
        center_refinement_factor=1.0,
        origin_exclusion=0.0,
        cert_method="crown",
        condition_margin=float(training_config.condition_margin),
        suppress_native_output=True,
        batch_size=32,
        abcrown_timeout=60.0,
    )


def parse_args(
    script_defaults: CoreInspectScriptConfig,
    certification_defaults: LyapunovCertificationConfig,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect cartpole Lyapunov core constraints with separate ABCrown checks."
    )
    script_defaults.add_to_argparse(parser)
    certification_defaults.add_to_argparse(
        parser,
        exclude_fields={"state_dim", "cert_bounds"},
    )
    return parser.parse_args()



def main() -> None:
    script_defaults = _build_script_defaults()
    certification_defaults = _build_certification_defaults(Path(script_defaults.lyapunov_dir).resolve())
    args = parse_args(script_defaults, certification_defaults)
    script_config = script_defaults.from_namespace(args)
    certification_config = certification_defaults.from_namespace(args)
    device = th.device(script_config.device)

    policy_dir = Path(script_config.policy_dir).resolve()
    lyapunov_dir = Path(script_config.lyapunov_dir).resolve()

    policy_model = load_policy_model(policy_dir, device)
    lyap_model = load_lyapunov_model(lyapunov_dir, device=device)

    dyn_model = CartpoleDynamics(dt=policy_model.net.global_config.dt).to(device)
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