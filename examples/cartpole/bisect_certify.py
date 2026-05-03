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


def _normalize_length(
    values: Sequence[float] | Sequence[int],
    state_dim: int,
    *,
    name: str,
) -> tuple[float, ...] | tuple[int, ...]:
    if len(values) == 1:
        repeated = tuple(values[0] for _ in range(state_dim))
        return repeated
    if len(values) != state_dim:
        raise ValueError(f"{name} must have length 1 or {state_dim}, got {len(values)}.")
    return tuple(values)


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
        cert_bounds=training_config.state_bounds,
        bins_per_dim=2,
        center_refinement_factor=1.0,
        origin_exclusion=0.0,
        cert_method="crown",
        condition_margin=float(training_config.condition_margin),
        suppress_native_output=True,
        batch_size=32,
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


def _build_cert_bounds(
    policy_model: CartpoleAngleWrapper,
    scales: Sequence[float],
) -> np.ndarray:
    feature_net = policy_model.net
    state_bounds = np.vstack(
        [feature_net.global_config.constraints.lbx, feature_net.global_config.constraints.ubx]
    ).astype(np.float32)
    cert_scales = np.asarray(scales, dtype=np.float32).reshape(1, -1)
    return state_bounds * cert_scales


def _sample_cert_box_values(
    lyap_model: NeuralLyapunovCandidate,
    cert_bounds: np.ndarray,
    device: th.device,
) -> dict[str, float]:
    bounds = th.as_tensor(cert_bounds, dtype=th.float32, device=device)
    per_dim_endpoints = [bounds[:, idx] for idx in range(bounds.shape[1])]
    corners = th.cartesian_prod(*per_dim_endpoints)
    center = bounds.mean(dim=0, keepdim=True)
    clipped_origin = th.clamp(th.zeros_like(center), min=bounds[0], max=bounds[1])
    samples = th.cat([corners, center, clipped_origin], dim=0)

    with th.no_grad():
        values = lyap_model(samples).reshape(-1)

    positive_values = values[values > 0.0]
    max_value = float(values.max().item()) if values.numel() > 0 else 0.0
    min_positive = float(positive_values.min().item()) if positive_values.numel() > 0 else 0.0
    median_positive = float(positive_values.median().item()) if positive_values.numel() > 0 else 0.0

    rho_max = max(1e-2, max_value)
    rho_min = max(1e-3, 0.05 * rho_max)

    return {
        "num_samples": float(samples.shape[0]),
        "center_value": float(values[-2].item()),
        "origin_value": float(values[-1].item()),
        "min_positive": min_positive,
        "median_positive": median_positive,
        "max_value": max_value,
        "rho_min": rho_min,
        "rho_max": rho_max,
    }


def _format_bounds(bounds: np.ndarray) -> str:
    lb = ", ".join(f"{float(value):.4g}" for value in bounds[0].tolist())
    ub = ", ".join(f"{float(value):.4g}" for value in bounds[1].tolist())
    return f"lb=[{lb}], ub=[{ub}]"


def main() -> None:
    script_config, certification_config = parse_args()
    device = th.device(script_config.device)

    policy_dir = Path(script_config.policy_dir).resolve()
    lyapunov_dir = Path(script_config.lyapunov_dir).resolve()
    lyapunov_training_config = LyapunovTrainingConfig.load(lyapunov_dir / "training_config.json")
    state_dim = int(lyapunov_training_config.state_dim)

    cert_bound_scales = _normalize_length(
        [float(value) for value in script_config.cert_bound_scales],
        state_dim,
        name="cert_bound_scales",
    )

    policy_model = load_policy_model(policy_dir, device)
    lyap_model = _load_lyapunov_model(lyapunov_dir, device=device)

    dyn_model = CartpoleDynamics(dt=policy_model.net.global_config.dt).to(device)
    dyn_model.eval()

    cert_bounds = _build_cert_bounds(policy_model, cert_bound_scales)
    lyap_box_summary = _sample_cert_box_values(lyap_model, cert_bounds, device)

    certification_config = replace(
        certification_config,
        state_dim=state_dim,
        cert_bounds=cert_bounds,
    )

    if script_config.rho_estimate is None:
        rho_estimate = max(
            float(lyap_box_summary["median_positive"]),
            float(lyap_box_summary["rho_min"]),
            1e-3,
        )
    else:
        rho_estimate = max(float(script_config.rho_estimate), 1e-3)

    save_dir = (
        Path(script_config.save_dir).resolve()
        if script_config.save_dir is not None
        else (lyapunov_dir / "bisect_certification").resolve()
    )

    __logger__.info("Cartpole bisect certification setup")
    __logger__.info("  policy dir: %s", policy_dir)
    __logger__.info("  lyapunov dir: %s", lyapunov_dir)
    __logger__.info("  save dir: %s", save_dir)
    __logger__.info("  cert bounds: %s", _format_bounds(cert_bounds))
    __logger__.info(
        "  sampled V on cert box: center=%.6f, origin=%.6f, min_positive=%.6f, median_positive=%.6f, max=%.6f",
        float(lyap_box_summary["center_value"]),
        float(lyap_box_summary["origin_value"]),
        float(lyap_box_summary["min_positive"]),
        float(lyap_box_summary["median_positive"]),
        float(lyap_box_summary["max_value"]),
    )
    __logger__.info("  using rho estimate %.6f", rho_estimate)

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