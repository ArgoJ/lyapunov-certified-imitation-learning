from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch as th

from lcil.certification.adaptive import AdaptiveCertifier, AdaptiveCertificationConfig
from lcil.imitation_learning_mlp import MLPPolicy
from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.models import NeuralLyapunovCandidate
from lcil.utils import MLP

from cartpole_dyn import CartpoleDynamics
from model import CartpoleAngleWrapper

__logger__ = logging.getLogger(__name__)

_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "cartpole"
_DEFAULT_CERT_BOUND_SCALES = (0.05, 0.05, 0.02, 0.05)


def _discover_latest_policy_dir(results_root: Path) -> Path:
    candidates = sorted(
        path for path in results_root.iterdir() if path.is_dir() and (path / "model.pt").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No cartpole policy run with model.pt found under '{results_root}'.")
    return candidates[-1]


def _discover_latest_lyapunov_dir(policy_dir: Path) -> Path:
    lyapunov_root = policy_dir / "lyapunov"
    if not lyapunov_root.exists():
        raise FileNotFoundError(f"No lyapunov directory found under '{policy_dir}'.")

    candidates = sorted(checkpoint.parent for checkpoint in lyapunov_root.rglob("lyapunov_model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No Lyapunov checkpoint 'lyapunov_model.pt' found under '{lyapunov_root}'."
        )
    return candidates[-1]


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


def _infer_layer_dims_from_checkpoint(path: Path) -> list[int]:
    state_dict = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state dict checkpoint at '{path}', got {type(state_dict).__name__}.")

    weight_items = []
    for key, tensor in state_dict.items():
        if key.startswith("feature_net.net.net.") and key.endswith(".weight"):
            layer_idx = int(key.split(".")[3])
            weight_items.append((layer_idx, tensor))

    if not weight_items:
        raise ValueError(f"Could not infer feature-net layers from checkpoint '{path}'.")

    weight_items.sort(key=lambda item: item[0])
    layer_dims = [int(weight_items[0][1].shape[1])]
    layer_dims.extend(int(weight.shape[0]) for _, weight in weight_items)
    return layer_dims


def _load_policy_model(policy_dir: Path, device: th.device) -> CartpoleAngleWrapper:
    policy_checkpoint = policy_dir / "model.pt"
    feature_net = MLPPolicy.load(policy_checkpoint, map_location=device)
    policy_model = CartpoleAngleWrapper(feature_net=feature_net).to(device)
    policy_model.eval()
    return policy_model


def _load_lyapunov_model(
    lyapunov_dir: Path,
    state_dim: int,
    device: th.device,
    *,
    hidden_activation: str,
    eps: float,
) -> NeuralLyapunovCandidate:
    checkpoint_path = lyapunov_dir / "lyapunov_model.pt"
    layer_dims = _infer_layer_dims_from_checkpoint(checkpoint_path)
    activations = [hidden_activation] * (len(layer_dims) - 2) + ["identity"]

    feature_net = MLP(layer_dims, activations).to(device)
    wrapped_feature_net = CartpoleAngleWrapper(feature_net=feature_net).to(device)
    lyap_model = NeuralLyapunovCandidate(
        feature_net=wrapped_feature_net,
        state_dim=state_dim,
        eps=eps,
    ).to(device)
    state_dict = th.load(checkpoint_path, map_location=device, weights_only=True)
    lyap_model.load_state_dict(state_dict)
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


def _point_to_dict(point) -> dict[str, object]:
    return {
        "rho": float(point.rho),
        "feasible": bool(point.feasible),
        "refinement_rounds": int(point.refinement_rounds),
        "certified_volume": float(point.volumes.certified_volume),
        "failed_inside_volume": float(point.volumes.failed_inside_volume),
        "boundary_volume": float(point.volumes.boundary_volume),
        "outside_volume": float(point.volumes.outside_volume),
        "relevant_volume": float(point.volumes.relevant_volume),
        "unresolved_volume": float(point.volumes.unresolved_volume),
        "unresolved_ratio": float(point.volumes.unresolved_ratio),
        "num_inside_regions": int(len(point.result.inside_regions)),
        "num_boundary_regions": int(len(point.result.boundary_regions)),
        "num_outside_regions": int(len(point.result.outside_regions)),
        "num_certified_inside_regions": int(len(point.result.certified_inside_regions)),
        "num_failed_inside_regions": int(len(point.result.failed_inside_regions)),
    }


def parse_args() -> argparse.Namespace:
    default_policy_dir = _discover_latest_policy_dir(_DEFAULT_RESULTS_ROOT)
    default_lyapunov_dir = _discover_latest_lyapunov_dir(default_policy_dir)

    parser = argparse.ArgumentParser(
        description="Minimal adaptive cartpole certification example using the adaptive ABCrown/LiRPA pipeline."
    )
    parser.add_argument("--policy-dir", type=Path, default=default_policy_dir)
    parser.add_argument("--lyapunov-dir", type=Path, default=default_lyapunov_dir)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rho-values", nargs="+", type=float, default=None)
    parser.add_argument("--rho-min", type=float, default=0.5)
    parser.add_argument("--rho-max", type=float, default=2.0)
    parser.add_argument("--num-points", type=int, default=4)
    parser.add_argument("--spacing", choices=["linear", "geometric"], default="geometric")
    parser.add_argument("--unresolved-tolerance", type=float, default=0.2)
    parser.add_argument("--max-refinement-rounds", type=int, default=1)
    parser.add_argument("--bins-per-dim", nargs="+", type=int, default=[1])
    parser.add_argument("--center-refinement-factor", nargs="+", type=float, default=[1.0])
    parser.add_argument("--origin-exclusion", type=float, default=0.0)
    parser.add_argument("--cert-bound-scales", nargs="+", type=float, default=list(_DEFAULT_CERT_BOUND_SCALES))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lirpa-bound-method", type=str, default="crown")
    parser.add_argument("--lyap-hidden-activation", type=str, default="tanh")
    parser.add_argument("--lyap-eps", type=float, default=0.1)
    parser.add_argument("--condition-margin", type=float, default=None)
    parser.add_argument("--show-solver-output", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    device = th.device(args.device)

    policy_dir = args.policy_dir.resolve()
    lyapunov_dir = args.lyapunov_dir.resolve()
    lyapunov_training_config = LyapunovTrainingConfig.load(lyapunov_dir / "training_config.json")
    state_dim = int(lyapunov_training_config.state_dim)

    bins_per_dim = _normalize_length(
        [int(value) for value in args.bins_per_dim],
        state_dim,
        name="bins_per_dim",
    )
    center_refinement_factor = _normalize_length(
        [float(value) for value in args.center_refinement_factor],
        state_dim,
        name="center_refinement_factor",
    )
    cert_bound_scales = _normalize_length(
        [float(value) for value in args.cert_bound_scales],
        state_dim,
        name="cert_bound_scales",
    )

    __logger__.info("Step 1/6: load policy checkpoint from %s", policy_dir)
    policy_model = _load_policy_model(policy_dir, device)

    __logger__.info("Step 2/6: load Lyapunov checkpoint from %s", lyapunov_dir)
    lyap_model = _load_lyapunov_model(
        lyapunov_dir,
        state_dim=state_dim,
        device=device,
        hidden_activation=args.lyap_hidden_activation,
        eps=float(args.lyap_eps),
    )

    __logger__.info("Step 3/6: build closed-loop cartpole dynamics")
    dyn_model = CartpoleDynamics(dt=policy_model.net.global_config.dt).to(device)
    dyn_model.eval()

    __logger__.info("Step 4/6: assemble adaptive certification config")
    cert_bounds = _build_cert_bounds(policy_model, cert_bound_scales)
    certification_config = AdaptiveCertificationConfig(
        state_dim=state_dim,
        cert_bounds=cert_bounds,
        kappa=float(lyapunov_training_config.kappa),
        bins_per_dim=tuple(int(value) for value in bins_per_dim),
        center_refinement_factor=tuple(float(value) for value in center_refinement_factor),
        origin_exclusion=float(args.origin_exclusion),
        lirpa_bound_method=args.lirpa_bound_method,
        sublevel_tolerance=float(lyapunov_training_config.condition_tolerance),
        condition_tolerance=float(lyapunov_training_config.condition_tolerance),
        condition_margin=(
            float(lyapunov_training_config.condition_margin)
            if args.condition_margin is None
            else float(args.condition_margin)
        ),
        suppress_native_output=not args.show_solver_output,
        batch_size=int(args.batch_size),
    )

    certifier = AdaptiveCertifier(
        policy_model=policy_model,
        lyap_model=lyap_model,
        dyn_model=dyn_model,
        config=certification_config,
        device=device,
    )

    __logger__.info("Step 5/6: run adaptive certify() with unresolved tolerance %.4f", args.unresolved_tolerance)
    certify_result = certifier.certify(
        rho_values=args.rho_values,
        unresolved_tolerance=float(args.unresolved_tolerance),
        rho_min=float(args.rho_min),
        rho_max=float(args.rho_max),
        num_points=int(args.num_points),
        spacing=args.spacing,
        max_refinement_rounds=int(args.max_refinement_rounds),
        reset_regions=True,
    )

    __logger__.info("Step 6/6: report Pareto points and selected rho")
    for point in certify_result.pareto_points:
        __logger__.info(
            "rho=%.6f | feasible=%s | rounds=%d | certified=%.6f | unresolved=%.6f | unresolved_ratio=%.6f",
            float(point.rho),
            point.feasible,
            int(point.refinement_rounds),
            float(point.volumes.certified_volume),
            float(point.volumes.unresolved_volume),
            float(point.volumes.unresolved_ratio),
        )

    if certify_result.best_point is None:
        __logger__.warning(
            "No rho in the sampled curve satisfied unresolved_ratio <= %.6f.",
            float(args.unresolved_tolerance),
        )
    else:
        __logger__.info(
            "Selected best rho %.6f with unresolved_ratio %.6f.",
            float(certify_result.best_point.rho),
            float(certify_result.best_point.volumes.unresolved_ratio),
        )

    if args.output_json is not None:
        payload = {
            "policy_dir": str(policy_dir),
            "lyapunov_dir": str(lyapunov_dir),
            "cert_bounds": cert_bounds.tolist(),
            "unresolved_tolerance": float(args.unresolved_tolerance),
            "best_rho": float(certify_result.best_rho),
            "best_point": None if certify_result.best_point is None else _point_to_dict(certify_result.best_point),
            "pareto_points": [_point_to_dict(point) for point in certify_result.pareto_points],
        }
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        __logger__.info("Wrote adaptive certification summary to %s", output_path)


if __name__ == "__main__":
    main()