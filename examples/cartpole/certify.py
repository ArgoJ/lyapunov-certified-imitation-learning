from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
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
_DEFAULT_CERT_BOUND_SCALES = (0.15, 0.15, 0.05, 0.15)


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


def _write_pareto_html(certify_result, output_path: Path) -> None:
    pareto_points = certify_result.pareto_points
    rho_values = [float(point.rho) for point in pareto_points]
    certified_volume = [float(point.volumes.certified_volume) for point in pareto_points]
    unresolved_volume = [float(point.volumes.unresolved_volume) for point in pareto_points]
    unresolved_ratio = [float(point.volumes.unresolved_ratio) for point in pareto_points]
    feasible = [bool(point.feasible) for point in pareto_points]

    marker_colors = ["#2e8b57" if is_feasible else "#c0392b" for is_feasible in feasible]
    hover_text = [
        (
            f"rho={point.rho:.6f}<br>"
            f"feasible={point.feasible}<br>"
            f"rounds={point.refinement_rounds}<br>"
            f"certified_volume={point.volumes.certified_volume:.6f}<br>"
            f"unresolved_volume={point.volumes.unresolved_volume:.6f}<br>"
            f"unresolved_ratio={point.volumes.unresolved_ratio:.6f}"
        )
        for point in pareto_points
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rho_values,
            y=certified_volume,
            mode="lines+markers",
            name="certified volume",
            marker={"color": marker_colors, "size": 9},
            line={"color": "#1f77b4", "width": 3},
            hovertext=hover_text,
            hoverinfo="text",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rho_values,
            y=unresolved_ratio,
            mode="lines+markers",
            name="unresolved ratio",
            marker={"color": marker_colors, "symbol": "diamond", "size": 9},
            line={"color": "#ff7f0e", "width": 3, "dash": "dot"},
            hovertext=hover_text,
            hoverinfo="text",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Bar(
            x=rho_values,
            y=unresolved_volume,
            name="unresolved volume",
            marker={"color": "rgba(214, 39, 40, 0.25)"},
            yaxis="y1",
            hovertext=hover_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="Adaptive Cartpole Pareto Curve",
        template="plotly_white",
        barmode="overlay",
        xaxis={"title": "rho"},
        yaxis={"title": "volume"},
        yaxis2={
            "title": "unresolved ratio",
            "overlaying": "y",
            "side": "right",
            "range": [0.0, max(1.0, max(unresolved_ratio, default=0.0) * 1.05)],
        },
        legend={"orientation": "h", "y": 1.12},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)


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
        description="Adaptive cartpole certification analysis using the adaptive ABCrown/LiRPA pipeline."
    )
    parser.add_argument("--policy-dir", type=Path, default=default_policy_dir)
    parser.add_argument("--lyapunov-dir", type=Path, default=default_lyapunov_dir)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rho-values", nargs="+", type=float, default=None)
    parser.add_argument("--rho-min", type=float, default=None)
    parser.add_argument("--rho-max", type=float, default=None)
    parser.add_argument("--num-points", type=int, default=8)
    parser.add_argument("--spacing", choices=["linear", "geometric"], default="geometric")
    parser.add_argument("--unresolved-tolerance", type=float, default=0.2)
    parser.add_argument("--max-refinement-rounds", type=int, default=2)
    parser.add_argument("--bins-per-dim", nargs="+", type=int, default=[2])
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
    parser.add_argument("--output-html", type=Path, default=None)
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
    lyap_box_summary = _sample_cert_box_values(lyap_model, cert_bounds, device)

    if args.rho_values is None and args.rho_min is None and args.rho_max is None:
        args.rho_min = float(lyap_box_summary["rho_min"])
        args.rho_max = float(lyap_box_summary["rho_max"])

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

    resolved_rho_values = certifier._resolve_rho_values(
        args.rho_values,
        rho_min=None if args.rho_min is None else float(args.rho_min),
        rho_max=None if args.rho_max is None else float(args.rho_max),
        num_points=int(args.num_points),
        spacing=args.spacing,
    )

    __logger__.info("Cartpole adaptive certification setup")
    __logger__.info("  policy dir: %s", policy_dir)
    __logger__.info("  lyapunov dir: %s", lyapunov_dir)
    __logger__.info("  cert bounds: %s", _format_bounds(cert_bounds))
    __logger__.info(
        "  sampled V on cert box: center=%.6f, origin=%.6f, min_positive=%.6f, median_positive=%.6f, max=%.6f",
        float(lyap_box_summary["center_value"]),
        float(lyap_box_summary["origin_value"]),
        float(lyap_box_summary["min_positive"]),
        float(lyap_box_summary["median_positive"]),
        float(lyap_box_summary["max_value"]),
    )
    __logger__.info(
        "  adaptive config: bins=%s, center_refinement=%s, kappa=%.6f, tolerance=%.6f, max_refinement_rounds=%d",
        certification_config.bins_per_dim,
        certification_config.center_refinement_factor,
        float(certification_config.kappa),
        float(args.unresolved_tolerance),
        int(args.max_refinement_rounds),
    )
    __logger__.info(
        "  rho sweep (%d points, %s): %s",
        len(resolved_rho_values),
        args.spacing,
        ", ".join(f"{rho:.6g}" for rho in resolved_rho_values),
    )

    __logger__.info("Step 5/6: run adaptive certify() with unresolved tolerance %.4f", args.unresolved_tolerance)
    certify_result = certifier.certify(
        rho_values=resolved_rho_values,
        unresolved_tolerance=float(args.unresolved_tolerance),
        rho_min=None,
        rho_max=None,
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

    if args.output_html is not None:
        output_html = args.output_html.resolve()
        _write_pareto_html(certify_result, output_html)
        __logger__.info("Wrote adaptive Pareto plot to %s", output_html)

    if args.output_json is not None:
        payload = {
            "policy_dir": str(policy_dir),
            "lyapunov_dir": str(lyapunov_dir),
            "cert_bounds": cert_bounds.tolist(),
            "sampled_lyapunov_box_values": lyap_box_summary,
            "resolved_rho_values": [float(rho) for rho in resolved_rho_values],
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