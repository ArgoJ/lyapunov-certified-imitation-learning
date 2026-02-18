import argparse
from pathlib import Path

import torch as th

from lyapunov_certified_imitation_learning.imitation_learning_mlp import MLPPolicy
from lyapunov_certified_imitation_learning.utils.package_logger import get_package_logger


__logger__ = get_package_logger(__name__)


def _parse_int_list(csv_value: str) -> list[int]:
    values = [part.strip() for part in csv_value.split(",") if part.strip()]
    if not values:
        raise ValueError("layer_sizes cannot be empty.")
    return [int(v) for v in values]


def _parse_str_list(csv_value: str) -> list[str]:
    values = [part.strip() for part in csv_value.split(",") if part.strip()]
    if not values:
        raise ValueError("activations cannot be empty.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a legacy MLPPolicy checkpoint (raw state_dict) to metadata-aware checkpoint format.",
    )
    parser.add_argument(
        "--legacy-path",
        type=str,
        default="results/models/double_integrator_policy.pt",
        help="Path to legacy checkpoint saved as raw state_dict.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/models/double_integrator_policy_with_metadata.pt",
        help="Path for converted checkpoint.",
    )
    parser.add_argument(
        "--layer-sizes",
        type=str,
        default="2,16,16,1",
        help="Comma-separated model layer sizes for the legacy checkpoint.",
    )
    parser.add_argument(
        "--activations",
        type=str,
        default="relu,tanh,identity",
        help="Comma-separated activations for the legacy checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading/checking checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = th.device(args.device)

    legacy_path = Path(args.legacy_path)
    output_path = Path(args.output_path)
    layer_sizes = _parse_int_list(args.layer_sizes)
    activations = _parse_str_list(args.activations)

    if len(activations) != len(layer_sizes) - 1:
        raise ValueError("activations length must equal len(layer_sizes) - 1.")

    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy checkpoint not found at '{legacy_path}'.")

    checkpoint = th.load(legacy_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported legacy checkpoint format. Expected a dictionary state_dict.")

    if "state_dict" in checkpoint and "layer_sizes" in checkpoint and "activations" in checkpoint:
        __logger__.info("Checkpoint already has metadata; validating and rewriting to %s", output_path)
        policy_model = MLPPolicy.load(legacy_path, map_location=device)
        policy_model.save(output_path)
    else:
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        u_min = state_dict.get("_u_min", None)
        u_max = state_dict.get("_u_max", None)

        policy_model = MLPPolicy(
            layer_sizes=layer_sizes,
            activations=activations,
            u_min=u_min,
            u_max=u_max,
        )
        policy_model.load_state_dict(state_dict, strict=True)
        policy_model.save(output_path)

    reloaded = MLPPolicy.load(output_path, map_location=device)
    reloaded.eval()

    __logger__.info("Converted checkpoint written to %s", output_path)
    __logger__.info(
        "Reload test passed (layer_sizes=%s, activations=%s)",
        reloaded.layer_sizes,
        reloaded.activations,
    )


if __name__ == "__main__":
    main()