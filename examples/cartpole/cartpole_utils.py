import torch as th

from pathlib import Path

from lcil.imitation_learning_mlp import MLPPolicy
from lcil.lyapunov_learning import NeuralLyapunovCandidate

from .model import CartpoleAngleWrapper


def discover_latest_policy_dir(results_root: Path) -> Path:
    candidates = sorted(
        path for path in results_root.iterdir() if path.is_dir() and (path / "model.pt").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No policy run with model.pt found under '{results_root}'.")
    return candidates[-1]


def discover_latest_lyapunov_dir(policy_dir: Path) -> Path:
    lyapunov_root = policy_dir / "lyapunov"
    if not lyapunov_root.exists():
        raise FileNotFoundError(f"No lyapunov directory found under '{policy_dir}'.")

    candidates = sorted(checkpoint.parent for checkpoint in lyapunov_root.rglob("lyapunov_model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No Lyapunov checkpoint 'lyapunov_model.pt' found under '{lyapunov_root}'."
        )
    return candidates[-1]


def load_old_policy_model(policy_dir: Path, device: th.device) -> CartpoleAngleWrapper:
    policy_checkpoint = policy_dir / "model.pt"
    try:
        feature_net = MLPPolicy.load(policy_checkpoint, map_location=device)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        raise ValueError(
            f"Policy checkpoint '{policy_checkpoint}' is not compatible with the new model save/load format. "
            "Re-save the model with MLPPolicy.save before using this script."
        )
    policy_model = CartpoleAngleWrapper(feature_net=feature_net).to(device)
    policy_model.eval()
    return policy_model

def load_policy_model(policy_dir: Path, device: th.device) -> CartpoleAngleWrapper:
    policy_checkpoint = policy_dir / "model.pt"
    try:
        policy_model = CartpoleAngleWrapper.load(policy_checkpoint, map_location=device).to(device)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return load_old_policy_model(policy_dir, device)
    policy_model.eval()
    return policy_model


def default_model_path() -> str:
    results_root = Path(__file__).resolve().parents[2] / "results" / "cartpole"
    return str(discover_latest_policy_dir(results_root) / "model.pt")

def load_lyapunov_model(lyapunov_dir: Path, device: th.device) -> NeuralLyapunovCandidate:
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