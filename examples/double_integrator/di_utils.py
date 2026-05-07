import torch as th

from pathlib import Path

from lcil.imitation_learning_mlp import MLPPolicy


def discover_latest_data(results_root: Path) -> Path:
	candidates = sorted(
		path for path in results_root.iterdir() if path.is_dir() and any(path.glob("*data.hdf5"))
	)
	if not candidates:
		raise FileNotFoundError(f"No data with 'data.hdf5' found under '{results_root}'.")
	root = candidates[-1]
	return sorted(root.glob("*data.hdf5"))[-1]


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


def load_policy_model(policy_dir: Path, device: th.device) -> MLPPolicy:
	policy_checkpoint = policy_dir / "model.pt"
	policy_model = MLPPolicy.load(policy_checkpoint, map_location=device).to(device)
	policy_model.eval()
	return policy_model


def default_model_path() -> str:
	results_root = Path(__file__).resolve().parents[2] / "results" / "double_integrator"
	return str(discover_latest_policy_dir(results_root) / "model.pt")


def default_dataset_path() -> str:
	results_root = Path(__file__).resolve().parents[2] / "results" / "double_integrator" / "data"
	return str(discover_latest_data(results_root))