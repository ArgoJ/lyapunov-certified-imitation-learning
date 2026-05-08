import torch as th
import torch.nn as nn
import logging

from pathlib import Path
from typing import Callable, TypeVar


__logger__ = logging.getLogger("lcil.examples.example_utils")


ModelT = TypeVar("ModelT", bound=nn.Module)


_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def discover_latest_model_dir(results_root: Path, checkpoint_name: str) -> Path:
    candidates = sorted(checkpoint.parent for checkpoint in results_root.rglob(checkpoint_name))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint '{checkpoint_name}' found under '{results_root}'.")
    return candidates[-1]


def discover_latest_policy_dir(results_root: Path) -> Path:
    return discover_latest_model_dir(results_root, "model.pt")


def discover_latest_lyapunov_dir(policy_dir: Path) -> Path:
    lyapunov_root = policy_dir / "lyapunov"
    if not lyapunov_root.exists():
        raise FileNotFoundError(f"No lyapunov directory found under '{policy_dir}'.")

    return discover_latest_model_dir(lyapunov_root, "lyapunov_model.pt")


def discover_latest_dataset_path(results_root: Path | str, dataset_pattern: str = "*.hdf5") -> Path:
    resolved_results_root = Path(results_root)
    candidates = sorted(
        dataset_path
        for dataset_path in resolved_results_root.rglob(dataset_pattern)
        if dataset_path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No dataset matching '{dataset_pattern}' found under '{resolved_results_root}'."
        )
    return candidates[-1]


def resolve_dataset_path(
    dataset_path: Path | str | None,
    results_root: Path | str,
    dataset_pattern: str = "*.hdf5",
) -> Path:
    if dataset_path is None or str(dataset_path) == "":
        return discover_latest_dataset_path(results_root, dataset_pattern)

    resolved_dataset_path = Path(dataset_path)
    if resolved_dataset_path.is_dir():
        return discover_latest_dataset_path(resolved_dataset_path, dataset_pattern)
    return resolved_dataset_path


class GenericModelLoader:
    def __init__(self, checkpoint_name: str):
        self.checkpoint_name = checkpoint_name

    def _resolve_model_dir(
        self,
        model_dir: Path | str | None,
        results_root: Path | str | None = None,
    ) -> Path:
        if model_dir is not None:
            return Path(model_dir)
        resolved_results_root = Path(results_root) if results_root is not None else _RESULTS_ROOT
        return discover_latest_model_dir(resolved_results_root, self.checkpoint_name)

    def __call__(
        self,
        model_cls: type[ModelT],
        model_dir: Path | str | None,
        device: th.device,
        results_root: Path | str | None = None,
    ) -> ModelT:
        resolved_model_dir = self._resolve_model_dir(model_dir, results_root=results_root)
        checkpoint_path = resolved_model_dir / self.checkpoint_name
        try:
            model = model_cls.load(checkpoint_path, map_location=device).to(device)
        except (AttributeError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' is not compatible with the new model save/load format. "
                f"Re-save the model with {model_cls.__name__}.save before using this script."
            ) from exc

        __logger__.info("Loaded model from %s", checkpoint_path)
        model.eval()
        return model

    def __getitem__(self, model_cls: type[ModelT]) -> Callable[..., ModelT]:
        def load_with_model_cls(
            model_dir: Path | str | None,
            device: th.device,
            results_root: Path | str | None = None,
        ) -> ModelT:
            return self(model_cls, model_dir, device, results_root=results_root)

        return load_with_model_cls


load_policy_model = GenericModelLoader("model.pt")
load_lyapunov_model = GenericModelLoader("lyapunov_model.pt")


def default_model_path(results_root: Path | str | None = None) -> str:
    resolved_results_root = Path(results_root) if results_root is not None else _RESULTS_ROOT
    return str(discover_latest_model_dir(resolved_results_root, "model.pt") / "model.pt")


def default_lyapunov_model_path(results_root: Path | str | None = None) -> str:
    resolved_results_root = Path(results_root) if results_root is not None else _RESULTS_ROOT
    return str(discover_latest_model_dir(resolved_results_root, "lyapunov_model.pt") / "lyapunov_model.pt")


def default_dataset_path(results_root: Path | str, dataset_pattern: str = "*.hdf5") -> str:
    return str(discover_latest_dataset_path(results_root, dataset_pattern))