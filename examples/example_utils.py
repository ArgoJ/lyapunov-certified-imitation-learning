import torch as th
import torch.nn as nn
import logging

from pathlib import Path
from typing import Callable, TypeVar


__logger__ = logging.getLogger("lcil.examples.example_utils")


ModelT = TypeVar("ModelT", bound=nn.Module)


_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"



def require_file(path: Path | str, *, name: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} not found: '{resolved}'.")
    return resolved


def require_dir(path: Path | str, *, name: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} not found: '{resolved}'.")
    return resolved


def resolve_root(
    root_path: Path | str | None,
    default_root: Path = _RESULTS_ROOT,
) -> Path:
    if root_path is not None:
        resolved_root = Path(root_path)
        return require_dir(resolved_root, name="Results root directory")
    return require_dir(default_root, name="Default results root directory")


def discover_latest_model_dir(results_root: Path, checkpoint_name: str) -> Path:
    resolved_results_root = resolve_root(results_root)
    candidates = sorted(checkpoint.parent for checkpoint in resolved_results_root.rglob(checkpoint_name))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint '{checkpoint_name}' found under '{resolved_results_root}'.")
    return candidates[-1]


def discover_latest_policy_dir(results_root: Path | str | None = None) -> Path:
    resolved_results_root = resolve_root(results_root)
    return discover_latest_model_dir(resolved_results_root, "model.pt")


def discover_latest_lyapunov_dir(policy_dir: Path | str) -> Path:
    resolved_policy_dir = require_dir(policy_dir, name="Policy run directory")
    lyapunov_root = require_dir(resolved_policy_dir / "lyapunov", name="Lyapunov root directory")
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


def discover_latest_cert_result_path(
    results_root: Path | str,
    result_name: str = "certification_details.npz",
) -> Path:
    resolved_results_root = Path(results_root)
    candidates = sorted(
        result_path
        for result_path in resolved_results_root.rglob(result_name)
        if result_path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No certification result '{result_name}' found under '{resolved_results_root}'."
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

    def _resolve_checkpoint_path(
        self,
        model_dir: Path | str | None,
        results_root: Path | str | None = None,
    ) -> Path:
        if model_dir is not None:
            resolved_model_path = Path(model_dir).resolve()
            if resolved_model_path.suffix:
                return require_file(
                    resolved_model_path,
                    name=f"{self.checkpoint_name} checkpoint",
                )
            return require_file(
                resolved_model_path / self.checkpoint_name,
                name=f"{self.checkpoint_name} checkpoint",
            )

        resolved_results_root = resolve_root(results_root)
        return discover_latest_model_dir(resolved_results_root, self.checkpoint_name) / self.checkpoint_name

    def __call__(
        self,
        model_cls: type[ModelT],
        model_dir: Path | str | None,
        device: th.device,
        results_root: Path | str | None = None,
    ) -> ModelT:
        checkpoint_path = self._resolve_checkpoint_path(model_dir, results_root=results_root)
        try:
            model = model_cls.load(checkpoint_path, map_location=device).to(device)
        except (AttributeError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' is not compatible with the new model save/load format. "
                f"Re-save the model with {model_cls.__name__}.save before using this script."
            ) from exc

        __logger__.info("Loaded %s from %s", model_cls.__name__, checkpoint_path)
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


def default_cert_result_path(
    results_root: Path | str,
    result_name: str = "certification_details.npz",
) -> str:
    return str(discover_latest_cert_result_path(results_root, result_name))