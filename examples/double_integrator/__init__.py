from .acados_ocp import get_batch_ocp_solver, get_model, get_ocp, get_ocp_solver
from .double_integrator_dyn import DoubleIntegratorDynamics

from pathlib import Path
from typing import TypeVar

import torch as th
from torch import nn

from lcil.imitation_learning import MLPPolicy, TransformerPolicy
from lcil.lyapunov_learning import NeuralLyapunovCandidate

from ..example_utils import (
    default_dataset_path as _default_dataset_path,
    default_lyapunov_model_path as _default_lyapunov_model_path,
    default_model_path as _default_model_path,
    resolve_dataset_path as _resolve_dataset_path,
    discover_latest_lyapunov_dir as _discover_latest_lyapunov_dir,
    discover_latest_policy_dir as _discover_latest_policy_dir,
    discover_latest_policy_and_lyapunov_dirs as _discover_latest_policy_and_lyapunov_dirs,
    GenericModelLoader as _GenericModelLoader,
)

_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "double_integrator"
_DATA_ROOT = _RESULTS_ROOT / "data"

PolicyModelT = TypeVar("PolicyModelT", bound=nn.Module)


def discover_latest_policy_dir(results_root: Path | str | None = None):
    return _discover_latest_policy_dir(results_root or _RESULTS_ROOT)


def discover_latest_lyapunov_dir(policy_dir: Path | str | None = None):
    resolved_policy_dir = discover_latest_policy_dir() if policy_dir is None else policy_dir
    return _discover_latest_lyapunov_dir(resolved_policy_dir)


def discover_latest_policy_and_lyapunov_dirs(results_root: Path | str | None = None, max_search: int = 100):
    return _discover_latest_policy_and_lyapunov_dirs(results_root or _RESULTS_ROOT, max_search=max_search)


def _resolve_policy_model_cls(model_path: Path) -> type[nn.Module]:
    checkpoint = th.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format in '{model_path}'.")

    if "layer_sizes" in checkpoint and "activations" in checkpoint:
        return MLPPolicy
    if "input_dim" in checkpoint and "output_dim" in checkpoint and "max_seq_len" in checkpoint:
        return TransformerPolicy

    raise ValueError(
        f"Could not infer policy model type from checkpoint '{model_path}'. "
        "Expected MLP or Transformer architecture metadata."
    )


def load_policy_model(
    path: Path | str | None,
    device: th.device | str,
    model_cls: type[PolicyModelT] | None = None,
    model_name: str = "model.pt",
) -> PolicyModelT | nn.Module:
    model_loader = _GenericModelLoader(model_name)
    checkpoint_path = model_loader._resolve_checkpoint_path(path, results_root=_RESULTS_ROOT)
    resolved_model_cls = _resolve_policy_model_cls(checkpoint_path) if model_cls is None else model_cls
    return model_loader[resolved_model_cls](checkpoint_path, device, _RESULTS_ROOT)


def load_lyapunov_model(path, device, model_name: str = "lyapunov_model.pt") -> NeuralLyapunovCandidate:
    model_loader = _GenericModelLoader(model_name)
    return model_loader[NeuralLyapunovCandidate](path, device, _RESULTS_ROOT)


def default_model_path(results_root: Path | str | None = None):
    return _default_model_path(results_root or _RESULTS_ROOT)


def default_lyapunov_model_path(results_root: Path | str | None = None):
    return _default_lyapunov_model_path(results_root or _RESULTS_ROOT)


def default_dataset_path(data_root: Path | str | None = None):
    return _default_dataset_path(data_root or _DATA_ROOT)


def resolve_dataset_path(dataset_path: Path | str | None, data_root: Path | str | None = None):
    return _resolve_dataset_path(dataset_path, data_root or _DATA_ROOT)