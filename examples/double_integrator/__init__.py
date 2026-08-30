import torch as th

from pathlib import Path
from typing import TypeVar
from torch import nn

from lcil.imitation_learning import BoundedPolicy, TransformerPolicy
from lcil.lyapunov_learning import NeuralLyapunovCandidate
from lcil.utils import load_mpc_config_for_run

from .acados_ocp import get_batch_ocp_solver, get_model, get_ocp, get_ocp_solver
from .double_integrator_dyn import DoubleIntegratorDynamics#
from .basis import *
from ..example_utils import (
    build_lyapunov_func,
    find_all_lyapunov_dirs,
    find_all_policy_dirs,
    require_file,
    require_dir,
    get_initial_states,
    sample_uncertified_regions,
    get_condition_violations,
    discover_source_rollout_path,
)
from ..metrics_collector import (
    LevelSetMetricsWriter,
    LevelSetMetricsCollector,
    save_level_set_metrics,
    add_entry,
)
from ..constants import *

from ..example_utils import (
    default_dataset_path as _default_dataset_path,
    resolve_dataset_path as _resolve_dataset_path,
    resolve_expert_dataset_path as _resolve_expert_dataset_path,
    discover_latest_lyapunov_dir as _discover_latest_lyapunov_dir,
    discover_latest_policy_dir as _discover_latest_policy_dir,
    discover_latest_policy_and_lyapunov_dirs as _discover_latest_policy_and_lyapunov_dirs,
    discover_latest_cert_lyapunov_path as _discover_latest_cert_lyapunov_path,
    GenericModelLoader as _GenericModelLoader,
)

DOUBLE_INTEGRATOR_RESULTS_DIR = RESULTS_DIR / "double_integrator"
DATA_DIR = DOUBLE_INTEGRATOR_RESULTS_DIR / "data"

PolicyModelT = TypeVar("PolicyModelT", bound=nn.Module)


def discover_latest_policy_dir(
    results_root: Path | str | None = None,
    enforce_iso_setup_layout: bool = True,
):
    return _discover_latest_policy_dir(
        results_root or DOUBLE_INTEGRATOR_RESULTS_DIR,
        enforce_iso_setup_layout=enforce_iso_setup_layout,
    )


def discover_latest_lyapunov_dir(
    policy_dir: Path | str | None = None,
    enforce_iso_setup_layout: bool = True,
):
    resolved_policy_dir = policy_dir or discover_latest_policy_dir()
    return _discover_latest_lyapunov_dir(
        resolved_policy_dir,
        enforce_iso_setup_layout=enforce_iso_setup_layout,
    )


def discover_latest_policy_and_lyapunov_dirs(
    results_root: Path | str | None = None,
    max_search: int = 100,
    enforce_iso_setup_layout: bool = True,
):
    return _discover_latest_policy_and_lyapunov_dirs(
        results_root or DOUBLE_INTEGRATOR_RESULTS_DIR,
        max_search=max_search,
        enforce_iso_setup_layout=enforce_iso_setup_layout,
    )


def discover_latest_cert_lyapunov_path(lyapunov_root: Path | None = None) -> Path:
    resolved_lyapunov_root = discover_latest_lyapunov_dir() if lyapunov_root is None else lyapunov_root
    return _discover_latest_cert_lyapunov_path(resolved_lyapunov_root.parent)


def _resolve_policy_model_cls(model_path: Path) -> type[nn.Module]:
    checkpoint = th.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format in '{model_path}'.")

    policy_type = checkpoint.get("policy_type", None)
    match policy_type:
        case "BoundedPolicy":
            return BoundedPolicy
        case "TransformerPolicy":
            return TransformerPolicy

    raise ValueError(
        f"Could not infer policy model type from checkpoint '{model_path}'. "
        "Expected a checkpoint saved with explicit 'policy_type' metadata."
    )


def load_policy_model(
    path: Path | str | None,
    device: th.device | str,
    model_cls: type[PolicyModelT] | None = None,
    model_name: str = POLICY_MODEL_FILENAME,
) -> PolicyModelT | nn.Module:
    model_loader = _GenericModelLoader(model_name)
    checkpoint_path = model_loader._resolve_checkpoint_path(path, results_root=DOUBLE_INTEGRATOR_RESULTS_DIR)
    resolved_model_cls = _resolve_policy_model_cls(checkpoint_path) if model_cls is None else model_cls
    return model_loader[resolved_model_cls](checkpoint_path, device, DOUBLE_INTEGRATOR_RESULTS_DIR)


def load_mpc_config(
    path: Path | str | None,
    model_name: str = POLICY_MODEL_FILENAME,
):
    model_loader = _GenericModelLoader(model_name)
    checkpoint_path = model_loader._resolve_checkpoint_path(path, results_root=DOUBLE_INTEGRATOR_RESULTS_DIR)
    return load_mpc_config_for_run(checkpoint_path.parent)


def load_lyapunov_model(path, device, model_name: str = LYAPUNOV_MODEL_FILENAME) -> NeuralLyapunovCandidate:
    model_loader = _GenericModelLoader(model_name)
    return model_loader[NeuralLyapunovCandidate](path, device, DOUBLE_INTEGRATOR_RESULTS_DIR)


def default_dataset_path(data_root: Path | str | None = None):
    return _default_dataset_path(data_root or DATA_DIR)


def resolve_dataset_path(dataset_path: Path | str | None, data_root: Path | str | None = None):
    return _resolve_dataset_path(dataset_path, data_root or DATA_DIR)


def resolve_expert_dataset_path(target_dir: Path | str) -> Path:
    return _resolve_expert_dataset_path(target_dir)