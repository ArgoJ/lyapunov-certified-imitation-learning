import torch as th

from .acados_ocp import get_ocp_solver, get_ocp, get_batch_ocp_solver, get_model
from .basis import *
from .model import CartpoleAngleWrapper
from .cartpole_dyn import CartpoleDynamics
from .sys_cfg import PendulumOnCartConfig

from pathlib import Path

from lcil.lyapunov_learning import NeuralLyapunovCandidate

from ..constants import *
from ..example_utils import (
    default_dataset_path as _default_dataset_path,
    resolve_dataset_path as _resolve_dataset_path,
    discover_latest_lyapunov_dir as _discover_latest_lyapunov_dir,
    discover_latest_policy_dir as _discover_latest_policy_dir,
    GenericModelLoader as _GenericModelLoader,
)

CARTPOLE_RESULTS_DIR = RESULTS_DIR / "cartpole"
DATA_DIR = CARTPOLE_RESULTS_DIR / "data"


def discover_latest_policy_dir(results_root: Path | str | None = None):
    return _discover_latest_policy_dir(results_root or CARTPOLE_RESULTS_DIR)


def discover_latest_lyapunov_dir(policy_dir: Path | str | None = None):
    resolved_policy_dir = discover_latest_policy_dir() if policy_dir is None else policy_dir
    return _discover_latest_lyapunov_dir(resolved_policy_dir)


def load_policy_model(
    path: Path | str | None,
    device: th.device | str,
    model_name: str = POLICY_MODEL_FILENAME,
) -> CartpoleAngleWrapper:
    model_loader = _GenericModelLoader(model_name)
    return model_loader[CartpoleAngleWrapper](path, device, CARTPOLE_RESULTS_DIR)


def load_lyapunov_model(
    path: Path | str | None,
    device: th.device | str,
    model_name: str = LYAPUNOV_MODEL_FILENAME,
) -> NeuralLyapunovCandidate:
    model_loader = _GenericModelLoader(model_name)
    return model_loader[NeuralLyapunovCandidate](path, device, CARTPOLE_RESULTS_DIR)


def default_dataset_path(data_root: Path | str | None = None):
    return _default_dataset_path(data_root or DATA_DIR)


def resolve_dataset_path(dataset_path: Path | str | None, data_root: Path | str | None = None):
    return _resolve_dataset_path(dataset_path, data_root or DATA_DIR)