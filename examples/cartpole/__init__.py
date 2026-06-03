from .acados_ocp import get_ocp_solver, get_ocp, get_batch_ocp_solver, get_model
from .basis import *
from .model import CartpoleAngleWrapper
from .cartpole_dyn import CartpoleDynamics
from .sys_cfg import PendulumOnCartConfig

from pathlib import Path

from lcil.lyapunov_learning import NeuralLyapunovCandidate

from ..example_utils import (
    default_dataset_path as _default_dataset_path,
    default_lyapunov_model_path as _default_lyapunov_model_path,
    default_model_path as _default_model_path,
    resolve_dataset_path as _resolve_dataset_path,
    discover_latest_lyapunov_dir as _discover_latest_lyapunov_dir,
    discover_latest_policy_dir as _discover_latest_policy_dir,
    load_lyapunov_model as _load_lyapunov_model,
    load_policy_model as _load_policy_model,
)

_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "cartpole"
_DATA_ROOT = _RESULTS_ROOT / "data"
CARTPOLE_RESULTS_DIR = _RESULTS_ROOT
DATA_DIR = _DATA_ROOT


def discover_latest_policy_dir(results_root: Path | str | None = None):
    return _discover_latest_policy_dir(results_root or _RESULTS_ROOT)


def discover_latest_lyapunov_dir(policy_dir: Path | str | None = None):
    resolved_policy_dir = discover_latest_policy_dir() if policy_dir is None else policy_dir
    return _discover_latest_lyapunov_dir(resolved_policy_dir)


def load_policy_model(path, device):
    return _load_policy_model[CartpoleAngleWrapper](path, device, _RESULTS_ROOT)


def load_lyapunov_model(path, device):
    return _load_lyapunov_model[NeuralLyapunovCandidate](path, device, _RESULTS_ROOT)


def default_model_path(results_root: Path | str | None = None):
    return _default_model_path(results_root or _RESULTS_ROOT)


def default_lyapunov_model_path(results_root: Path | str | None = None):
    return _default_lyapunov_model_path(results_root or _RESULTS_ROOT)


def default_dataset_path(data_root: Path | str | None = None):
    return _default_dataset_path(data_root or _DATA_ROOT)


def resolve_dataset_path(dataset_path: Path | str | None, data_root: Path | str | None = None):
    return _resolve_dataset_path(dataset_path, data_root or _DATA_ROOT)