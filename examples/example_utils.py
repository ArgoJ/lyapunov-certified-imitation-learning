import torch as th
import torch.nn as nn
import logging
import re

from pathlib import Path
from typing import Callable, TypeVar
from numpy.typing import NDArray

from .constants import *


__logger__ = logging.getLogger("lcil.examples.example_utils")


ModelT = TypeVar("ModelT", bound=nn.Module)
ISO_PATTERN = re.compile(r"^\d{8}_\d{6}$")


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
    default_root: Path = RESULTS_DIR,
) -> Path:
    if root_path is not None:
        resolved_root = Path(root_path)
        return require_dir(resolved_root, name="Results root directory")
    return require_dir(default_root, name="Default results root directory")


def discover_model_dir(results_root: Path, checkpoint_name: str, n: int = -1, sorting_idx: int | slice = 0) -> Path:
    """Discover the directory containing the nth latest checkpoint with the given name under the results root.

    Parameters
    ----------
    results_root : Path
        Root directory under which to search for checkpoint-containing subdirectories.
    checkpoint_name : str
        Name of the checkpoint file to search for.
    n : int, optional
        Index of the checkpoint to return, by default -1 (latest checkpoint).
    sorting_idx : int | slice, optional
        Index or slice of the parent directory name to use for sorting, by default 0

    Returns
    -------
    Path
        Directory containing the nth latest checkpoint.

    Raises
    ------
    FileNotFoundError
        If no checkpoint with the given name is found.
    """
    resolved_results_root = resolve_root(results_root)

    candidates: list[tuple[Path, str]] = []
    for path in resolved_results_root.rglob(checkpoint_name):
        if isinstance(sorting_idx, slice):
            parents_to_check = path.parents[sorting_idx]
        else:
            try:
                parents_to_check = [path.parents[sorting_idx]]
            except IndexError:
                continue

        found_iso_name = None
        for parent in parents_to_check:
            if ISO_PATTERN.match(parent.name):
                found_iso_name = parent.name
                break

            if parent == resolved_results_root or resolved_results_root not in parent.parents:
                break

        if found_iso_name is None:
            if isinstance(sorting_idx, slice):
                 found_iso_name = path.parents[sorting_idx.start or 0].name
            else:
                 found_iso_name = path.parents[sorting_idx].name

        candidates.append((path, found_iso_name))
            
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint '{checkpoint_name}' found under '{resolved_results_root}'."
        )
        
    candidates = sorted(candidates, key=lambda x: x[1])
    return candidates[n][0].parent


def discover_latest_policy_dir(dir: Path | str | None = None) -> Path:
    """Discover the directory containing the nth latest policy checkpoint.

    Parameters
    ----------
    dir : Path | str | None, optional
        Root directory under which to search for policy checkpoints, by default None

    Returns
    -------
    Path
        Directory containing the nth latest policy checkpoint.
    """
    return discover_model_dir(dir, POLICY_MODEL_FILENAME, n=-1, sorting_idx=slice(0, 2))


def discover_latest_lyapunov_dir(dir: Path | str | None = None) -> Path:
    """Discover the directory containing the nth latest Lyapunov checkpoint.

    Parameters
    ----------
    dir : Path | str | None, optional
        Root directory under which to search for Lyapunov checkpoints, by default None

        Directory containing the policy checkpoints.

    Returns
    -------
    Path
        Directory containing the nth latest Lyapunov checkpoint.
    """
    if dir is not None:
        dir = require_dir(dir, name="Directory to search for Lyapunov checkpoint")
        if LYAPUNOV_DIRNAME in dir.parts:
            __logger__.warning(f"Provided directory '{dir}' already contains '{LYAPUNOV_DIRNAME}' in its path. Searching for Lyapunov checkpoint directly under the provided directory.")
            return discover_model_dir(dir, LYAPUNOV_MODEL_FILENAME, n=-1, sorting_idx=slice(0, 2))
    
    # latest policy path must be found
    policy_dir = discover_latest_policy_dir(dir)
    return discover_model_dir(policy_dir / LYAPUNOV_DIRNAME, LYAPUNOV_MODEL_FILENAME, n=-1, sorting_idx=slice(0, 2))


def discover_latest_policy_and_lyapunov_dirs(results_root: Path | str | None = None, max_search: int = 100) -> tuple[Path, Path]:
    """Discover the directories containing the latest policy and Lyapunov checkpoints.

    Parameters
    ----------
    dir : Path | str | None, optional
        Root directory under which to search for policy and Lyapunov checkpoints, by default None
    max_search : int, optional
        Maximum number of recent policy checkpoints to search through, by default 100

    Returns
    -------
    tuple[Path, Path]
        Directories containing the latest policy and Lyapunov checkpoints.

    Raises
    ------
    FileNotFoundError
        If no matching policy and Lyapunov checkpoints are found.
    """
    for n in range(1, max_search + 1):
        try:
            policy_dir = discover_model_dir(results_root, POLICY_MODEL_FILENAME, n=-n)
            lyapunov_dir = discover_model_dir(policy_dir / LYAPUNOV_DIRNAME, LYAPUNOV_MODEL_FILENAME, n=-1)
            return policy_dir, lyapunov_dir
        except FileNotFoundError:
            pass
    raise FileNotFoundError(
        f"No matching policy and Lyapunov checkpoints found under '{results_root}' after searching "
        f"through the {max_search} most recent policy checkpoints."
    )


def discover_latest_dataset_path(results_root: Path | str, dataset_pattern: str = "*.hdf5") -> Path:
    """Discover the latest dataset file matching the given pattern under the results root.

    Parameters
    ----------
    results_root : Path | str
        Root directory under which to search for dataset files.
    dataset_pattern : str, optional
        Pattern to match dataset files, by default "*.hdf5"

    Returns
    -------
    Path
        Path to the latest dataset file matching the pattern.

    Raises
    ------
    FileNotFoundError
        If no dataset file matching the pattern is found.
    """
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


def discover_latest_cert_lyapunov_path(lyapunov_root: Path | str | None = None) -> Path:
    resolved_lyapunov_root = require_dir(lyapunov_root, name="Directory to search for certification result")
    candidates = [
        result_path
        for result_path in resolved_lyapunov_root.rglob(CERTIFICATION_DETAILS_FILENAME)
        if result_path.is_file()
    ]
    candidates = sorted(candidates, key=lambda x: x.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No certification result '{CERTIFICATION_DETAILS_FILENAME}' found under '{resolved_lyapunov_root}'."
        )
    return candidates[-1].parent.parent


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
            if resolved_model_path.is_dir():
                return require_file(
                    resolved_model_path / self.checkpoint_name,
                    name=f"{self.checkpoint_name} checkpoint",
                )
            if resolved_model_path.is_file():
                return require_file(
                    resolved_model_path,
                    name=f"{self.checkpoint_name} checkpoint",
                )
            raise FileNotFoundError(
                f"Could not resolve '{resolved_model_path}' as either a directory containing "
                f"'{self.checkpoint_name}' or as a checkpoint file."
            )

        resolved_results_root = resolve_root(results_root)
        return discover_model_dir(resolved_results_root, self.checkpoint_name) / self.checkpoint_name

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


def default_model_path(results_root: Path | str | None = None) -> str:
    resolved_results_root = Path(results_root) if results_root is not None else RESULTS_DIR
    return str(discover_model_dir(resolved_results_root, POLICY_MODEL_FILENAME) / POLICY_MODEL_FILENAME)


def default_lyapunov_model_path(results_root: Path | str | None = None) -> str:
    resolved_results_root = Path(results_root) if results_root is not None else RESULTS_DIR
    return str(discover_model_dir(resolved_results_root, LYAPUNOV_MODEL_FILENAME) / LYAPUNOV_MODEL_FILENAME)


def default_dataset_path(results_root: Path | str, dataset_pattern: str = "*.hdf5") -> str:
    return str(discover_latest_dataset_path(results_root, dataset_pattern))


def default_cert_result_path(
    results_root: Path | str,
    result_name: str = CERTIFICATION_DETAILS_FILENAME,
) -> str:
    return str(discover_latest_cert_lyapunov_path(results_root, result_name))


def build_lyapunov_func(
    lyap_model,
    device: th.device,
) -> Callable[[NDArray], NDArray]:
    def lyapunov_func(states: NDArray) -> NDArray:
        x = th.as_tensor(states, dtype=th.float32, device=device)
        with th.no_grad():
            values = lyap_model(x)
        return values.detach().cpu().numpy().reshape(-1)

    return lyapunov_func


def find_all_lyapunov_dirs(path: Path) -> list[Path]:
    if not path.exists():
        __logger__.warning(f"Directory {path} not found.")
        return []

    lyap_dirs = []
    for model_file in path.rglob(LYAPUNOV_MODEL_FILENAME):
        lyap_dirs.append(model_file.parent)
        
    return lyap_dirs