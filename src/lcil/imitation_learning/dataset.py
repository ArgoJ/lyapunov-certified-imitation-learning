from __future__ import annotations
import inspect
import os
import logging
import numpy as np
import torch as th

from collections.abc import Sequence
from pathlib import Path
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from typing import Literal, TypeVar
from rich.progress import track
from mpc_datagen import MPCDataset, MPCMeta, MPCTrajectory

from .config import ImitationTrainingConfig
from ..utils.base_models import build_generator

__logger__ = logging.getLogger(__name__)

TConfigValue = TypeVar("TConfigValue")


def _to_tensor(array: NDArray | th.Tensor, name: str, dims: int) -> th.Tensor:
    """Convert input arrays/tensors to contiguous CPU tensors with the specified number of dimensions."""
    if isinstance(array, th.Tensor):
        tensor = array.detach().cpu()
    else:
        tensor = th.as_tensor(np.asarray(array))

    if tensor.ndim != dims:
        raise ValueError(f"{name} must be {dims}D, got shape {tuple(tensor.shape)}.")

    return tensor.contiguous()

def _resolve_dtype(dtype_value: th.dtype | str) -> th.dtype:
    """Resolve serialized dtype payloads to ``torch.dtype``."""
    if isinstance(dtype_value, th.dtype):
        return dtype_value

    if isinstance(dtype_value, str):
        normalized = dtype_value.removeprefix("torch.")
        if hasattr(th, normalized):
            resolved = getattr(th, normalized)
            if isinstance(resolved, th.dtype):
                return resolved

    raise ValueError(f"Invalid dtype in dataset file: {dtype_value}")

def _resolve_mpc_dataset(mpc_dataset: MPCDataset | os.PathLike) -> MPCDataset:
    """Resolve `MPCDataset` input from object or filesystem path."""
    if isinstance(mpc_dataset, MPCDataset):
        return mpc_dataset

    if isinstance(mpc_dataset, (str, os.PathLike)):
        return MPCDataset.load(Path(mpc_dataset))

    raise TypeError(
        "mpc_dataset must be an MPCDataset or path-like object "
        f"(str, pathlib.Path, os.PathLike), got {type(mpc_dataset).__name__}."
    )


def _materialize_subset_dataset(
    dataset_subset: Dataset[tuple[th.Tensor, th.Tensor]],
    dataset_cls: type[Dataset[tuple[th.Tensor, th.Tensor]]] | None = None,
    dtype: th.dtype | None = None,
) -> Dataset[tuple[th.Tensor, th.Tensor]]:
    """Materialize a subset into a concrete in-memory dataset instance."""
    if not (hasattr(dataset_subset, "dataset") and hasattr(dataset_subset, "indices")):
        raise TypeError("dataset_subset must expose 'dataset' and 'indices' attributes.")

    parent_dataset = getattr(dataset_subset, "dataset")
    resolved_dataset_cls = type(parent_dataset) if dataset_cls is None else dataset_cls
    subset_indices = th.as_tensor(getattr(dataset_subset, "indices"), dtype=th.int64)

    parent_states = getattr(parent_dataset, "_states", None)
    parent_actions = getattr(parent_dataset, "_actions", None)
    parent_refs = getattr(parent_dataset, "_refs", None)
    if parent_states is None or parent_actions is None:
        raise TypeError("Subset parent dataset must expose '_states' and '_actions' tensors.")

    init_kwargs: dict[str, object] = {
        "states": parent_states.index_select(0, subset_indices).detach().cpu(),
        "actions": parent_actions.index_select(0, subset_indices).detach().cpu(),
    }
    if parent_refs is not None:
        init_kwargs["refs"] = parent_refs.index_select(0, subset_indices).detach().cpu()

    resolved_dtype = dtype
    if resolved_dtype is None:
        parent_dtype = getattr(parent_dataset, "dtype", None)
        resolved_dtype = parent_dtype if isinstance(parent_dtype, th.dtype) else init_kwargs["states"].dtype
    init_kwargs["dtype"] = resolved_dtype

    constructor_parameters = inspect.signature(resolved_dataset_cls.__init__).parameters
    optional_values = {
        "target_mode": getattr(parent_dataset, "target_mode", None),
        "stride": getattr(parent_dataset, "stride", None),
    }

    parent_trajectory_ids = getattr(parent_dataset, "_trajectory_ids", None)
    if parent_trajectory_ids is not None:
        optional_values["trajectory_ids"] = (
            parent_trajectory_ids.index_select(0, subset_indices).detach().cpu()
        )

    parent_start_indices = getattr(parent_dataset, "_start_indices", None)
    if parent_start_indices is not None:
        optional_values["start_indices"] = (
            parent_start_indices.index_select(0, subset_indices).detach().cpu()
        )

    for name, value in optional_values.items():
        if name in constructor_parameters:
            init_kwargs[name] = value

    return resolved_dataset_cls(**init_kwargs)



class StateActionDataset(Dataset[tuple[th.Tensor, th.Tensor]]):
    """In-memory imitation-learning dataset backed by ``MPCDataset``."""

    def __init__(
        self,
        states: NDArray | th.Tensor,
        actions: NDArray | th.Tensor,
        refs: NDArray | th.Tensor | None = None,
        dtype: th.dtype = th.float32,
        near_duplicate_radius: float | None = None,
    ):
        """Initialize an in-memory state-action dataset.

        Parameters
        ----------
        states : numpy.ndarray or torch.Tensor
            State samples with shape ``(num_samples, nx)``.
        actions : numpy.ndarray or torch.Tensor
            Action samples with shape ``(num_samples, nu)``.
        refs : numpy.ndarray or torch.Tensor, optional
            Reference samples with shape ``(num_samples, nref)``.
        dtype : torch.dtype, optional
            Output tensor dtype for states and actions.
        near_duplicate_radius : float, optional
            Optional near-duplicate filter radius in L2 distance over the
            concatenated state-action vector.
            Uses a vectorized voxel deduplication approximation to remove dense,
            very similar samples efficiently.
        """
        if near_duplicate_radius is not None and near_duplicate_radius <= 0:
            raise ValueError("near_duplicate_radius must be positive when provided.")

        states_tensor = _to_tensor(states, name="states", dims=2)
        actions_tensor = _to_tensor(actions, name="actions", dims=2)
        refs_tensor = _to_tensor(refs, name="refs", dims=2) if refs is not None else None
        if states_tensor.shape[0] != actions_tensor.shape[0]:
            raise ValueError(
                "states/actions sample count mismatch: "
                f"got {states_tensor.shape[0]} states and {actions_tensor.shape[0]} actions."
            )
        if refs_tensor is not None and states_tensor.shape[0] != refs_tensor.shape[0]:
            raise ValueError(
                "states/refs sample count mismatch: "
                f"got {states_tensor.shape[0]} states and {refs_tensor.shape[0]} refs."
            )

        if near_duplicate_radius is not None and states_tensor.shape[0] > 0:
            keep_idx = self._near_duplicate_keep_indices(
                states=states_tensor,
                actions=actions_tensor,
                radius=near_duplicate_radius,
            )
            states_tensor = states_tensor.index_select(0, keep_idx)
            actions_tensor = actions_tensor.index_select(0, keep_idx)
            if refs_tensor is not None:
                refs_tensor = refs_tensor.index_select(0, keep_idx)

        self.dtype = dtype
        self.near_duplicate_radius = near_duplicate_radius

        self._states = states_tensor.to(dtype=self.dtype)
        self._actions = actions_tensor.to(dtype=self.dtype)
        self._refs = refs_tensor.to(dtype=self.dtype) if refs_tensor is not None else None
        self._total_samples = int(self._states.shape[0])

    def __len__(self) -> int:
        """Total number of training samples across all trajectories."""
        return self._total_samples

    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor] | tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return one imitation pair ``(state, action)`` at global index ``idx``."""
        if self._total_samples == 0:
            raise IndexError("Cannot index an empty imitation-learning dataset.")

        if idx < 0:
            idx += self._total_samples
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset size {self._total_samples}.")

        if self._states is None or self._actions is None:
            raise RuntimeError("In-memory tensors are not available.")

        if self._refs is not None:
            return self._states[idx], self._actions[idx], self._refs[idx]
        return self._states[idx], self._actions[idx]

    def save(self, path: os.PathLike) -> None:
        """Save the preprocessed (optionally filtered) state-action pairs to a ``.pt`` file.

        Parameters
        ----------
        path : str or os.PathLike
            Output path for ``torch.save``.
        """
        target_path = Path(path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "states": self._states.detach().cpu(),
            "actions": self._actions.detach().cpu(),
            "dtype": self.dtype,
            "near_duplicate_radius": self.near_duplicate_radius,
            "num_samples": self._total_samples,
            "refs": self._refs.detach().cpu() if self._refs is not None else None,
        }
        th.save(payload, target_path)

    @classmethod
    def load(cls, path: os.PathLike) -> StateActionDataset:
        """Load a preprocessed state-action dataset from a ``.pt`` file.

        Parameters
        ----------
        path : str or os.PathLike
            Path to the saved dataset file (``.pt``).

        Returns
        -------
        StateActionDataset
            The loaded dataset object with in-memory tensors.
        """
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Dataset file not found at {source_path}.")

        payload = th.load(source_path, map_location="cpu")
        required_keys = {"states", "actions", "dtype", "num_samples", "refs"}
        if not required_keys.issubset(payload.keys()):
            missing = required_keys - payload.keys()
            raise ValueError(f"Missing keys in dataset file: {missing}")

        states = payload["states"]
        actions = payload["actions"]
        references = payload["refs"]
        dtype_payload = payload["dtype"]
        dtype = _resolve_dtype(dtype_payload)
        dataset = cls(
            states=states,
            actions=actions,
            refs=references,
            dtype=dtype,
        )
        
        return dataset

    @classmethod
    def from_mpc_dataset(
        cls,
        mpc_dataset: MPCDataset | os.PathLike,
        dtype: th.dtype = th.float32,
        use_references: bool = False,
        near_duplicate_radius: float | None = None,
    ) -> StateActionDataset:
        """Create a ``StateActionDataset`` by extracting samples from an MPC dataset."""
        resolved_mpc_dataset = _resolve_mpc_dataset(mpc_dataset)
        states, actions = cls._load_samples_from_mpc_dataset(resolved_mpc_dataset)

        return cls(
            states=states,
            actions=actions,
            dtype=dtype,
            refs=None if not use_references else None, # TODO: compute references from cost if requested
            near_duplicate_radius=near_duplicate_radius,
        )

    @classmethod
    def from_subset(
        cls,
        dataset_subset: Dataset[tuple[th.Tensor, th.Tensor]],
        dtype: th.dtype | None = None,
    ) -> StateActionDataset:
        """Create a ``StateActionDataset`` from a subset wrapper dataset."""
        materialized = _materialize_subset_dataset(dataset_subset, dataset_cls=cls, dtype=dtype)
        if not isinstance(materialized, cls):
            raise TypeError(f"Expected materialized subset to be of type '{cls.__name__}'.")
        return materialized

    @staticmethod
    def _load_samples_from_mpc_dataset(mpc_dataset: MPCDataset) -> tuple[NDArray, NDArray]:
        """Load all valid samples from ``MPCDataset`` into contiguous NumPy arrays."""
        if mpc_dataset.memory_buffer:
            mpc_dataset.save(mode="a")

        if mpc_dataset._h5_file is None:
            raise ValueError("MPCDataset must have an HDF5 file loaded to build the imitation-learning dataset.")

        states_chunks: list[NDArray] = []
        actions_chunks: list[NDArray] = []
        references_chunks: list[NDArray] = []
        expected_nx: int | None = None
        expected_nu: int | None = None

        indices = list(mpc_dataset._indices)
        for key in track(indices, description="Extracting samples"):
            grp = mpc_dataset._h5_file[key]
            meta = MPCMeta.from_hdf5(grp)
            steps = int(meta.steps_simulated)
            if steps <= 0 or meta.feasible == False:
                continue

            traj = MPCTrajectory.from_hdf5(grp, fields=["states", "inputs"])
            states = traj.states[:steps, :]
            actions = traj.inputs[:steps, :]

            nx = int(states.shape[1])
            nu = int(actions.shape[1])
            if expected_nx is None:
                expected_nx = nx
                expected_nu = nu
            elif nx != expected_nx or nu != expected_nu:
                raise ValueError(
                    "Inconsistent state/action dimensions across trajectories: "
                    f"expected (nx={expected_nx}, nu={expected_nu}), "
                    f"got (nx={nx}, nu={nu}) for trajectory ID {meta.id}."
                )

            states_chunks.append(states)
            actions_chunks.append(actions)

        if not states_chunks or not actions_chunks:
            return np.empty((0, 0)), np.empty((0, 0))

        states = np.ascontiguousarray(np.concatenate(states_chunks, axis=0))
        actions = np.ascontiguousarray(np.concatenate(actions_chunks, axis=0))

        if expected_nx is not None and states.shape[1] != expected_nx:
            raise ValueError(
                f"Concatenated states dimension mismatch: expected nx={expected_nx}, got nx={states.shape[1]}."
            )
        if expected_nu is not None and actions.shape[1] != expected_nu:
            raise ValueError(
                f"Concatenated actions dimension mismatch: expected nu={expected_nu}, got nu={actions.shape[1]}."
            )

        return states, actions

    @staticmethod
    def _near_duplicate_keep_indices(
        states: NDArray | th.Tensor,
        actions: NDArray | th.Tensor,
        radius: float,
    ) -> th.Tensor:
        """Keep sample indices after adaptive near-duplicate voxel deduplication.

        Parameters
        ----------
        states : numpy.ndarray or torch.Tensor
            State samples with shape `(n_samples, nx)`.
        actions : numpy.ndarray or torch.Tensor
            Action samples with shape `(n_samples, nu)`.
        radius : float
            Base L2 radius controlling voxel size in normalized feature space.
            The base cell size is `radius / sqrt(d)` where `d = nx + nu`.

        Returns
        -------
        keep_idx : torch.Tensor
            Sorted indices of kept samples with shape `(n_kept,)`.

        Notes
        -----
        Concatenated `[state, action]` features are robustly normalized with
        median/IQR for quantization. Radial scaling is computed from either
        provided residuals or median-centered min/max bounds so that voxels are
        effectively larger near reference and smaller toward bounds.
        """
        states_tensor = th.as_tensor(states, dtype=th.float64)
        actions_tensor = th.as_tensor(actions, dtype=th.float64)

        if states_tensor.shape[0] == 0:
            return th.empty((0,), dtype=th.int64)

        features = th.cat((states_tensor, actions_tensor), dim=1)
        feature_median = th.median(features, dim=0).values
        feature_iqr = th.quantile(features, 0.75, dim=0) - th.quantile(features, 0.25, dim=0)
        feature_iqr = th.clamp(feature_iqr, min=1e-8)
        normalized_features = (features - feature_median) / feature_iqr

        feature_dim = int(features.shape[1])
        cell_size = radius / np.sqrt(float(feature_dim))

        # median-centered radial source with adaptive scaling based on either residuals or feature bounds
        feature_min = th.min(features, dim=0).values
        feature_max = th.max(features, dim=0).values
        half_range = 0.5 * (feature_max - feature_min)
        half_range = th.clamp(half_range, min=1e-8)
        radial_source = (features - feature_median) / half_range

        radial_distance = th.linalg.norm(radial_source, dim=1)
        radial_scale = float(th.quantile(radial_distance, 0.90).item())
        radial_scale = max(radial_scale, 1e-8)
        t = radial_distance / radial_scale
        adaptive_scale = 0.5 + 1.5 * (t / (1.0 + t))
        transformed_features = normalized_features * adaptive_scale[:, None]

        quantized = th.floor(transformed_features / cell_size).to(dtype=th.int64)
        _, inverse = th.unique(quantized, dim=0, return_inverse=True)
        positions = th.arange(inverse.shape[0], dtype=th.int64)

        order = positions[th.argsort(inverse, stable=True)]
        sorted_inverse = inverse[order]
        first_mask = th.ones_like(sorted_inverse, dtype=th.bool)
        first_mask[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
        keep_idx = order[first_mask]
        return th.sort(keep_idx).values


class SequenceStateActionDataset(Dataset[tuple[th.Tensor, th.Tensor]]):
    """Trajectory-aware imitation dataset backed by fixed-length state windows."""

    def __init__(
        self,
        states: NDArray | th.Tensor,
        actions: NDArray | th.Tensor,
        refs: NDArray | th.Tensor | None = None,
        dtype: th.dtype = th.float32,
        target_mode: Literal["last", "per_step", "all"] = "last",
        stride: int = 1,
        trajectory_ids: NDArray | th.Tensor | None = None,
        start_indices: NDArray | th.Tensor | None = None,
    ) -> None:
        """Initialize a fixed-length sequence imitation dataset.

        Parameters
        ----------
        states : numpy.ndarray or torch.Tensor
            State windows with shape ``(num_windows, sequence_length, nx)``.
        actions : numpy.ndarray or torch.Tensor
            Action targets with shape ``(num_windows, nu)`` for ``target_mode='last'``
            or ``(num_windows, sequence_length, nu)`` for ``target_mode='per_step'``.
        refs : numpy.ndarray or torch.Tensor, optional
            Optional reference windows with shape ``(num_windows, sequence_length, nref)``.
        dtype : torch.dtype, optional
            Output tensor dtype for states, actions, and references.
        target_mode : {"last", "per_step"}, optional
            Whether each sequence predicts only the last action or one action per token.
        stride : int, optional
            Sliding-window stride used when constructing the dataset.
        trajectory_ids : numpy.ndarray or torch.Tensor, optional
            Integer trajectory identifier per window with shape ``(num_windows,)``.
        start_indices : numpy.ndarray or torch.Tensor, optional
            Start index of each window inside its source trajectory with shape ``(num_windows,)``.
        """
        if stride <= 0:
            raise ValueError("stride must be positive.")

        states_tensor = _to_tensor(states, name="states", dims=3)
        actions_tensor = _to_tensor(actions, name="actions", dims=(2 if target_mode == "last" else 3))
        refs_tensor = _to_tensor(refs, name="refs", dims=3) if refs is not None else None

        if states_tensor.shape[0] != actions_tensor.shape[0]:
            raise ValueError(
                "states/actions sample count mismatch: "
                f"got {states_tensor.shape[0]} state windows and {actions_tensor.shape[0]} action targets."
            )
        if refs_tensor is not None and states_tensor.shape[:2] != refs_tensor.shape[:2]:
            raise ValueError(
                "states/refs window shape mismatch: "
                f"got states shape {tuple(states_tensor.shape)} and refs shape {tuple(refs_tensor.shape)}."
            )
        if target_mode == "per_step" and states_tensor.shape[:2] != actions_tensor.shape[:2]:
            raise ValueError(
                "states/actions window shape mismatch for per_step targets: "
                f"got states shape {tuple(states_tensor.shape)} and actions shape {tuple(actions_tensor.shape)}."
            )

        total_samples = int(states_tensor.shape[0])

        if trajectory_ids is None:
            trajectory_ids_tensor = th.full((total_samples,), -1, dtype=th.int64)
        else:
            trajectory_ids_tensor = th.as_tensor(trajectory_ids, dtype=th.int64).flatten().contiguous()
        if trajectory_ids_tensor.numel() != total_samples:
            raise ValueError(
                f"trajectory_ids must have {total_samples} elements, got {trajectory_ids_tensor.numel()}."
            )

        if start_indices is None:
            start_indices_tensor = th.full((total_samples,), -1, dtype=th.int64)
        else:
            start_indices_tensor = th.as_tensor(start_indices, dtype=th.int64).flatten().contiguous()
        if start_indices_tensor.numel() != total_samples:
            raise ValueError(
                f"start_indices must have {total_samples} elements, got {start_indices_tensor.numel()}."
            )

        self.dtype = dtype
        self.target_mode = target_mode
        self.stride = int(stride)
        self.sequence_length = int(states_tensor.shape[1]) if states_tensor.ndim == 3 else 0

        self._states = states_tensor.to(dtype=self.dtype)
        self._actions = actions_tensor.to(dtype=self.dtype)
        self._refs = refs_tensor.to(dtype=self.dtype) if refs_tensor is not None else None
        self._trajectory_ids = trajectory_ids_tensor
        self._start_indices = start_indices_tensor
        self._total_samples = total_samples

    def __len__(self) -> int:
        """Return the number of sequence windows in the dataset."""
        return self._total_samples

    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor] | tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return one sequence window and its action target."""
        if self._total_samples == 0:
            raise IndexError("Cannot index an empty sequence dataset.")

        if idx < 0:
            idx += self._total_samples
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset size {self._total_samples}.")

        if self._refs is not None:
            return self._states[idx], self._actions[idx], self._refs[idx]
        return self._states[idx], self._actions[idx]

    @property
    def trajectory_ids(self) -> th.Tensor:
        """Return the source trajectory id for each sequence window."""
        return self._trajectory_ids

    @property
    def start_indices(self) -> th.Tensor:
        """Return the source start index for each sequence window."""
        return self._start_indices

    def save(self, path: os.PathLike) -> None:
        """Save sequence windows and metadata to a ``.pt`` file."""
        target_path = Path(path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "states": self._states.detach().cpu(),
            "actions": self._actions.detach().cpu(),
            "refs": self._refs.detach().cpu() if self._refs is not None else None,
            "dtype": self.dtype,
            "num_samples": self._total_samples,
            "target_mode": self.target_mode,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "trajectory_ids": self._trajectory_ids.detach().cpu(),
            "start_indices": self._start_indices.detach().cpu(),
        }
        th.save(payload, target_path)

    @classmethod
    def load(cls, path: os.PathLike) -> "SequenceStateActionDataset":
        """Load a serialized sequence dataset from a ``.pt`` file."""
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Dataset file not found at {source_path}.")

        payload = th.load(source_path, map_location="cpu")
        required_keys = {
            "states",
            "actions",
            "refs",
            "dtype",
            "num_samples",
            "target_mode",
            "sequence_length",
            "stride",
            "trajectory_ids",
            "start_indices",
        }
        if not required_keys.issubset(payload.keys()):
            missing = required_keys - payload.keys()
            raise ValueError(f"Missing keys in dataset file: {missing}")

        dtype = _resolve_dtype(payload["dtype"])
        dataset = cls(
            states=payload["states"],
            actions=payload["actions"],
            refs=payload["refs"],
            dtype=dtype,
            target_mode=str(payload["target_mode"]),
            stride=int(payload["stride"]),
            trajectory_ids=payload["trajectory_ids"],
            start_indices=payload["start_indices"],
        )
        return dataset

    @classmethod
    def from_subset(
        cls,
        dataset_subset: Dataset[tuple[th.Tensor, th.Tensor]],
        dtype: th.dtype | None = None,
    ) -> "SequenceStateActionDataset":
        """Create a ``SequenceStateActionDataset`` from a subset wrapper dataset."""
        materialized = _materialize_subset_dataset(dataset_subset, dataset_cls=cls, dtype=dtype)
        if not isinstance(materialized, cls):
            raise TypeError(f"Expected materialized subset to be of type '{cls.__name__}'.")
        return materialized

    @classmethod
    def from_trajectories(
        cls,
        state_trajectories: Sequence[NDArray | th.Tensor],
        action_trajectories: Sequence[NDArray | th.Tensor],
        sequence_length: int,
        stride: int = 1,
        refs_trajectories: Sequence[NDArray | th.Tensor] | None = None,
        dtype: th.dtype = th.float32,
        target_mode: Literal["last", "per_step", "all"] = "last",
        trajectory_ids: Sequence[int] | None = None,
    ) -> "SequenceStateActionDataset":
        """Build fixed-length sequence windows from in-memory trajectories."""
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if len(state_trajectories) != len(action_trajectories):
            raise ValueError(
                "state_trajectories/action_trajectories length mismatch: "
                f"got {len(state_trajectories)} states and {len(action_trajectories)} actions."
            )
        if refs_trajectories is not None and len(state_trajectories) != len(refs_trajectories):
            raise ValueError(
                "state_trajectories/refs_trajectories length mismatch: "
                f"got {len(state_trajectories)} states and {len(refs_trajectories)} refs."
            )
        if trajectory_ids is not None and len(state_trajectories) != len(trajectory_ids):
            raise ValueError(
                "trajectory_ids must have one entry per trajectory, got "
                f"{len(trajectory_ids)} ids for {len(state_trajectories)} trajectories."
            )

        state_windows: list[th.Tensor] = []
        action_targets: list[th.Tensor] = []
        refs_windows: list[th.Tensor] = []
        window_trajectory_ids: list[int] = []
        window_start_indices: list[int] = []

        expected_nx: int | None = None
        expected_nu: int | None = None
        expected_nref: int | None = None

        for traj_index, (states, actions) in enumerate(zip(state_trajectories, action_trajectories, strict=True)):
            states_tensor = _to_tensor(states, name="states", dims=2)
            actions_tensor = _to_tensor(actions, name="actions", dims=2)
            refs_tensor = (
                _to_tensor(refs_trajectories[traj_index], name="refs", dims=2)
                if refs_trajectories is not None else None
            )

            if states_tensor.shape[0] != actions_tensor.shape[0]:
                raise ValueError(
                    "Trajectory state/action length mismatch: "
                    f"got {states_tensor.shape[0]} states and {actions_tensor.shape[0]} actions "
                    f"for trajectory index {traj_index}."
                )
            if refs_tensor is not None and states_tensor.shape[0] != refs_tensor.shape[0]:
                raise ValueError(
                    "Trajectory state/ref length mismatch: "
                    f"got {states_tensor.shape[0]} states and {refs_tensor.shape[0]} refs "
                    f"for trajectory index {traj_index}."
                )

            nx = int(states_tensor.shape[1])
            nu = int(actions_tensor.shape[1])
            if expected_nx is None:
                expected_nx = nx
                expected_nu = nu
                expected_nref = int(refs_tensor.shape[1]) if refs_tensor is not None else None
            elif nx != expected_nx or nu != expected_nu:
                raise ValueError(
                    "Inconsistent state/action dimensions across trajectories: "
                    f"expected (nx={expected_nx}, nu={expected_nu}), got (nx={nx}, nu={nu}) "
                    f"for trajectory index {traj_index}."
                )
            if refs_tensor is not None and expected_nref is not None and refs_tensor.shape[1] != expected_nref:
                raise ValueError(
                    "Inconsistent reference dimension across trajectories: "
                    f"expected nref={expected_nref}, got nref={refs_tensor.shape[1]} "
                    f"for trajectory index {traj_index}."
                )

            trajectory_id = traj_index if trajectory_ids is None else int(trajectory_ids[traj_index])
            num_steps = int(states_tensor.shape[0])
            if num_steps < sequence_length:
                continue

            for start_idx in range(0, num_steps - sequence_length + 1, stride):
                end_idx = start_idx + sequence_length
                state_windows.append(states_tensor[start_idx:end_idx])
                if target_mode == "last":
                    action_targets.append(actions_tensor[end_idx - 1])
                else:
                    action_targets.append(actions_tensor[start_idx:end_idx])
                if refs_tensor is not None:
                    refs_windows.append(refs_tensor[start_idx:end_idx])
                window_trajectory_ids.append(trajectory_id)
                window_start_indices.append(start_idx)

        nx = 0 if expected_nx is None else expected_nx
        nu = 0 if expected_nu is None else expected_nu
        nref = 0 if expected_nref is None else expected_nref
        if state_windows:
            states_tensor = th.stack(state_windows, dim=0)
            actions_tensor = th.stack(action_targets, dim=0)
            refs_tensor = th.stack(refs_windows, dim=0) if refs_windows else None
        else:
            states_tensor = th.empty((0, sequence_length, nx), dtype=dtype)
            if target_mode == "last":
                actions_tensor = th.empty((0, nu), dtype=dtype)
            else:
                actions_tensor = th.empty((0, sequence_length, nu), dtype=dtype)
            refs_tensor = th.empty((0, sequence_length, nref), dtype=dtype) if refs_trajectories is not None else None

        return cls(
            states=states_tensor,
            actions=actions_tensor,
            refs=refs_tensor,
            dtype=dtype,
            target_mode=target_mode,
            stride=stride,
            trajectory_ids=window_trajectory_ids,
            start_indices=window_start_indices,
        )

    @classmethod
    def from_mpc_dataset(
        cls,
        mpc_dataset: MPCDataset | os.PathLike,
        sequence_length: int,
        stride: int = 1,
        dtype: th.dtype = th.float32,
        use_references: bool = False,
        target_mode: Literal["last", "per_step", "all"] = "last",
    ) -> "SequenceStateActionDataset":
        """Create a sequence dataset by extracting fixed windows from an MPC dataset."""
        if use_references:
            raise NotImplementedError("Sequence reference extraction is not implemented yet.")

        resolved_mpc_dataset = _resolve_mpc_dataset(mpc_dataset)
        if resolved_mpc_dataset.memory_buffer:
            resolved_mpc_dataset.save(mode="a")

        if resolved_mpc_dataset._h5_file is None:
            raise ValueError("MPCDataset must have an HDF5 file loaded to build the sequence dataset.")

        state_trajectories: list[NDArray] = []
        action_trajectories: list[NDArray] = []
        trajectory_ids: list[int] = []

        indices = list(resolved_mpc_dataset._indices)
        for key in track(indices, description="Extracting sequence windows"):
            grp = resolved_mpc_dataset._h5_file[key]
            meta = MPCMeta.from_hdf5(grp)
            steps = int(meta.steps_simulated)
            if steps <= 0 or meta.feasible == False:
                continue

            traj = MPCTrajectory.from_hdf5(grp, fields=["states", "inputs"])
            state_trajectories.append(np.ascontiguousarray(traj.states[:steps, :]))
            action_trajectories.append(np.ascontiguousarray(traj.inputs[:steps, :]))
            trajectory_ids.append(int(meta.id))

        return cls.from_trajectories(
            state_trajectories=state_trajectories,
            action_trajectories=action_trajectories,
            sequence_length=sequence_length,
            stride=stride,
            dtype=dtype,
            target_mode=target_mode,
            trajectory_ids=trajectory_ids,
        )


def load_imitation_dataset(
    path: os.PathLike,
) -> StateActionDataset | SequenceStateActionDataset:
    """Load either a flat or sequence imitation dataset from a serialized ``.pt`` file."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset file not found at {source_path}.")

    payload = th.load(source_path, map_location="cpu")
    if not isinstance(payload, dict) or "states" not in payload:
        raise TypeError("Unsupported dataset file format. Expected a dict payload with 'states'.")

    states = payload["states"]
    states_tensor = states if isinstance(states, th.Tensor) else th.as_tensor(states)
    if states_tensor.ndim == 2:
        return StateActionDataset.load(source_path)
    if states_tensor.ndim == 3:
        return SequenceStateActionDataset.load(source_path)
    raise ValueError(
        "Unsupported serialized dataset state shape: "
        f"expected 2D or 3D states, got {tuple(states_tensor.shape)}."
    )


def split_sequence_dataset_by_trajectory(
    dataset: SequenceStateActionDataset,
    val_fraction: float = 0.1,
    generator: th.Generator | None = None,
) -> tuple[Subset[tuple[th.Tensor, th.Tensor]], Subset[tuple[th.Tensor, th.Tensor]]]:
    """Split a sequence dataset into train/validation subsets without trajectory leakage."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0., 1.).")
    if len(dataset) <= 0:
        raise ValueError("Cannot split an empty sequence dataset.")

    trajectory_ids = dataset.trajectory_ids.detach().cpu().to(dtype=th.int64)
    if trajectory_ids.numel() != len(dataset):
        raise ValueError(
            "trajectory_ids length mismatch: "
            f"got {trajectory_ids.numel()} ids for dataset size {len(dataset)}."
        )
    if th.any(trajectory_ids < 0):
        raise ValueError(
            "Trajectory-aware splitting requires non-negative trajectory_ids for all sequence windows."
        )

    unique_trajectory_ids = th.unique(trajectory_ids, sorted=True)
    if unique_trajectory_ids.numel() < 2:
        raise ValueError("Need at least two unique trajectories for a train/validation split.")

    num_val_trajectories = int(np.floor(float(unique_trajectory_ids.numel()) * val_fraction))
    num_val_trajectories = max(1, min(num_val_trajectories, int(unique_trajectory_ids.numel()) - 1))

    permutation = th.randperm(unique_trajectory_ids.numel(), generator=generator)
    val_trajectory_ids = unique_trajectory_ids.index_select(0, permutation[:num_val_trajectories])
    val_mask = th.isin(trajectory_ids, val_trajectory_ids)
    train_mask = ~val_mask

    train_indices = th.nonzero(train_mask, as_tuple=False).flatten().tolist()
    val_indices = th.nonzero(val_mask, as_tuple=False).flatten().tolist()
    if not train_indices or not val_indices:
        raise ValueError("Trajectory-aware split produced an empty train or validation subset.")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def save_state_action_dataset_subset(
    dataset: Dataset[tuple[th.Tensor, th.Tensor]],
    path: os.PathLike,
) -> bool:
    """Save a state-action dataset or subset split to ``.pt`` format.

    Parameters
    ----------
    dataset : Dataset[tuple[torch.Tensor, torch.Tensor]]
        Dataset instance or subset wrapper.
    path : str or os.PathLike
        Output path for the serialized split.

    Returns
    -------
    bool
        ``True`` if saving succeeded, otherwise ``False``.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(dataset, "save") and callable(dataset.save):
        dataset.save(target_path)
        return True

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        try:
            split_dataset = _materialize_subset_dataset(dataset)
            split_dataset.save(target_path)
            return True
        except (TypeError, ValueError) as exc:
            __logger__.warning(
                "Could not materialize dataset subset for type '%s': %s",
                type(dataset).__name__,
                exc,
            )

    __logger__.warning(f"Could not save dataset split for type '{type(dataset).__name__}'.")
    return False


def create_train_and_val_dataloader(
    training_config: ImitationTrainingConfig,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: th.dtype = th.float32,
) -> tuple[DataLoader[tuple[th.Tensor, th.Tensor]], DataLoader[tuple[th.Tensor, th.Tensor]]]:
    """
    Create train and validation DataLoaders for imitation learning from an ``MPCDataset``.

    Parameters
    ----------
    training_config : ImitationTrainingConfig
        Training configuration controlling dataset construction and splitting.
    shuffle : bool, optional
        Whether to shuffle sample indices each epoch.
    drop_last : bool, optional
        Drop the last incomplete batch if True.
    num_workers : int, optional
        Number of DataLoader worker processes. Defaults to ``0`` to avoid
        multiprocessing issues with shared HDF5 file handles.
    pin_memory : bool, optional
        Enable pinned host memory for faster host-to-device transfer.
    dtype : torch.dtype, optional
        Tensor dtype for states/actions emitted by the dataset.

    Returns
    -------
    train_dataloader : DataLoader[tuple[th.Tensor, th.Tensor]]
        The training DataLoader.
    val_dataloader : DataLoader[tuple[th.Tensor, th.Tensor]]
        The validation DataLoader.
        
    Raises
    ------
        ValueError if no samples are available in the dataset or if the split configuration is invalid.
    """
    val_fraction = float(training_config.val_fraction)
    seed = int(training_config.seed)
    use_references = bool(training_config.use_references)
    batch_size = int(training_config.batch_size)
    sequence_length = int(training_config.sequence_length)

    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0., 1.).")

    generator = build_generator(seed)

    if sequence_length <= 1:
        if training_config.split_strategy != "random":
            raise ValueError(
                "split_strategy='trajectory' requires sequence_length > 1 because "
                "StateActionDataset samples do not retain trajectory ids."
            )

        dataset = StateActionDataset.from_mpc_dataset(
            mpc_dataset=training_config.dataset_path,
            dtype=dtype,
            use_references=use_references,
            near_duplicate_radius=training_config.near_duplicate_radius,
        )
        if len(dataset) <= 0:
            raise ValueError(
                "No imitation-learning samples available in MPC dataset. "
                "Ensure at least one trajectory has steps_simulated > 0 and valid state/input arrays."
            )

        num_train = int(len(dataset) * (1.0 - val_fraction))
        num_val = len(dataset) - num_train
        train_dataset, val_dataset = random_split(
            dataset,
            [num_train, num_val],
            generator=generator,
        )
    else:
        dataset = SequenceStateActionDataset.from_mpc_dataset(
            mpc_dataset=training_config.dataset_path,
            sequence_length=sequence_length,
            stride=training_config.stride,
            dtype=dtype,
            use_references=use_references,
            target_mode=training_config.target_mode,
        )
        if len(dataset) <= 0:
            raise ValueError(
                "No sequence imitation-learning samples available in MPC dataset. "
                "Ensure at least one feasible trajectory has at least sequence_length simulated steps."
            )

        if training_config.split_strategy == "trajectory":
            train_dataset, val_dataset = split_sequence_dataset_by_trajectory(
                dataset=dataset,
                val_fraction=val_fraction,
                generator=generator,
            )
        elif training_config.split_strategy == "random":
            num_train = int(len(dataset) * (1.0 - val_fraction))
            num_val = len(dataset) - num_train
            train_dataset, val_dataset = random_split(
                dataset,
                [num_train, num_val],
                generator=generator,
            )
        else:
            raise ValueError(f"Unsupported split_strategy: {training_config.split_strategy}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, val_dataloader
