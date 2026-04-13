from __future__ import annotations
import os
import logging
import numpy as np
import torch as th

from pathlib import Path
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, random_split
from rich.progress import track
from mpc_datagen import MPCDataset, MPCMeta, MPCTrajectory

__logger__ = logging.getLogger(__name__)


class StateActionDataset(Dataset[tuple[th.Tensor, th.Tensor]]):
    """In-memory imitation-learning dataset backed by ``MPCDataset``."""

    @staticmethod
    def _to_tensor_2d(array: NDArray | th.Tensor, name: str) -> th.Tensor:
        """Convert input arrays/tensors to contiguous 2D CPU tensors."""
        if isinstance(array, th.Tensor):
            tensor = array.detach().cpu()
        else:
            tensor = th.as_tensor(np.asarray(array))

        if tensor.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}.")

        return tensor.contiguous()

    @staticmethod
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

    @staticmethod
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

        states_tensor = self._to_tensor_2d(states, name="states")
        actions_tensor = self._to_tensor_2d(actions, name="actions")
        refs_tensor = self._to_tensor_2d(refs, name="refs") if refs is not None else None
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
        required_keys = {"states", "actions", "dtype", "near_duplicate_radius", "num_samples", "refs"}
        if not required_keys.issubset(payload.keys()):
            missing = required_keys - payload.keys()
            raise ValueError(f"Missing keys in dataset file: {missing}")

        states = payload["states"]
        actions = payload["actions"]
        references = payload["refs"]
        dtype_payload = payload["dtype"]
        near_duplicate_radius = payload["near_duplicate_radius"]

        dtype = cls._resolve_dtype(dtype_payload)
        dataset = cls(
            states=states,
            actions=actions,
            refs=references,
            dtype=dtype,
            near_duplicate_radius=None,
        )
        dataset.near_duplicate_radius = near_duplicate_radius
        
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
        resolved_mpc_dataset = cls._resolve_mpc_dataset(mpc_dataset)
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
        if not (hasattr(dataset_subset, "dataset") and hasattr(dataset_subset, "indices")):
            raise TypeError("dataset_subset must expose 'dataset' and 'indices' attributes.")

        parent_dataset = getattr(dataset_subset, "dataset")
        subset_indices = th.as_tensor(getattr(dataset_subset, "indices"), dtype=th.int64)
        parent_states = getattr(parent_dataset, "_states", None)
        parent_actions = getattr(parent_dataset, "_actions", None)
        parent_refs = getattr(parent_dataset, "_refs", None)

        if parent_states is None or parent_actions is None:
            raise TypeError("Subset parent dataset must expose '_states' and '_actions' tensors.")

        split_states = parent_states.index_select(0, subset_indices).detach().cpu()
        split_actions = parent_actions.index_select(0, subset_indices).detach().cpu()
        split_refs = parent_refs.index_select(0, subset_indices).detach().cpu() if parent_refs is not None else None

        resolved_dtype = dtype
        if resolved_dtype is None:
            parent_dtype = getattr(parent_dataset, "dtype", None)
            resolved_dtype = parent_dtype if isinstance(parent_dtype, th.dtype) else split_states.dtype

        return cls(
            states=split_states,
            actions=split_actions,
            refs=split_refs,
            dtype=resolved_dtype,
            near_duplicate_radius=None,
        )

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
        split_dataset = StateActionDataset.from_subset(dataset)
        split_dataset.save(target_path)
        return True

    __logger__.warning(f"Could not save dataset split for type '{type(dataset).__name__}'.")
    return False


def create_state_action_dataloader(
    mpc_dataset: MPCDataset | os.PathLike,
    batch_size: int = 256,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: th.dtype = th.float32,
    near_duplicate_radius: float | None = None,
) -> DataLoader[tuple[th.Tensor, th.Tensor]]:
    """Create a DataLoader for imitation learning from an ``MPCDataset``.

    Parameters
    ----------
    mpc_dataset : MPCDataset or path-like
        Source MPC dataset object or path to an HDF5 dataset.
    batch_size : int, optional
        Number of samples per batch.
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
    near_duplicate_radius : float, optional
        Optional near-duplicate filter radius in L2 distance over concatenated
        state-action vectors. The voxel representative is selected by distance
        to per-sample stage references computed from ``LinearLSCost`` via
        ``cost.get_y(states, actions) - cost.yref`` when available.
    """
    dataset = StateActionDataset.from_mpc_dataset(
        mpc_dataset=mpc_dataset,
        dtype=dtype,
        near_duplicate_radius=near_duplicate_radius,
    )
    if len(dataset) <= 0:
        raise ValueError(
            "No imitation-learning samples available in MPC dataset. "
            "Ensure at least one trajectory has steps_simulated > 0 and valid state/input arrays."
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    

def create_train_and_val_dataloader(
    mpc_dataset: MPCDataset | os.PathLike,
    batch_size: int = 256,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: th.dtype = th.float32,
    near_duplicate_radius: float | None = None,
    val_fraction: float = 0.1,
) -> tuple[DataLoader[tuple[th.Tensor, th.Tensor]], DataLoader[tuple[th.Tensor, th.Tensor]]]:
    """
    Create train and validation DataLoaders for imitation learning from an ``MPCDataset``.

    Parameters
    ----------
    mpc_dataset : MPCDataset or path-like
        Source MPC dataset object or path to an HDF5 dataset.
    batch_size : int, optional
        Number of samples per batch.
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
    near_duplicate_radius : float, optional
        Optional near-duplicate filter radius in L2 distance over concatenated
        state-action vectors. The voxel representative is selected by distance
        to per-sample stage references computed from ``LinearLSCost`` via
        ``cost.get_y(states, actions) - cost.yref`` when available.
    val_fraction : float, optional
        Fraction of samples to use for validation (default is 0.1).
        
    Returns
    -------
    train_dataloader : DataLoader[tuple[th.Tensor, th.Tensor]]
        The training DataLoader.
    val_dataloader : DataLoader[tuple[th.Tensor, th.Tensor]]
        The validation DataLoader.
        
    Raises
    ------
        ValueError if no samples are available in the dataset or if val_fraction is not in (0., 1.).
    """
    
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0., 1.).")

    dataset = StateActionDataset.from_mpc_dataset(
        mpc_dataset=mpc_dataset,
        dtype=dtype,
        near_duplicate_radius=near_duplicate_radius,
    )
    if len(dataset) <= 0:
        raise ValueError(
            "No imitation-learning samples available in MPC dataset. "
            "Ensure at least one trajectory has steps_simulated > 0 and valid state/input arrays."
        )

    num_train = int(len(dataset) * (1.0 - val_fraction))
    num_val = len(dataset) - num_train

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

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
