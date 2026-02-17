from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import torch as th

from torch.utils.data import DataLoader, Dataset
from mpc_datagen import MPCDataset, MPCMeta, MPCTrajectory

from ..utils.package_logger import get_package_logger

__logger__ = get_package_logger(__name__)


class StateActionDataset(Dataset[tuple[th.Tensor, th.Tensor]]):
    """In-memory imitation-learning dataset backed by ``MPCDataset``."""

    @staticmethod
    def _resolve_mpc_dataset(mpc_dataset: MPCDataset | os.PathLike[str]) -> MPCDataset:
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
        mpc_dataset: MPCDataset | os.PathLike[str],
        dtype: th.dtype = th.float32,
        near_duplicate_radius: float | None = None,
    ):
        """Initialize an in-memory dataset from an ``MPCDataset``.

        Parameters
        ----------
        mpc_dataset : MPCDataset or path-like
            Source MPC dataset (in-memory and/or HDF5-backed). Path-like
            values are treated as paths to an HDF5-backed dataset and loaded accordingly.
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

        self.mpc_dataset = self._resolve_mpc_dataset(mpc_dataset)
        self.dtype = dtype
        self.near_duplicate_radius = near_duplicate_radius
        
        self._states_np, self._actions_np = self._load_samples()
        self._states = th.as_tensor(self._states_np, dtype=self.dtype)
        self._actions = th.as_tensor(self._actions_np, dtype=self.dtype)
        self._total_samples = int(self._states_np.shape[0])

    def __len__(self) -> int:
        """Total number of training samples across all trajectories."""
        return self._total_samples

    def _load_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Load all valid samples into contiguous NumPy arrays."""
        if self.mpc_dataset.memory_buffer:
            self.mpc_dataset.save(mode="a")

        if self.mpc_dataset._h5_file is None:
            raise ValueError("MPCDataset must have an HDF5 file loaded to build the imitation-learning dataset.")

        states_chunks: list[np.ndarray] = []
        actions_chunks: list[np.ndarray] = []
        expected_nx: int | None = None
        expected_nu: int | None = None

        indices = list(self.mpc_dataset._indices)
        with __logger__.tqdm(indices, desc="Preloading trajectories") as pbar:
            for key in pbar:
                grp = self.mpc_dataset._h5_file[key]
                meta = MPCMeta.from_hdf5(grp)
                steps = int(meta.steps_simulated)
                if steps <= 0:
                    continue

                traj = MPCTrajectory.from_hdf5(grp, fields=["states", "inputs"])
                states = traj.states
                actions = traj.inputs
                if (states.shape[0] - 1) != steps or actions.shape[0] != steps:
                    raise ValueError(
                        f"Trajectory data length mismatch for ID {meta.id}: "
                        f"expected {steps} steps, got {states.shape[0]} states and {actions.shape[0]} actions."
                    )

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

                states_chunks.append(states[:steps, :])
                actions_chunks.append(actions[:steps, :])

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

        if self.near_duplicate_radius is None:
            return states, actions

        keep_idx = self._near_duplicate_keep_indices(states=states, actions=actions, radius=self.near_duplicate_radius)
        if keep_idx.size == states.shape[0]:
            return states, actions

        __logger__.info(
            "Near-duplicate filter kept %d/%d samples (radius=%.4e).",
            keep_idx.size,
            states.shape[0],
            self.near_duplicate_radius,
        )
        return np.ascontiguousarray(states[keep_idx]), np.ascontiguousarray(actions[keep_idx])

    @staticmethod
    def _near_duplicate_keep_indices(states: np.ndarray, actions: np.ndarray, radius: float) -> np.ndarray:
        """Return indices kept after vectorized near-duplicate voxel deduplication.

        The method robustly normalizes the concatenated ``[state, action]``
        features (median/IQR), then quantizes them into a regular grid and
        keeps at most one sample per voxel. Voxel size is chosen as
        ``radius / sqrt(d)`` for ``d`` total dimensions, so points inside one
        voxel are guaranteed to be within the L2 radius bound in normalized space.
        """
        if states.shape[0] == 0:
            return np.empty((0,), dtype=np.int64)

        features = np.concatenate((states, actions), axis=1)
        feature_median = np.median(features, axis=0)
        feature_iqr = np.percentile(features, 75, axis=0) - np.percentile(features, 25, axis=0)
        feature_iqr = np.maximum(feature_iqr, 1e-8)
        normalized_features = (features - feature_median) / feature_iqr

        feature_dim = features.shape[1]
        cell_size = radius / np.sqrt(float(feature_dim))

        quantized = np.floor(normalized_features / cell_size)
        _, unique_idx = np.unique(quantized, axis=0, return_index=True)
        return np.sort(unique_idx.astype(np.int64, copy=False))

    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor]:
        """Return one imitation pair ``(state, action)`` at global index ``idx``."""
        if self._total_samples == 0:
            raise IndexError("Cannot index an empty imitation-learning dataset.")

        if idx < 0:
            idx += self._total_samples
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset size {self._total_samples}.")

        if self._states is None or self._actions is None:
            raise RuntimeError("In-memory tensors are not available.")

        return self._states[idx], self._actions[idx]


def create_imitation_learning_dataloader(
    mpc_dataset: MPCDataset | str | os.PathLike[str],
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
        state-action vectors.
    """
    dataset = StateActionDataset(
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