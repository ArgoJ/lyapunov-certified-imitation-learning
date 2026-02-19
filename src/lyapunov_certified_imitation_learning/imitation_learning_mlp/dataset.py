from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import torch as th

from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, random_split
from mpc_datagen import LinearLSCost, MPCConfig, MPCDataset, MPCMeta, MPCTrajectory

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

    def save_torch(self, path: str | os.PathLike[str]) -> None:
        """Save the preprocessed (optionally filtered) state-action pairs to a ``.pt`` file.

        Parameters
        ----------
        path : str or os.PathLike
            Output path for ``torch.save``.
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "states": self._states.detach().cpu(),
            "actions": self._actions.detach().cpu(),
            "dtype": str(self.dtype),
            "near_duplicate_radius": self.near_duplicate_radius,
            "num_samples": self._total_samples,
        }
        th.save(payload, target_path)

    def _load_samples(self) -> tuple[NDArray, NDArray]:
        """Load all valid samples into contiguous NumPy arrays."""
        if self.mpc_dataset.memory_buffer:
            self.mpc_dataset.save(mode="a")

        if self.mpc_dataset._h5_file is None:
            raise ValueError("MPCDataset must have an HDF5 file loaded to build the imitation-learning dataset.")

        states_chunks: list[NDArray] = []
        actions_chunks: list[NDArray] = []
        y_res_chunks: list[NDArray] = []
        steps_chunks: list[int] = []
        expected_nx: int | None = None
        expected_nu: int | None = None
        filter_duplicates = self.near_duplicate_radius is not None

        indices = list(self.mpc_dataset._indices)
        with __logger__.tqdm(indices, desc="Preloading trajectories") as pbar:
            for key in pbar:
                grp = self.mpc_dataset._h5_file[key]
                meta = MPCMeta.from_hdf5(grp)
                steps = int(meta.steps_simulated)
                if steps <= 0:
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
                steps_chunks.append(steps)

                if filter_duplicates:
                    global_cfg_grp = MPCConfig._get_global_cfg_grp(grp)
                    local_cfg_grp = MPCConfig._get_local_cfg_grp(grp)

                    base_grp = global_cfg_grp if global_cfg_grp is not None else local_cfg_grp
                    overwrite_grp = local_cfg_grp if local_cfg_grp is not None else None
                    cost = LinearLSCost.from_hdf5(base_grp, overwrite_grp)
                    y_res_chunks.append(cost.get_y(states, actions) - cost.yref)

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

        if not filter_duplicates:
            return states, actions

        y_residuals: NDArray | None = None
        if y_res_chunks:
            y_residuals = np.ascontiguousarray(np.concatenate(y_res_chunks, axis=0))

        keep_idx = self._near_duplicate_keep_indices(
            states=states,
            actions=actions,
            radius=self.near_duplicate_radius,
            y_residuals=y_residuals,
        )
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
    def _near_duplicate_keep_indices(
        states: NDArray,
        actions: NDArray,
        radius: float,
        y_residuals: NDArray | None = None,
    ) -> NDArray:
        """Keep sample indices after adaptive near-duplicate voxel deduplication.

        Parameters
        ----------
        states : numpy.ndarray
            State samples with shape `(n_samples, nx)`.
        actions : numpy.ndarray
            Action samples with shape `(n_samples, nu)`.
        radius : float
            Base L2 radius controlling voxel size in normalized feature space.
            The base cell size is `radius / sqrt(d)` where `d = nx + nu`.
        y_residuals : numpy.ndarray or None, optional
            Optional residual vectors with shape `(n_samples, ny)` (or
            `(n_samples,)`). When provided, residual norm defines radial
            distance to reference for adaptive scaling and residual energy is
            used to choose the representative within each voxel.

        Returns
        -------
        keep_idx : numpy.ndarray
            Sorted indices of kept samples with shape `(n_kept,)`.

        Notes
        -----
        Concatenated `[state, action]` features are robustly normalized with
        median/IQR, then adaptively scaled radially before quantization so that
        voxels are effectively larger near reference and smaller toward bounds.
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

        if y_residuals is not None:
            y_res = np.asarray(y_residuals, dtype=np.float64)
            if y_res.ndim == 1:
                y_res = y_res.reshape(-1, 1)
            radial_source = y_res
        else:
            radial_source = normalized_features

        radial_distance = np.linalg.norm(radial_source, axis=1)
        radial_scale = float(np.percentile(radial_distance, 90))
        radial_scale = max(radial_scale, 1e-8)
        t = radial_distance / radial_scale
        adaptive_scale = 0.5 + 1.5 * (t / (1.0 + t))
        transformed_features = normalized_features * adaptive_scale[:, None]

        quantized = np.floor(transformed_features / cell_size).astype(np.int64, copy=False)
        _, unique_idx, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)

        if y_residuals is None:
            return np.sort(unique_idx.astype(np.int64, copy=False))

        squared_distance = np.einsum("ij,ij->i", y_res, y_res)
        if squared_distance.shape[0] != states.shape[0]:
            raise ValueError(
                "y_residuals length mismatch: "
                f"expected {states.shape[0]}, got {squared_distance.shape[0]}."
            )

        sort_order = np.lexsort((squared_distance, inverse))
        sorted_inverse = inverse[sort_order]
        _, first_pos = np.unique(sorted_inverse, return_index=True)
        keep_idx = sort_order[first_pos]
        return np.sort(keep_idx.astype(np.int64, copy=False))

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


def create_state_action_dataloader(
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
        state-action vectors. The voxel representative is selected by distance
        to per-sample stage references computed from ``LinearLSCost`` via
        ``cost.get_y(states, actions) - cost.yref`` when available.
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
    

def create_train_and_val_dataloader(
    mpc_dataset: MPCDataset | str | os.PathLike[str],
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
