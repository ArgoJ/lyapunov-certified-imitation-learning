from __future__ import annotations
import os
from pathlib import Path

import torch as th

from bisect import bisect_right
from itertools import accumulate
from torch.utils.data import DataLoader, Dataset
from mpc_datagen import MPCDataset, MPCMeta


class ImitationLearningDataset(Dataset[tuple[th.Tensor, th.Tensor]]):
    """Lazy imitation-learning dataset backed by ``MPCDataset``."""

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
    
    def __init__(self, mpc_dataset: MPCDataset | os.PathLike[str], dtype: th.dtype = th.float32):
        """Initialize a lazy dataset from an `MPCDataset`.

        Parameters
        ----------
        mpc_dataset : MPCDataset or path-like
            Source MPC dataset (in-memory and/or HDF5-backed). Path-like
            values are treated as paths to an HDF5-backed dataset and loaded accordingly.
        dtype : torch.dtype, optional
            Output tensor dtype for states and actions.
        """
        self.mpc_dataset = self._resolve_mpc_dataset(mpc_dataset)
        self.dtype = dtype

        self.id_to_sim_steps, self._traj_locations, self._traj_lengths = self._build_index()
        self._cum_lengths = list(accumulate(self._traj_lengths))
        self._total_samples = self._cum_lengths[-1] if self._cum_lengths else 0

    def _build_index(self) -> tuple[dict[int, int], list[tuple[str, int | str]], list[int]]:
        """Build ID->steps map and trajectory access map without loading trajectory arrays."""
        id_to_sim_steps: dict[int, int] = {}
        traj_locations: list[tuple[str, int | str]] = []
        traj_lengths: list[int] = []

        # Ensure memory buffer is flushed to HDF5 for consistent indexing
        if self.mpc_dataset.memory_buffer:
            self.mpc_dataset.save(mode="a")

        if self.mpc_dataset._h5_file is None:
            raise ValueError("MPCDataset must have an HDF5 file loaded to build the imitation-learning dataset index.")

        for key in self.mpc_dataset._indices:
            grp = self.mpc_dataset._h5_file[key]
            meta = MPCMeta.from_hdf5(grp)

            if meta.steps_simulated <= 0:
                continue

            id_to_sim_steps[meta.id] = meta.steps_simulated
            traj_locations.append(("file", key))
            traj_lengths.append(meta.steps_simulated)

        return id_to_sim_steps, traj_locations, traj_lengths

    def __len__(self) -> int:
        """Total number of training samples across all trajectories."""
        return self._total_samples

    def _map_sample_to_traj(self, idx: int) -> tuple[int, int]:
        """Map global sample index to ``(trajectory_index, step_index)``."""
        if self._total_samples == 0:
            raise IndexError("Cannot index an empty imitation-learning dataset.")

        if idx < 0:
            idx += self._total_samples
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset size {self._total_samples}.")

        traj_idx = bisect_right(self._cum_lengths, idx)
        prev_end = 0 if traj_idx == 0 else self._cum_lengths[traj_idx - 1]
        step_idx = idx - prev_end
        return traj_idx, step_idx

    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor]:
        """Return one imitation pair ``(state, action)`` at global index ``idx``."""
        traj_idx, step_idx = self._map_sample_to_traj(idx)
        source, ref = self._traj_locations[traj_idx]

        if source == "mem":
            entry = self.mpc_dataset.memory_buffer[int(ref)]
            state_np = entry.trajectory.states[step_idx]
            action_np = entry.trajectory.inputs[step_idx]
        else:
            grp = self.mpc_dataset._h5_file[str(ref)]
            traj_grp = grp.get("trajectory", None)
            if traj_grp is None:
                raise ValueError("No 'trajectory' group found in the provided HDF5 group.")

            state_np = traj_grp["states"][step_idx, :]
            action_np = traj_grp["inputs"][step_idx, :]

        state = th.as_tensor(state_np, dtype=self.dtype)
        action = th.as_tensor(action_np, dtype=self.dtype)
        return state, action


def create_imitation_learning_dataloader(
    mpc_dataset: MPCDataset | str | os.PathLike[str],
    batch_size: int = 256,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: th.dtype = th.float32,
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
    """
    dataset = ImitationLearningDataset(mpc_dataset=mpc_dataset, dtype=dtype)
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