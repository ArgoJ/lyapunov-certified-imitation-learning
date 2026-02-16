from __future__ import annotations

import os
import h5py
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mpc_datagen import MPCDataset, MPCMeta
from mpc_datagen.mpc_data import BaseDataset

from ..utils.package_logger import get_package_logger

__logger__ = get_package_logger(__name__)


@dataclass(slots=True)
class StateActionPair:
    """One imitation-learning sample with state mapped to action."""

    state: np.ndarray
    action: np.ndarray

    def to_hdf5(self, grp: h5py.Group) -> None:
        """Serialize pair to HDF5 group."""
        grp.create_dataset("state", data=self.state)
        grp.create_dataset("action", data=self.action)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> StateActionPair:
        """Deserialize pair from HDF5 group."""
        state = grp["state"][:]
        action = grp["action"][:]
        return cls(state=state, action=action)


class StateActionDataset(BaseDataset[StateActionPair]):
    """`BaseDataset` implementation storing only state-action pairs."""

    DATA_CLS = StateActionPair
    DEFAULT_GRP_PREFIX = "pair_"

    def __init__(
        self,
        file_path: str | os.PathLike[str] | None = None,
        data_buffer: list[StateActionPair] | None = None,
        enforce_unique: bool = False,
        distance_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        distance_eps: float = 1e-6,
    ):
        super().__init__(
            file_path=str(file_path) if file_path is not None else None,
            data_buffer=data_buffer,
            grp_prefix=self.DEFAULT_GRP_PREFIX,
            data_cls=self.DATA_CLS,
        )
        if distance_eps < 0:
            raise ValueError("distance_eps must be non-negative.")

        self.enforce_unique = bool(enforce_unique)
        self._use_default_l2 = distance_fn is None
        self.distance_fn = distance_fn or (lambda a, b: float(np.linalg.norm(a - b)))
        self.distance_eps = float(distance_eps)

        self._known_states: np.ndarray | None = None
        self._known_actions: np.ndarray | None = None
        self._cache_initialized = False

    def _initialize_uniqueness_cache(self) -> None:
        if self._cache_initialized:
            return

        state_blocks: list[np.ndarray] = []
        action_blocks: list[np.ndarray] = []

        if self.memory_buffer:
            state_blocks.append(np.vstack([pair.state for pair in self.memory_buffer]))
            action_blocks.append(np.vstack([pair.action for pair in self.memory_buffer]))

        if self._h5_file is not None:
            for key in self._indices:
                grp = self._h5_file[key]
                pair = StateActionPair.from_hdf5(grp)
                state_blocks.append(np.asarray(pair.state, dtype=np.float32).reshape(1, -1))
                action_blocks.append(np.asarray(pair.action, dtype=np.float32).reshape(1, -1))

        if state_blocks:
            self._known_states = np.vstack(state_blocks)
            self._known_actions = np.vstack(action_blocks)
        else:
            self._known_states = None
            self._known_actions = None

        self._cache_initialized = True

    def _append_to_cache(self, states: np.ndarray, actions: np.ndarray) -> None:
        if states.size == 0:
            return

        if self._known_states is None or self._known_actions is None:
            self._known_states = states.copy()
            self._known_actions = actions.copy()
            return

        self._known_states = np.vstack((self._known_states, states))
        self._known_actions = np.vstack((self._known_actions, actions))

    def _is_duplicate_against(
        self,
        state: np.ndarray,
        action: np.ndarray,
        ref_states: np.ndarray | None,
        ref_actions: np.ndarray | None,
    ) -> bool:
        if ref_states is None or ref_actions is None or ref_states.shape[0] == 0:
            return False

        if self._use_default_l2:
            state_dist = np.linalg.norm(ref_states - state.reshape(1, -1), axis=1)
            state_match = state_dist <= self.distance_eps
            if not np.any(state_match):
                return False
            action_dist = np.linalg.norm(ref_actions[state_match] - action.reshape(1, -1), axis=1)
            return bool(np.any(action_dist <= self.distance_eps))

        for ref_state, ref_action in zip(ref_states, ref_actions, strict=True):
            if (
                self.distance_fn(state, ref_state) <= self.distance_eps
                and self.distance_fn(action, ref_action) <= self.distance_eps
            ):
                return True
        return False

    def _flush_to_path(self, path: Path, mode: str) -> None:
        """Flush buffer safely by closing any read handle before opening write handle."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
        self.save(path=path, mode=mode)

    def add_batch(self, states: np.ndarray, actions: np.ndarray) -> int:
        """Add a batch of state-action pairs and return number of inserted entries."""
        states_arr = np.asarray(states, dtype=np.float32)
        actions_arr = np.asarray(actions, dtype=np.float32)

        if states_arr.ndim == 1:
            states_arr = states_arr.reshape(1, -1)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr.reshape(1, -1)
        if states_arr.shape[0] != actions_arr.shape[0]:
            raise ValueError(
                f"states/actions batch mismatch: {states_arr.shape[0]} != {actions_arr.shape[0]}."
            )
        if states_arr.shape[0] == 0:
            return 0

        if not self.enforce_unique:
            self.memory_buffer.extend(
                StateActionPair(state=s, action=a)
                for s, a in zip(states_arr, actions_arr, strict=True)
            )
            return int(states_arr.shape[0])

        self._initialize_uniqueness_cache()

        accepted_states: list[np.ndarray] = []
        accepted_actions: list[np.ndarray] = []
        local_states: np.ndarray | None = None
        local_actions: np.ndarray | None = None

        for state, action in zip(states_arr, actions_arr, strict=True):
            if self._is_duplicate_against(state, action, self._known_states, self._known_actions):
                continue
            if self._is_duplicate_against(state, action, local_states, local_actions):
                continue

            accepted_states.append(state)
            accepted_actions.append(action)

            if local_states is None:
                local_states = state.reshape(1, -1)
                local_actions = action.reshape(1, -1)
            else:
                local_states = np.vstack((local_states, state.reshape(1, -1)))
                local_actions = np.vstack((local_actions, action.reshape(1, -1)))

        if not accepted_states:
            return 0

        accepted_states_arr = np.vstack(accepted_states)
        accepted_actions_arr = np.vstack(accepted_actions)

        self.memory_buffer.extend(
            StateActionPair(state=s, action=a)
            for s, a in zip(accepted_states_arr, accepted_actions_arr, strict=True)
        )
        self._append_to_cache(accepted_states_arr, accepted_actions_arr)
        return int(accepted_states_arr.shape[0])

    def add(self, entry: StateActionPair) -> bool:
        """Add pair to dataset. Returns True if inserted, False if filtered as duplicate."""
        inserted = self.add_batch(
            states=np.asarray(entry.state, dtype=np.float32),
            actions=np.asarray(entry.action, dtype=np.float32),
        )
        return bool(inserted)

    @staticmethod
    def _resolve_mpc_dataset(mpc_dataset: MPCDataset | os.PathLike[str]) -> MPCDataset:
        """Resolve `MPCDataset` from object or file path."""
        if isinstance(mpc_dataset, MPCDataset):
            return mpc_dataset

        if isinstance(mpc_dataset, (str, os.PathLike)):
            return MPCDataset.load(Path(mpc_dataset))

        raise TypeError(
            "mpc_dataset must be an MPCDataset or path-like object "
            f"(str, pathlib.Path, os.PathLike), got {type(mpc_dataset).__name__}."
        )

    @classmethod
    def from_mpc_dataset(
        cls,
        mpc_dataset: MPCDataset | os.PathLike[str],
        output_path: str | os.PathLike[str] | None = None,
        dtype: np.dtype | str = np.float32,
        flush_every: int = 50_000,
        enforce_unique: bool = True,
        distance_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        distance_eps: float = 1e-6,
    ) -> "StateActionDataset":
        """Build a state-action dataset by extracting pairs from an MPC dataset.

        Parameters
        ----------
        mpc_dataset : MPCDataset or path-like
            Source MPC dataset object or path to an HDF5 dataset.
        output_path : str or path-like, optional
            If provided, write converted pairs to this HDF5 file using append flushes.
            If omitted, result remains in memory.
        dtype : np.dtype or str, optional
            Numeric dtype used for saved state/action arrays.
        flush_every : int, optional
            Number of pairs buffered in memory before flushing to disk.
        enforce_unique : bool, optional
            If True, add-time uniqueness filtering is enabled.
        distance_fn : callable, optional
            Distance function for both states and actions: ``f(a, b) -> float``.
            Defaults to Euclidean norm.
        distance_eps : float, optional
            Maximum distance to be considered equal for both states and actions.
        """
        source = cls._resolve_mpc_dataset(mpc_dataset)
        out_dataset = cls(
            enforce_unique=enforce_unique,
            distance_fn=distance_fn,
            distance_eps=distance_eps,
        )
        target_path = Path(output_path) if output_path is not None else None
        target_exists = target_path is not None and target_path.exists()
        target_mode = "a" if target_exists else "w"
        out_dtype = np.dtype(dtype)

        if flush_every <= 0:
            raise ValueError("flush_every must be positive.")

        if source.memory_buffer and source._h5_file is not None:
            source.save(mode="a")

        pair_count = 0
        kept_count = 0

        if source._h5_file is not None:
            for key in __logger__.tqdm(source._indices, desc="Converting MPC trajectories"):
                grp = source._h5_file[key]
                meta = MPCMeta.from_hdf5(grp)
                steps = int(meta.steps_simulated)
                if steps <= 0:
                    continue

                traj_grp = grp.get("trajectory", None)
                if traj_grp is None:
                    raise ValueError("No 'trajectory' group found in the provided HDF5 group.")

                states = np.asarray(traj_grp["states"][:steps, :], dtype=out_dtype)
                actions = np.asarray(traj_grp["inputs"][:steps, :], dtype=out_dtype)
                kept_count += out_dataset.add_batch(states=states, actions=actions)
                pair_count += int(states.shape[0])

                if target_path is not None and len(out_dataset.memory_buffer) >= flush_every:
                    out_dataset._flush_to_path(path=target_path, mode=target_mode)
                    target_mode = "a"
        else:
            for entry in __logger__.tqdm(source.memory_buffer, desc="Converting MPC trajectories"):
                steps = int(entry.meta.steps_simulated)
                if steps <= 0:
                    continue

                states = np.asarray(entry.trajectory.states[:steps, :], dtype=out_dtype)
                actions = np.asarray(entry.trajectory.inputs[:steps, :], dtype=out_dtype)
                kept_count += out_dataset.add_batch(states=states, actions=actions)
                pair_count += int(states.shape[0])

                if target_path is not None and len(out_dataset.memory_buffer) >= flush_every:
                    out_dataset._flush_to_path(path=target_path, mode=target_mode)
                    target_mode = "a"

        if target_path is not None:
            if out_dataset.memory_buffer:
                out_dataset._flush_to_path(path=target_path, mode=target_mode)
            out_dataset = cls.load(target_path)

        __logger__.info(
            f"Converted {pair_count} state-action pairs from MPC dataset. "
            f"Kept {kept_count} after uniqueness filter."
        )
        return out_dataset