import torch as th
import numpy as np
import h5py
from pathlib import Path
import shutil

from contextlib import contextmanager
from collections.abc import Iterator
from numpy.typing import NDArray
from mpc_datagen import MPCDataset

from ..utils.package_logger import get_package_logger

__logger__ = get_package_logger(__name__)


class LyapunovRollout:

    def __init__(
            self,
            mpc_dataset: MPCDataset,
            lyap_model: th.nn.Module,
            device: th.device | str = "cpu"
    ) -> None:
        self.device = th.device(device)
        self.dataset = mpc_dataset
        self.lyap_model = lyap_model.to(self.device)

    def _compute_lyapunov_values(self, states: NDArray) -> NDArray:
        with th.no_grad():
            x_tensor = th.as_tensor(states, dtype=th.float32, device=self.device)
            v_tensor = self.lyap_model(x_tensor)

        v = np.asarray(v_tensor.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        if v.shape[0] != states.shape[0]:
            raise ValueError(
                "Lyapunov model output shape mismatch: "
                f"expected {states.shape[0]} values, got {v.shape[0]}."
            )
        return v

    @contextmanager
    def _dataset_write_mode(self) -> Iterator[None]:
        """Temporarily open the backing HDF5 file in ``r+`` mode and restore ``r`` mode after."""
        if self.dataset.file_path is None:
            raise ValueError("Dataset file path is not set. Cannot perform rollout. Save the dataset before.")

        previous_handle = self.dataset._h5_file
        if previous_handle is not None:
            previous_handle.close()
            self.dataset._h5_file = None

        try:
            self.dataset._h5_file = h5py.File(self.dataset.file_path, "r+")
            self.dataset._indices = self.dataset._collect_indices(self.dataset._h5_file)
            yield
        finally:
            if self.dataset._h5_file is not None:
                self.dataset._h5_file.close()
            self.dataset._h5_file = h5py.File(self.dataset.file_path, "r")
            self.dataset._indices = self.dataset._collect_indices(self.dataset._h5_file)

    def _prepare_output_path(self, output_path: str | Path | None) -> None:
        """If requested, copy the source dataset to ``output_path`` and switch this rollout to that file."""
        if output_path is None:
            return

        if self.dataset.file_path is None:
            raise ValueError("Dataset file path is not set. Cannot copy dataset to output path.")

        src_path = Path(self.dataset.file_path)
        dst_path = Path(output_path)

        try:
            same_path = src_path.resolve() == dst_path.resolve()
        except FileNotFoundError:
            same_path = False
        if same_path:
            return

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if self.dataset._h5_file is not None:
            self.dataset._h5_file.close()
            self.dataset._h5_file = None

        shutil.copy2(src_path, dst_path)

        self.dataset.file_path = dst_path
        self.dataset._h5_file = h5py.File(self.dataset.file_path, "r")
        self.dataset._indices = self.dataset._collect_indices(self.dataset._h5_file)

    def rollout(self, output_path: str | Path | None = None) -> Path:
        """Compute ``trajectory.V_N`` and write it in-place or into a copied dataset file.

        Parameters
        ----------
        output_path : str | Path | None
            If provided, the current dataset file is copied to this path and ``V_N`` is
            written to the copied file. If ``None``, the current dataset file is updated.

        Returns
        -------
        Path
            Path to the dataset file that was updated.
        """
        self._prepare_output_path(output_path)

        self.lyap_model.eval()
        with self._dataset_write_mode():
            file_idx_offset = len(self.dataset.memory_buffer)
            for file_idx, _ in enumerate(self.dataset._indices):
                dataset_idx = file_idx_offset + file_idx
                entry_grp = self.dataset.get_grp(dataset_idx)
                traj_grp = entry_grp.get("trajectory", None)
                if traj_grp is None:
                    raise ValueError(f"Missing 'trajectory' group in entry '{dataset_idx}'.")

                states = np.asarray(traj_grp["states"][:, :], dtype=np.float32)
                v = self._compute_lyapunov_values(states)

                if "V_N" in traj_grp:
                    del traj_grp["V_N"]
                traj_grp.create_dataset("V_N", data=v, compression="gzip")

        if self.dataset.file_path is None:
            raise RuntimeError("Dataset file path unexpectedly missing after rollout.")
        return Path(self.dataset.file_path)