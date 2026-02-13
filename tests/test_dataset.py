import os
import unittest
from pathlib import Path
from time import perf_counter

import torch as th

from lyapunov_certified_imitation_learning.imitation_learning.dataset import ImitationLearningDataset


class TestImitationLearningDatasetTiming(unittest.TestCase):
	"""Unit tests with timing information for lazy imitation-learning dataset calls."""
	PATH_FILE = Path(__file__).with_name("mpc_dataset_path.txt")

	@classmethod
	def _resolve_dataset_path(cls) -> str:
		"""Resolve dataset path from env var or local path file."""
		env_path = os.environ.get("MPC_DATASET_PATH", "").strip()
		if env_path:
			return env_path

		if cls.PATH_FILE.exists():
			file_path = cls.PATH_FILE.read_text(encoding="utf-8").strip()
			if file_path:
				return file_path

		return ""

	@classmethod
	def setUpClass(cls) -> None:
		dataset_path = cls._resolve_dataset_path()
		if not dataset_path:
			raise unittest.SkipTest(
				"Set MPC_DATASET_PATH or create tests/mpc_dataset_path.txt with an existing MPC HDF5 dataset path."
			)

		cls.dataset_path = Path(dataset_path)
		if not cls.dataset_path.exists():
			raise unittest.SkipTest(f"Resolved dataset path does not exist: {cls.dataset_path}")

		cls._timings: dict[str, float] = {}

	@classmethod
	def tearDownClass(cls) -> None:
		if getattr(cls, "_timings", None):
			print("\nTiming summary (seconds):")
			for name, value in sorted(cls._timings.items()):
				print(f"  {name}: {value:.6f}")

	@staticmethod
	def _measure_call(fn, *args, repeats: int = 1, **kwargs):
		best = float("inf")
		result = None
		for _ in range(max(repeats, 1)):
			t0 = perf_counter()
			result = fn(*args, **kwargs)
			dt = perf_counter() - t0
			best = min(best, dt)
		return result, best

	def test_dataset_init_and_len_timing(self) -> None:
		dataset, t_init = self._measure_call(
			ImitationLearningDataset,
			str(self.dataset_path),
			repeats=3,
		)
		self.__class__._timings["init_from_path"] = t_init

		size, t_len = self._measure_call(len, dataset, repeats=5)
		self.__class__._timings["len"] = t_len

		self.assertGreater(size, 0)
		self.assertGreater(len(dataset.id_to_sim_steps), 0)

	def test_map_sample_to_traj_timing(self) -> None:
		dataset = ImitationLearningDataset(str(self.dataset_path))
		mid_idx = len(dataset) // 2

		mapped, t_map = self._measure_call(
			dataset._map_sample_to_traj,
			mid_idx,
			repeats=50,
		)
		self.__class__._timings["map_sample_to_traj"] = t_map

		traj_idx, step_idx = mapped
		self.assertGreaterEqual(traj_idx, 0)
		self.assertGreaterEqual(step_idx, 0)

	def test_getitem_timing_and_shapes(self) -> None:
		dataset = ImitationLearningDataset(str(self.dataset_path))
		n = len(dataset)
		indices = [0, n // 2, n - 1] if n > 2 else list(range(n))

		total_time = 0.0
		for i, idx in enumerate(indices):
			sample, t_get = self._measure_call(dataset.__getitem__, idx, repeats=10)
			total_time += t_get
			state, action = sample

			self.assertIsInstance(state, th.Tensor)
			self.assertIsInstance(action, th.Tensor)
			self.assertEqual(state.dtype, th.float32)
			self.assertEqual(action.dtype, th.float32)
			self.assertEqual(state.ndim, 1)
			self.assertEqual(action.ndim, 1)

		avg_time = total_time / max(len(indices), 1)
		self.__class__._timings["getitem_avg"] = avg_time


if __name__ == "__main__":
	unittest.main(verbosity=2)
