import os
import unittest
from pathlib import Path
from time import perf_counter

import torch as th

from lyapunov_certified_imitation_learning.imitation_learning.dataset import StateActionDataset


class TestStateActionDatasetTiming(unittest.TestCase):
	"""Unit tests with timing information for in-memory imitation-learning dataset calls."""
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
			StateActionDataset,
			str(self.dataset_path),
			repeats=3,
		)
		self.__class__._timings["init_from_path"] = t_init
		self.assertGreater(len(dataset), 0)

	def test_getitem_timing_and_shapes(self) -> None:
		dataset = StateActionDataset(str(self.dataset_path))
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

	def test_near_duplicate_radius_effect(self) -> None:
		dataset_full = StateActionDataset(str(self.dataset_path))
		dataset_filtered, t_filtered_init = self._measure_call(
			StateActionDataset,
			str(self.dataset_path),
			repeats=3,
			near_duplicate_radius=1e-3,
		)
		self.__class__._timings["init_filtered"] = t_filtered_init

		self.assertGreater(len(dataset_full), 0)
		self.assertGreater(len(dataset_filtered), 0)
		self.assertLessEqual(len(dataset_filtered), len(dataset_full))

		filtered_count = len(dataset_full) - len(dataset_filtered)
		filtered_ratio = filtered_count / len(dataset_full)
		print(
			f"Filtered samples: {filtered_count}/{len(dataset_full)} "
			f"({filtered_ratio:.2%}) with near_duplicate_radius=1e-3"
		)

		n = len(dataset_filtered)
		indices = [0, n // 2, n - 1] if n > 2 else list(range(n))
		for idx in indices:
			state_filtered, action_filtered = dataset_filtered[idx]
			self.assertIsInstance(state_filtered, th.Tensor)
			self.assertIsInstance(action_filtered, th.Tensor)

	def test_near_duplicate_radius_validation(self) -> None:
		with self.assertRaises(ValueError):
			StateActionDataset(str(self.dataset_path), near_duplicate_radius=0.0)

		with self.assertRaises(ValueError):
			StateActionDataset(str(self.dataset_path), near_duplicate_radius=-1e-3)


if __name__ == "__main__":
	unittest.main(verbosity=2)
