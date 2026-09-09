import os
import unittest
from pathlib import Path
from time import perf_counter

import torch as th

from lcil.imitation_learning.dataset import StateActionDataset


class TestStateActionDatasetTiming(unittest.TestCase):
    """Unit tests with timing information for in-memory imitation-learning dataset calls."""
    PATH_FILE = Path(__file__).with_name("mpc_dataset_path.txt")
    _temp_dir = None

    @classmethod
    def _resolve_dataset_path(cls) -> str:
        """Resolve dataset path from env var or local path file, or generate a dummy dataset."""
        env_path = os.environ.get("MPC_DATASET_PATH", "").strip()
        if env_path:
            return env_path

        if cls.PATH_FILE.exists():
            file_path = cls.PATH_FILE.read_text(encoding="utf-8").strip()
            if file_path:
                return file_path

        import tempfile
        from shared_utils import create_dummy_mpc_dataset

        cls._temp_dir = tempfile.TemporaryDirectory()
        dummy_path = Path(cls._temp_dir.name) / "dummy_mpc_dataset.h5"
        create_dummy_mpc_dataset(dummy_path, num_trajectories=5, length=20, nx=4, nu=1)
        return str(dummy_path)

    @classmethod
    def setUpClass(cls) -> None:
        dataset_path = cls._resolve_dataset_path()
        cls.dataset_path = Path(dataset_path)
        cls._timings: dict[str, float] = {}

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._temp_dir is not None:
            cls._temp_dir.cleanup()
            cls._temp_dir = None

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
            StateActionDataset.from_mpc_dataset,
            str(self.dataset_path),
            repeats=3,
        )
        self.__class__._timings["init_from_path"] = t_init
        self.assertGreater(len(dataset), 0)

    def test_getitem_timing_and_shapes(self) -> None:
        dataset = StateActionDataset.from_mpc_dataset(str(self.dataset_path))
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
        dataset_full = StateActionDataset.from_mpc_dataset(str(self.dataset_path))
        dataset_filtered, t_filtered_init = self._measure_call(
            StateActionDataset.from_mpc_dataset,
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
            StateActionDataset.from_mpc_dataset(str(self.dataset_path), near_duplicate_radius=-1e-3)

        with self.assertRaises(ValueError):
            StateActionDataset(th.zeros(2, 2), th.zeros(2, 1), near_duplicate_radius=-0.5)


class TestStateActionDataset(unittest.TestCase):
    """Unit tests for StateActionDataset functionality."""

    def test_constructor_and_indexing(self) -> None:
        states = th.tensor([[1.0, 2.0], [3.0, 4.0]])
        actions = th.tensor([[0.1], [0.2]])
        refs = th.tensor([[1.0, 0.0], [3.0, 0.0]])

        dataset = StateActionDataset(states=states, actions=actions, refs=refs)
        self.assertEqual(len(dataset), 2)

        s, a, r = dataset[0]
        th.testing.assert_close(s, states[0])
        th.testing.assert_close(a, actions[0])
        th.testing.assert_close(r, refs[0])

        with self.assertRaises(IndexError):
            _ = dataset[2]

    def test_sample_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample count mismatch"):
            StateActionDataset(states=th.zeros(3, 2), actions=th.zeros(2, 1))

        with self.assertRaisesRegex(ValueError, "sample count mismatch"):
            StateActionDataset(states=th.zeros(3, 2), actions=th.zeros(3, 1), refs=th.zeros(2, 2))

    def test_save_and_load(self) -> None:
        import tempfile
        states = th.tensor([[1.0, 2.0], [3.0, 4.0]])
        actions = th.tensor([[0.1], [0.2]])
        dataset = StateActionDataset(states=states, actions=actions)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dataset.pt"
            dataset.save(path)
            loaded = StateActionDataset.load(path)
            self.assertEqual(len(loaded), 2)
            th.testing.assert_close(loaded[0][0], states[0])
            th.testing.assert_close(loaded[0][1], actions[0])
            th.testing.assert_close(loaded[1][0], states[1])
            th.testing.assert_close(loaded[1][1], actions[1])

    def test_save_state_action_dataset_subset(self) -> None:
        import tempfile
        from torch.utils.data import Subset
        from lcil.imitation_learning.dataset import save_state_action_dataset_subset

        dataset = StateActionDataset(
            states=th.tensor([[0.0], [1.0], [2.0], [3.0]]),
            actions=th.tensor([[5.0], [6.0], [7.0], [8.0]]),
        )
        subset = Subset(dataset, [0, 2])

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "state_subset.pt"
            self.assertTrue(save_state_action_dataset_subset(subset, checkpoint_path))
            loaded = StateActionDataset.load(checkpoint_path)

        self.assertEqual(len(loaded), 2)
        th.testing.assert_close(loaded[0][0], th.tensor([0.0]))
        th.testing.assert_close(loaded[1][1], th.tensor([7.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

