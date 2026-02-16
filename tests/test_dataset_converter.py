import os
import tempfile
import unittest
from pathlib import Path

from mpc_datagen import MPCDataset

from lyapunov_certified_imitation_learning.imitation_learning.dataset_converter import (
    StateActionDataset,
)


class TestStateActionDatasetConverter(unittest.TestCase):
    PATH_FILE = Path(__file__).with_name("mpc_dataset_path.txt")

    @classmethod
    def _resolve_dataset_path(cls) -> str:
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

    def test_from_mpc_dataset_converts_expected_pairs(self) -> None:
        source = MPCDataset.load(self.dataset_path)
        expected_pairs = sum(max(int(entry.meta.steps_simulated), 0) for entry in source)
        self.assertGreater(expected_pairs, 0)

        with tempfile.TemporaryDirectory(prefix="state_action_dataset_") as tmp_dir:
            output_path = Path(tmp_dir) / "converted_state_action_pairs.hdf5"

            converted = StateActionDataset.from_mpc_dataset(
                mpc_dataset=self.dataset_path,
                output_path=output_path,
                enforce_unique=False,
                flush_every=10_000,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(len(converted), expected_pairs)

            first_pair = converted[0]
            self.assertEqual(first_pair.state.ndim, 1)
            self.assertEqual(first_pair.action.ndim, 1)
            self.assertGreater(first_pair.state.size, 0)
            self.assertGreater(first_pair.action.size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
