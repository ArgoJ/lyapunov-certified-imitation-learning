import os
import tempfile
import unittest
from unittest.mock import patch

from pathlib import Path

import torch as th
from torch.utils.data import Subset

from lcil.imitation_learning.config import ImitationTrainingConfig
from lcil.imitation_learning.dataset import (
    StateActionDataset,
    SequenceStateActionDataset,
    create_train_and_val_dataloader,
    save_state_action_dataset_subset,
    split_sequence_dataset_by_trajectory,
)


class TestSequenceStateActionDataset(unittest.TestCase):
    def test_save_helper_supports_state_action_dataset_subsets(self) -> None:
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

    def test_from_trajectories_builds_last_token_windows(self) -> None:
        state_trajectories = [
            th.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]]),
            th.tensor([[10.0], [11.0], [12.0], [13.0], [14.0]]),
        ]
        action_trajectories = [
            th.tensor([[100.0], [101.0], [102.0], [103.0], [104.0]]),
            th.tensor([[200.0], [201.0], [202.0], [203.0], [204.0]]),
        ]

        dataset = SequenceStateActionDataset.from_trajectories(
            state_trajectories=state_trajectories,
            action_trajectories=action_trajectories,
            sequence_length=3,
            stride=1,
        )

        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.sequence_length, 3)
        self.assertEqual(dataset.target_mode, "last")
        th.testing.assert_close(dataset.trajectory_ids, th.tensor([0, 0, 0, 1, 1, 1]))
        th.testing.assert_close(dataset.start_indices, th.tensor([0, 1, 2, 0, 1, 2]))

        first_states, first_action = dataset[0]
        th.testing.assert_close(first_states, th.tensor([[0.0], [1.0], [2.0]]))
        th.testing.assert_close(first_action, th.tensor([102.0]))

        fourth_states, fourth_action = dataset[3]
        th.testing.assert_close(fourth_states, th.tensor([[10.0], [11.0], [12.0]]))
        th.testing.assert_close(fourth_action, th.tensor([202.0]))

    def test_per_step_targets_preserve_window_dimension(self) -> None:
        dataset = SequenceStateActionDataset.from_trajectories(
            state_trajectories=[th.tensor([[0.0], [1.0], [2.0], [3.0]])],
            action_trajectories=[th.tensor([[10.0], [11.0], [12.0], [13.0]])],
            sequence_length=2,
            stride=1,
            target_mode="per_step",
        )

        states, actions = dataset[1]

        self.assertEqual(states.shape, (2, 1))
        self.assertEqual(actions.shape, (2, 1))
        th.testing.assert_close(actions, th.tensor([[11.0], [12.0]]))

    def test_target_mode_all_is_accepted_as_per_step_alias(self) -> None:
        dataset = SequenceStateActionDataset.from_trajectories(
            state_trajectories=[th.tensor([[0.0], [1.0], [2.0], [3.0]])],
            action_trajectories=[th.tensor([[10.0], [11.0], [12.0], [13.0]])],
            sequence_length=2,
            stride=1,
            target_mode="all",
        )

        _, actions = dataset[0]

        self.assertEqual(dataset.target_mode, "all")
        self.assertEqual(actions.shape, (2, 1))

    def test_save_and_load_round_trip_preserves_metadata(self) -> None:
        dataset = SequenceStateActionDataset.from_trajectories(
            state_trajectories=[th.tensor([[0.0], [1.0], [2.0], [3.0]])],
            action_trajectories=[th.tensor([[5.0], [6.0], [7.0], [8.0]])],
            sequence_length=2,
            stride=2,
            target_mode="last",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "sequence_dataset.pt"
            dataset.save(checkpoint_path)
            loaded = SequenceStateActionDataset.load(checkpoint_path)

        self.assertEqual(loaded.sequence_length, 2)
        self.assertEqual(loaded.stride, 2)
        self.assertEqual(loaded.target_mode, "last")
        th.testing.assert_close(loaded.trajectory_ids, dataset.trajectory_ids)
        th.testing.assert_close(loaded.start_indices, dataset.start_indices)
        th.testing.assert_close(loaded._states, dataset._states)
        th.testing.assert_close(loaded._actions, dataset._actions)

    def test_save_helper_supports_sequence_dataset_subsets(self) -> None:
        dataset = SequenceStateActionDataset.from_trajectories(
            state_trajectories=[th.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])],
            action_trajectories=[th.tensor([[9.0], [10.0], [11.0], [12.0], [13.0]])],
            sequence_length=2,
            stride=1,
        )
        subset = Subset(dataset, [1, 3])

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "subset.pt"
            self.assertTrue(save_state_action_dataset_subset(subset, checkpoint_path))
            loaded = SequenceStateActionDataset.load(checkpoint_path)

        self.assertEqual(len(loaded), 2)
        th.testing.assert_close(loaded.trajectory_ids, th.tensor([0, 0]))
        th.testing.assert_close(loaded.start_indices, th.tensor([1, 3]))
        th.testing.assert_close(loaded[0][0], dataset[1][0])
        th.testing.assert_close(loaded[1][1], dataset[3][1])

    def test_split_by_trajectory_keeps_disjoint_trajectory_ids(self) -> None:
        dataset = SequenceStateActionDataset.from_trajectories(
            state_trajectories=[
                th.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]]),
                th.tensor([[10.0], [11.0], [12.0], [13.0], [14.0]]),
                th.tensor([[20.0], [21.0], [22.0], [23.0], [24.0]]),
            ],
            action_trajectories=[
                th.tensor([[100.0], [101.0], [102.0], [103.0], [104.0]]),
                th.tensor([[200.0], [201.0], [202.0], [203.0], [204.0]]),
                th.tensor([[300.0], [301.0], [302.0], [303.0], [304.0]]),
            ],
            sequence_length=3,
            stride=1,
        )

        generator = th.Generator().manual_seed(7)
        train_subset, val_subset = split_sequence_dataset_by_trajectory(
            dataset,
            val_fraction=1.0 / 3.0,
            generator=generator,
        )

        train_split = SequenceStateActionDataset.from_subset(train_subset)
        val_split = SequenceStateActionDataset.from_subset(val_subset)

        train_ids = set(train_split.trajectory_ids.tolist())
        val_ids = set(val_split.trajectory_ids.tolist())

        self.assertFalse(train_ids & val_ids)
        self.assertEqual(len(train_split) + len(val_split), len(dataset))
        self.assertEqual(train_ids | val_ids, set(dataset.trajectory_ids.tolist()))

    def test_split_by_trajectory_requires_multiple_trajectories(self) -> None:
        dataset = SequenceStateActionDataset.from_trajectories(
            state_trajectories=[th.tensor([[0.0], [1.0], [2.0], [3.0]])],
            action_trajectories=[th.tensor([[5.0], [6.0], [7.0], [8.0]])],
            sequence_length=2,
            stride=1,
        )

        with self.assertRaises(ValueError):
            split_sequence_dataset_by_trajectory(dataset, val_fraction=0.5)

    def test_flat_dataloader_split_seed_works_with_cuda_training_device(self) -> None:
        training_cfg = ImitationTrainingConfig(
            dataset_path="unused.h5",
            sequence_length=1,
            val_fraction=0.5,
            seed=7,
            split_strategy="random",
            batch_size=2,
            epochs=1,
        )
        dataset = StateActionDataset(
            states=th.tensor([[0.0], [1.0], [2.0], [3.0]]),
            actions=th.tensor([[10.0], [11.0], [12.0], [13.0]]),
        )

        with patch(
            "lcil.imitation_learning.dataset.StateActionDataset.from_mpc_dataset",
            return_value=dataset,
        ):
            train_loader, val_loader = create_train_and_val_dataloader(
                training_cfg
            )

        self.assertEqual(len(train_loader.dataset), 2)
        self.assertEqual(len(val_loader.dataset), 2)


class TestSequenceStateActionDatasetFromMPCDataset(unittest.TestCase):
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

    def _assert_mpc_window_shapes(self, sequence_length: int) -> None:
        dataset = SequenceStateActionDataset.from_mpc_dataset(
            self.dataset_path,
            sequence_length=sequence_length,
            stride=1,
            target_mode="last",
        )

        self.assertGreater(len(dataset), 0)
        self.assertEqual(dataset.sequence_length, sequence_length)
        self.assertEqual(dataset.target_mode, "last")
        self.assertEqual(dataset.trajectory_ids.shape[0], len(dataset))
        self.assertEqual(dataset.start_indices.shape[0], len(dataset))

        states, actions = dataset[0]
        self.assertIsInstance(states, th.Tensor)
        self.assertIsInstance(actions, th.Tensor)
        self.assertEqual(states.dtype, th.float32)
        self.assertEqual(actions.dtype, th.float32)
        self.assertEqual(states.ndim, 2)
        self.assertEqual(actions.ndim, 1)
        self.assertEqual(states.shape[0], sequence_length)

    def test_from_mpc_dataset_supports_single_step_windows(self) -> None:
        self._assert_mpc_window_shapes(sequence_length=1)

    def test_from_mpc_dataset_supports_multi_step_windows(self) -> None:
        self._assert_mpc_window_shapes(sequence_length=5)

    def test_create_train_and_val_dataloader_uses_sequence_config_values(self) -> None:
        training_cfg = ImitationTrainingConfig(
            dataset_path=self.dataset_path,
            sequence_length=3,
            stride=1,
            target_mode="all",
            val_fraction=0.5,
            seed=7,
            split_strategy="trajectory",
            batch_size=4,
            epochs=1,
        )

        train_loader, val_loader = create_train_and_val_dataloader(
            training_cfg,
        )

        train_dataset = SequenceStateActionDataset.from_subset(train_loader.dataset)
        val_dataset = SequenceStateActionDataset.from_subset(val_loader.dataset)

        self.assertEqual(train_dataset.sequence_length, 3)
        self.assertEqual(train_dataset.target_mode, "all")
        self.assertEqual(train_loader.batch_size, 4)
        self.assertFalse(set(train_dataset.trajectory_ids.tolist()) & set(val_dataset.trajectory_ids.tolist()))

    def test_create_train_and_val_dataloader_uses_flat_dataset_when_sequence_length_is_one(self) -> None:
        training_cfg = ImitationTrainingConfig(
            dataset_path=self.dataset_path,
            sequence_length=1,
            val_fraction=0.5,
            seed=7,
            split_strategy="random",
            near_duplicate_radius=1e-3,
            batch_size=3,
            epochs=1,
        )

        train_loader, val_loader = create_train_and_val_dataloader(
            training_cfg,
        )

        train_dataset = StateActionDataset.from_subset(train_loader.dataset)
        val_dataset = StateActionDataset.from_subset(val_loader.dataset)

        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(val_dataset), 0)
        self.assertEqual(train_loader.batch_size, 3)

    def test_flat_dataloader_rejects_trajectory_split(self) -> None:
        training_cfg = ImitationTrainingConfig(
            dataset_path=self.dataset_path,
            sequence_length=1,
            split_strategy="trajectory",
            epochs=1,
        )

        with self.assertRaises(ValueError):
            create_train_and_val_dataloader(training_cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)