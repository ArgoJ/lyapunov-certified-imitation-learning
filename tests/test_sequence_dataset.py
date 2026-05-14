import tempfile
import unittest

from pathlib import Path

import torch as th
from torch.utils.data import Subset

from lcil.imitation_learning_mlp.dataset import (
    SequenceStateActionDataset,
    save_state_action_dataset_subset,
)


class TestSequenceStateActionDataset(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)