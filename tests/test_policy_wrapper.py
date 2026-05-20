import tempfile
import unittest

import torch as th

from pathlib import Path
from shared_utils import _RecordingSequencePolicy

from lcil.imitation_learning import SequenceStateActionDataset, StateActionDataset
from lcil.lyapunov_learning import FromRolloutsPolicyWrapper, RepeatCurrentPolicyWrapper


class TestRepeatCurrentPolicyWrapper(unittest.TestCase):
    def test_repeat_current_wrapper_repeats_state_window(self) -> None:
        policy = _RecordingSequencePolicy(max_seq_len=3, output_mode="sum")
        wrapper = RepeatCurrentPolicyWrapper(policy)

        x = th.tensor([[1.0, 2.0], [3.0, 4.0]])
        u = wrapper(x)

        assert policy.last_forward_input is not None
        self.assertEqual(tuple(policy.last_forward_input.shape), (2, 3, 2))
        th.testing.assert_close(policy.last_forward_input[0], th.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]))
        th.testing.assert_close(u, th.tensor([[3.0], [7.0]]))

    def test_repeat_current_wrapper_forward_raw_reduces_last_token(self) -> None:
        policy = _RecordingSequencePolicy(max_seq_len=4, output_mode="sum")
        wrapper = RepeatCurrentPolicyWrapper(policy)

        x = th.tensor([[2.0, 5.0]])
        u = wrapper.forward_raw(x)

        assert policy.last_forward_raw_input is not None
        self.assertEqual(tuple(policy.last_forward_raw_input.shape), (1, 4, 2))
        th.testing.assert_close(u, th.tensor([[4.0]]))


class TestFromRolloutsPolicyWrapper(unittest.TestCase):
    def test_from_rollouts_wrapper_uses_nearest_history_prefix(self) -> None:
        policy = _RecordingSequencePolicy(max_seq_len=3, output_mode="sum")
        rollout_dataset = SequenceStateActionDataset(
            states=th.tensor(
                [
                    [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                    [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],
                ]
            ),
            actions=th.tensor([[0.0], [1.0]]),
            target_mode="last",
            stride=1,
        )
        wrapper = FromRolloutsPolicyWrapper(policy, rollout_dataset)

        x = th.tensor([[2.5, 2.5], [11.5, 11.5]])
        u = wrapper(x)

        assert policy.last_forward_input is not None
        self.assertEqual(tuple(policy.last_forward_input.shape), (2, 3, 2))
        th.testing.assert_close(
            policy.last_forward_input[0],
            th.tensor([[0.0, 0.0], [1.0, 1.0], [2.5, 2.5]]),
        )
        th.testing.assert_close(
            policy.last_forward_input[1],
            th.tensor([[10.0, 10.0], [11.0, 11.0], [11.5, 11.5]]),
        )
        th.testing.assert_close(u, th.tensor([[5.0], [23.0]]))

    def test_from_rollouts_classmethod_loads_sequence_dataset_file(self) -> None:
        policy = _RecordingSequencePolicy(max_seq_len=2, output_mode="sum")
        rollout_dataset = SequenceStateActionDataset(
            states=th.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            actions=th.tensor([[0.0]]),
            target_mode="last",
            stride=1,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "rollout_windows.pt"
            rollout_dataset.save(dataset_path)
            wrapper = FromRolloutsPolicyWrapper.from_rollouts(policy, dataset_path)

        x = th.tensor([[5.0, 6.0]])
        _ = wrapper(x)

        assert policy.last_forward_input is not None
        th.testing.assert_close(
            policy.last_forward_input[0],
            th.tensor([[1.0, 2.0], [5.0, 6.0]]),
        )

    def test_from_rollouts_rejects_flat_dataset_file(self) -> None:
        policy = _RecordingSequencePolicy(max_seq_len=2, output_mode="sum")
        flat_dataset = StateActionDataset(
            states=th.tensor([[0.0, 1.0], [2.0, 3.0]]),
            actions=th.tensor([[0.0], [1.0]]),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "flat_dataset.pt"
            flat_dataset.save(dataset_path)

            with self.assertRaises(ValueError):
                FromRolloutsPolicyWrapper.from_rollouts(policy, dataset_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)