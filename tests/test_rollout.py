import unittest
import tempfile

import numpy as np
import torch as th
import torch.nn as nn

from pathlib import Path

from lcil.lyapunov_learning import LyapunovRollout
from lcil.imitation_learning import SequenceStateActionDataset, StateActionDataset, load_imitation_dataset
from lcil.imitation_learning.policy_rollout import (
    FeasibleSetSampler,
    PolicyRolloutConfig,
    PolicyRolloutGenerator,
    RandomBoundsSampler,
)


class _ZeroPolicy(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)


class _IdentitySimulator(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return x


class _IncrementSimulator(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return x + 1.0


class _RecordingSequencePolicy(nn.Module):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.seen_inputs: list[np.ndarray] = []

    def forward(self, x: th.Tensor) -> th.Tensor:
        self.seen_inputs.append(x.detach().cpu().numpy().copy())
        return th.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)


class _QuadraticLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.sum(x * x, dim=1, keepdim=True)


class TestPolicyRolloutConfig(unittest.TestCase):
    def test_config_bounds_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            PolicyRolloutConfig(
                T_sim=10,
                dt=0.1,
                nx=2,
                nu=1,
                state_bounds=np.array([[-1.0, -1.0]]),
            )

    def test_random_bounds_sampler_samples_in_box(self) -> None:
        bounds = np.array([[-1.0, -2.0], [1.0, 2.0]], dtype=float)
        sampler = RandomBoundsSampler(bounds=bounds, seed=7)

        x0 = sampler.sample_x0()

        self.assertEqual(x0.shape, (2,))
        self.assertTrue(np.all(x0 >= bounds[0]))
        self.assertTrue(np.all(x0 <= bounds[1]))

    def test_feasible_set_sampler_uses_last_state_for_sequence_dataset(self) -> None:
        dataset = SequenceStateActionDataset.from_trajectories(
            state_trajectories=[th.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])],
            action_trajectories=[th.tensor([[0.0], [0.1], [0.2]])],
            sequence_length=2,
            stride=1,
            target_mode="last",
        )
        sampler = FeasibleSetSampler(dataset=dataset, seed=0)

        x0 = sampler.sample_x0()

        self.assertEqual(x0.shape, (2,))
        self.assertTrue(any(np.allclose(x0, candidate) for candidate in ([3.0, 4.0], [5.0, 6.0])))

    def test_load_imitation_dataset_detects_flat_and_sequence_payloads(self) -> None:
        flat_dataset = StateActionDataset(
            states=th.tensor([[0.0, 1.0], [2.0, 3.0]]),
            actions=th.tensor([[0.0], [1.0]]),
        )
        sequence_dataset = SequenceStateActionDataset.from_trajectories(
            state_trajectories=[th.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])],
            action_trajectories=[th.tensor([[0.0], [0.1], [0.2]])],
            sequence_length=2,
            stride=1,
            target_mode="last",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            flat_path = Path(tmp_dir) / "flat.pt"
            sequence_path = Path(tmp_dir) / "sequence.pt"
            flat_dataset.save(flat_path)
            sequence_dataset.save(sequence_path)

            loaded_flat = load_imitation_dataset(flat_path)
            loaded_sequence = load_imitation_dataset(sequence_path)

        self.assertIsInstance(loaded_flat, StateActionDataset)
        self.assertIsInstance(loaded_sequence, SequenceStateActionDataset)

    def test_generator_and_lyapunov_rollout_accept_fixed_initial_states(self) -> None:
        rollout_cfg = PolicyRolloutConfig(
            T_sim=3,
            dt=0.1,
            nx=2,
            nu=1,
        )
        generator = PolicyRolloutGenerator(
            policy=_ZeroPolicy(),
            simulator=_IdentitySimulator(),
            cfg=rollout_cfg,
            sampler=None,
            device="cpu",
        )

        initial_states = np.array(
            [
                [0.0, 0.0],
                [1.0, -2.0],
            ],
            dtype=np.float32,
        )

        dataset = generator.generate_from_states(initial_states)
        self.assertEqual(len(dataset), 2)

        first_traj = dataset[0].trajectory
        second_traj = dataset[1].trajectory
        np.testing.assert_allclose(first_traj.states[0], initial_states[0])
        np.testing.assert_allclose(second_traj.states[0], initial_states[1])
        np.testing.assert_allclose(first_traj.inputs, 0.0)
        np.testing.assert_allclose(second_traj.inputs, 0.0)

        lyapunov_rollout = LyapunovRollout(
            mpc_dataset=dataset,
            lyap_model=_QuadraticLyapunov(),
            device="cpu",
        )
        result_path = lyapunov_rollout.rollout()

        self.assertIsNone(result_path)
        np.testing.assert_allclose(first_traj.V_N, np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(second_traj.V_N, np.full(3, 5.0, dtype=np.float32))

    def test_generator_uses_recent_state_window_for_sequence_policy(self) -> None:
        rollout_cfg = PolicyRolloutConfig(
            T_sim=4,
            dt=0.1,
            nx=2,
            nu=1,
        )
        policy = _RecordingSequencePolicy(max_seq_len=3)
        generator = PolicyRolloutGenerator(
            policy=policy,
            simulator=_IncrementSimulator(),
            cfg=rollout_cfg,
            sampler=None,
            device="cpu",
        )

        dataset = generator.generate_from_states(np.array([[1.0, -2.0]], dtype=np.float32))

        self.assertEqual(len(policy.seen_inputs), 4)
        self.assertEqual([sample.shape for sample in policy.seen_inputs], [
            (1, 1, 2),
            (1, 2, 2),
            (1, 3, 2),
            (1, 3, 2),
        ])
        np.testing.assert_allclose(
            policy.seen_inputs[0][0],
            np.array([[1.0, -2.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            policy.seen_inputs[1][0],
            np.array([[1.0, -2.0], [2.0, -1.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            policy.seen_inputs[2][0],
            np.array([[1.0, -2.0], [2.0, -1.0], [3.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            policy.seen_inputs[3][0],
            np.array([[2.0, -1.0], [3.0, 0.0], [4.0, 1.0]], dtype=np.float32),
        )

        trajectory = dataset[0].trajectory
        np.testing.assert_allclose(
            trajectory.states,
            np.array(
                [
                    [1.0, -2.0],
                    [2.0, -1.0],
                    [3.0, 0.0],
                    [4.0, 1.0],
                    [5.0, 2.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_lyapunov_rollout_writes_successor_values_to_output_hdf5(self) -> None:
        rollout_cfg = PolicyRolloutConfig(
            T_sim=3,
            dt=0.1,
            nx=2,
            nu=1,
        )
        generator = PolicyRolloutGenerator(
            policy=_ZeroPolicy(),
            simulator=_IdentitySimulator(),
            cfg=rollout_cfg,
            sampler=None,
            device="cpu",
        )

        dataset = generator.generate_from_states(np.array([[1.0, -2.0]], dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source_rollout.hdf5"
            output_path = Path(tmp_dir) / "lyapunov_rollout.hdf5"
            dataset.save(path=source_path, save_ocp_trajs=False)

            file_backed_dataset = type(dataset).load(source_path)
            result_path = LyapunovRollout(
                mpc_dataset=file_backed_dataset,
                lyap_model=_QuadraticLyapunov(),
                device="cpu",
            ).rollout(output_path=output_path)

            self.assertEqual(result_path, output_path)

            written_dataset = type(dataset).load(output_path)
            np.testing.assert_allclose(
                written_dataset[0].trajectory.V_N,
                np.full(3, 5.0, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
