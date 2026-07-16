import unittest

import numpy as np
import torch as th

from torch.utils.data import DataLoader

from lcil.imitation_learning import (
    DynamicsAwareLoss,
    ImitationTrainingConfig,
    BaselineDynamicsAwareLoss,
    ScaledMSELoss,
    SequenceStateActionDataset,
    PolicyTrainer,
    TransformerPolicy
)


class AdditiveDynamics(th.nn.Module):
    def forward(self, states: th.Tensor, actions: th.Tensor) -> th.Tensor:
        return states + actions


class TestSequencePolicyTrainer(unittest.TestCase):
    def test_trainer_runs_with_last_token_sequence_batches(self) -> None:
        th.manual_seed(0)

        dataset = SequenceStateActionDataset.from_trajectories(
            state_trajectories=[th.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])],
            action_trajectories=[th.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])],
            sequence_length=2,
            stride=1,
            target_mode="last",
        )
        self.assertEqual(len(dataset), 4)

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        model = TransformerPolicy(
            input_dim=1,
            output_dim=1,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=16,
            max_seq_len=2,
            dropout=0.0,
            output_mode="last",
        )
        trainer = PolicyTrainer(
            model=model,
            dataloader=dataloader,
            training_config=ImitationTrainingConfig(epochs=1, learning_rate=1e-2),
        )

        nn_inputs, _, actions = trainer._extract_batch(next(iter(dataloader)))
        pred_actions = trainer._predict_actions(nn_inputs)

        self.assertEqual(nn_inputs.shape, (2, 2, 1))
        self.assertEqual(actions.shape, (2, 1))
        self.assertEqual(pred_actions.shape, (2, 1))

        trainer.training_config.tb_log_dir = None
        metrics = trainer.train()

        self.assertEqual(metrics.epochs_completed, 1)
        self.assertTrue(np.isfinite(metrics.train_loss[0]))
        self.assertTrue(np.isnan(metrics.train_scaled_raw[0]))
        self.assertTrue(np.isnan(metrics.train_dynamics_raw[0]))
        self.assertTrue(np.isnan(metrics.train_scaled[0]))
        self.assertTrue(np.isnan(metrics.train_dynamics[0]))

    def test_dynamics_aware_loss_accepts_last_token_sequence_batches(self) -> None:
        th.manual_seed(0)

        dataset = SequenceStateActionDataset.from_trajectories(
            state_trajectories=[th.tensor([[0.0], [0.1], [0.2], [0.3], [0.4]])],
            action_trajectories=[th.tensor([[0.0], [0.05], [0.1], [0.15], [0.2]])],
            sequence_length=3,
            stride=1,
            target_mode="last",
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        model = TransformerPolicy(
            input_dim=1,
            output_dim=1,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=16,
            max_seq_len=3,
            dropout=0.0,
            output_mode="last",
        )
        loss_fn = BaselineDynamicsAwareLoss(
            base_loss=ScaledMSELoss(scale=[1.0]),
            dynamics_loss=DynamicsAwareLoss(
                dynamics=AdditiveDynamics(),
                x_min=th.tensor([-1.0]),
                x_max=th.tensor([1.0]),
            ),
            dynamics_weight=1.0,
        )
        trainer = PolicyTrainer(
            model=model,
            dataloader=dataloader,
            training_config=ImitationTrainingConfig(epochs=1, learning_rate=1e-2),
            loss_fn=loss_fn,
        )

        trainer.training_config.tb_log_dir = None
        metrics = trainer.train()

        self.assertEqual(metrics.epochs_completed, 1)
        self.assertTrue(np.isfinite(metrics.train_loss[0]))
        self.assertTrue(np.isfinite(metrics.train_scaled_raw[0]))
        self.assertTrue(np.isfinite(metrics.train_dynamics_raw[0]))
        self.assertTrue(np.isfinite(metrics.train_scaled[0]))
        self.assertTrue(np.isfinite(metrics.train_dynamics[0]))
        self.assertTrue(np.isnan(metrics.val_scaled_raw[0]))
        self.assertTrue(np.isnan(metrics.val_dynamics_raw[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)