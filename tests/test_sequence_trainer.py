import unittest

import numpy as np
import torch as th

from torch.utils.data import DataLoader

from lcil.imitation_learning_mlp.config import ImitationTrainingConfig
from lcil.imitation_learning_mlp.dataset import SequenceStateActionDataset
from lcil.imitation_learning_mlp.trainer import PolicyTrainer
from lcil.imitation_learning_mlp.transformer import TransformerPolicy


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

        metrics = trainer.train()

        self.assertEqual(metrics.epochs_completed, 1)
        self.assertTrue(np.isfinite(metrics.train_loss[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)