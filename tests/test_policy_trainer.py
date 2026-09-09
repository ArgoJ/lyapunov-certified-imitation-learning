import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th
from torch.utils.data import DataLoader

from lcil.imitation_learning.config import ImitationTrainingConfig
from lcil.imitation_learning.dataset import StateActionDataset
from lcil.imitation_learning.loss import ImitationLearningLossParts
from lcil.imitation_learning.models import BoundedPolicy
from lcil.imitation_learning.trainer import (
    PolicyEpochLossSummary,
    PolicyEpochMetrics,
    PolicyTrainer,
    _PolicyEpochLossAccumulator,
)
from lcil.utils.base_models import MLP


class TestPolicyEpochMetricsAndAccumulator(unittest.TestCase):
    def test_accumulator_without_loss_parts(self) -> None:
        acc = _PolicyEpochLossAccumulator()
        loss = th.tensor(2.5)
        acc.update(batch_size=4, loss=loss, loss_parts=None)
        summary = acc.finalize()

        self.assertEqual(summary.total, 2.5)
        self.assertTrue(np.isnan(summary.base_raw))
        self.assertTrue(np.isnan(summary.dynamics_raw))
        self.assertTrue(np.isnan(summary.base))
        self.assertTrue(np.isnan(summary.dynamics))

    def test_accumulator_with_loss_parts(self) -> None:
        acc = _PolicyEpochLossAccumulator()
        parts1 = ImitationLearningLossParts(
            base_raw=th.tensor(1.0),
            dynamics_raw=th.tensor(2.0),
            base_weight=1.5,
            dynamics_weight=1.5,
        )
        parts2 = ImitationLearningLossParts(
            base_raw=th.tensor(3.0),
            dynamics_raw=th.tensor(4.0),
            base_weight=1.5,
            dynamics_weight=1.5,
        )
        acc.update(batch_size=2, loss=th.tensor(4.5), loss_parts=parts1)
        acc.update(batch_size=2, loss=th.tensor(8.5), loss_parts=parts2)
        summary = acc.finalize()

        self.assertAlmostEqual(summary.total, 6.5)
        self.assertAlmostEqual(summary.base_raw, 2.0)
        self.assertAlmostEqual(summary.dynamics_raw, 3.0)
        self.assertAlmostEqual(summary.base, 3.0)
        self.assertAlmostEqual(summary.dynamics, 4.5)

    def test_metrics_num_epochs_validation(self) -> None:
        with self.assertRaises(ValueError):
            PolicyEpochMetrics.from_num_epochs(0)
        with self.assertRaises(ValueError):
            PolicyEpochMetrics.from_num_epochs(-3)

    def test_metrics_update_and_bounds(self) -> None:
        metrics = PolicyEpochMetrics.from_num_epochs(3)
        summary = PolicyEpochLossSummary(total=1.2, base_raw=0.5, dynamics_raw=0.7, base=0.5, dynamics=0.7)

        # None summary is a no-op
        metrics.update(0, None)
        self.assertEqual(metrics.epochs_completed, 0)
        self.assertTrue(np.isnan(metrics.loss[0]))

        # Out of bounds
        with self.assertRaises(IndexError):
            metrics.update(-1, summary)
        with self.assertRaises(IndexError):
            metrics.update(3, summary)

        # Valid update
        metrics.update(0, summary)
        self.assertEqual(metrics.epochs_completed, 1)
        self.assertEqual(metrics.loss[0], 1.2)
        self.assertEqual(metrics.base_raw[0], 0.5)

    def test_metrics_save(self) -> None:
        metrics = PolicyEpochMetrics.from_num_epochs(2)
        summary = PolicyEpochLossSummary(total=0.5, base_raw=0.2, dynamics_raw=0.3, base=0.2, dynamics=0.3)
        metrics.update(0, summary)

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "metrics.npz"
            metrics.save(file_path)
            self.assertTrue(file_path.exists())

            loaded = np.load(file_path)
            self.assertIn("loss", loaded)
            self.assertIn("epochs_completed", loaded)
            self.assertEqual(int(loaded["epochs_completed"]), 1)
            self.assertAlmostEqual(float(loaded["loss"][0]), 0.5)


class TestPolicyTrainer(unittest.TestCase):
    def _create_simple_model_and_dataloader(self, with_refs: bool = False):
        feature_net = MLP(layer_dims=[2, 4, 1], activations=["relu", "identity"])
        model = BoundedPolicy(feature_net=feature_net, u_min=-2.0, u_max=2.0)

        states = th.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        actions = th.tensor([[0.1], [0.2], [0.3], [0.4]])
        refs = th.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0]]) if with_refs else None

        dataset = StateActionDataset(states=states, actions=actions, refs=refs)
        if with_refs:
            dataset.refs = refs
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        return model, dataloader

    def test_configure_schedulers(self) -> None:
        model, loader = self._create_simple_model_and_dataloader()

        # Step
        cfg_step = ImitationTrainingConfig(
            epochs=5, scheduler_type="step", scheduler_kwargs={"step_size": 2, "gamma": 0.2}
        )
        trainer_step = PolicyTrainer(model=model, dataloader=loader, training_config=cfg_step)
        self.assertIsInstance(trainer_step.scheduler, th.optim.lr_scheduler.StepLR)

        # Cosine
        cfg_cos = ImitationTrainingConfig(
            epochs=5, scheduler_type="cosine", scheduler_kwargs={"eta_min": 1e-4}
        )
        trainer_cos = PolicyTrainer(model=model, dataloader=loader, training_config=cfg_cos)
        self.assertIsInstance(trainer_cos.scheduler, th.optim.lr_scheduler.CosineAnnealingLR)

        # Plateau
        cfg_plat = ImitationTrainingConfig(
            epochs=5, scheduler_type="plateau", scheduler_kwargs={"factor": 0.5, "patience": 1}
        )
        trainer_plat = PolicyTrainer(model=model, dataloader=loader, training_config=cfg_plat)
        self.assertIsInstance(trainer_plat.scheduler, th.optim.lr_scheduler.ReduceLROnPlateau)

        # None
        cfg_none = ImitationTrainingConfig(epochs=5, scheduler_type="none")
        trainer_none = PolicyTrainer(model=model, dataloader=loader, training_config=cfg_none)
        self.assertIsNone(trainer_none.scheduler)

        # Unknown
        cfg_unk = ImitationTrainingConfig(epochs=5, scheduler_type="none")
        object.__setattr__(cfg_unk, "scheduler_type", "unsupported_type")
        trainer_unk = PolicyTrainer(model=model, dataloader=loader, training_config=cfg_unk)
        self.assertIsNone(trainer_unk.scheduler)

    def test_scheduler_stepping(self) -> None:
        model, loader = self._create_simple_model_and_dataloader()

        # Step LR step
        cfg_step = ImitationTrainingConfig(
            epochs=5, learning_rate=0.1, scheduler_type="step", scheduler_kwargs={"step_size": 1, "gamma": 0.5}
        )
        trainer = PolicyTrainer(model=model, dataloader=loader, training_config=cfg_step)
        trainer.optimizer.step()
        trainer._scheduller_step(monitored_metric=1.0)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.05)

        # Plateau step
        cfg_plat = ImitationTrainingConfig(
            epochs=5, learning_rate=0.1, scheduler_type="plateau", scheduler_kwargs={"factor": 0.5, "patience": 0}
        )
        trainer_plat = PolicyTrainer(model=model, dataloader=loader, training_config=cfg_plat)
        trainer_plat._scheduller_step(monitored_metric=1.0)
        trainer_plat._scheduller_step(monitored_metric=1.0)
        self.assertAlmostEqual(trainer_plat.optimizer.param_groups[0]["lr"], 0.05)

    def test_trainer_applies_weight_decay_from_config(self) -> None:
        model, loader = self._create_simple_model_and_dataloader()
        trainer = PolicyTrainer(
            model=model,
            dataloader=loader,
            training_config=ImitationTrainingConfig(epochs=1, weight_decay=1e-2),
        )
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["weight_decay"], 1e-2)

    def test_extract_batch_with_references(self) -> None:
        model, loader = self._create_simple_model_and_dataloader(with_refs=True)
        trainer = PolicyTrainer(model=model, dataloader=loader, training_config=ImitationTrainingConfig(epochs=1))

        batch = next(iter(loader))
        nn_inputs, states, actions = trainer._extract_batch(batch)

        # With references: nn_inputs = states - refs
        expected_nn_inputs = states - batch[2]
        th.testing.assert_close(nn_inputs, expected_nn_inputs)
        th.testing.assert_close(actions, batch[1])

    def test_train_with_validation(self) -> None:
        model, train_loader = self._create_simple_model_and_dataloader()
        _, val_loader = self._create_simple_model_and_dataloader()

        cfg = ImitationTrainingConfig(
            epochs=2,
            learning_rate=1e-2,
        )
        trainer = PolicyTrainer(
            model=model,
            dataloader=train_loader,
            val_dataloader=val_loader,
            training_config=cfg,
        )

        train_metrics, val_metrics = trainer.train()

        self.assertIsNotNone(val_metrics)
        self.assertGreater(train_metrics.epochs_completed, 0)
        self.assertGreater(val_metrics.epochs_completed, 0)

    def test_trainer_save(self) -> None:
        model, train_loader = self._create_simple_model_and_dataloader()
        _, val_loader = self._create_simple_model_and_dataloader()

        cfg = ImitationTrainingConfig(epochs=1, learning_rate=1e-2)
        trainer = PolicyTrainer(
            model=model,
            dataloader=train_loader,
            val_dataloader=val_loader,
            training_config=cfg,
        )
        trainer.train()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer.save(tmp_dir)
            save_path = Path(tmp_dir)
            self.assertTrue((save_path / "train_dataset.pt").exists())
            self.assertTrue((save_path / "val_dataset.pt").exists())
            self.assertTrue((save_path / "training_config.json").exists())
            self.assertTrue((save_path / "policy_model.pt").exists())
            self.assertTrue((save_path / "training_metrics.npz").exists())
            self.assertTrue((save_path / "val_training_metrics.npz").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
