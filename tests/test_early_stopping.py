import unittest
import torch as th
import torch.nn as nn

from lcil.utils.early_stopping import EarlyStopping


class TestEarlyStopping(unittest.TestCase):
    """Unit tests for the EarlyStopping utility class."""

    def test_initial_call_registers_baseline(self) -> None:
        model = nn.Linear(2, 1)
        es = EarlyStopping(patience=3, delta=0.0)

        self.assertIsNone(es.best_score)
        self.assertFalse(es.early_stop)
        self.assertEqual(es.counter, 0)

        es(val_loss=1.5, model=model)

        self.assertEqual(es.best_score, -1.5)
        self.assertIsNotNone(es.best_model_state)
        self.assertEqual(es.counter, 0)
        self.assertFalse(es.early_stop)

    def test_improvement_resets_counter_and_updates_model_state(self) -> None:
        model = nn.Linear(2, 1)
        es = EarlyStopping(patience=3, delta=0.0)

        es(val_loss=2.0, model=model)
        self.assertEqual(es.best_score, -2.0)

        # Loss increases -> counter increments
        es(val_loss=2.1, model=model)
        self.assertEqual(es.counter, 1)
        self.assertFalse(es.early_stop)

        # Loss improves (decreases) -> counter resets to 0, best score updates
        es(val_loss=1.8, model=model)
        self.assertEqual(es.counter, 0)
        self.assertEqual(es.best_score, -1.8)
        self.assertFalse(es.early_stop)

    def test_patience_exceeded_triggers_early_stop(self) -> None:
        model = nn.Linear(2, 1)
        es = EarlyStopping(patience=2, delta=0.0)

        es(val_loss=1.0, model=model)
        es(val_loss=1.1, model=model)
        self.assertEqual(es.counter, 1)
        self.assertFalse(es.early_stop)

        es(val_loss=1.2, model=model)
        self.assertEqual(es.counter, 2)
        self.assertTrue(es.early_stop)

    def test_delta_threshold_requires_sufficient_improvement(self) -> None:
        model = nn.Linear(2, 1)
        es = EarlyStopping(patience=2, delta=0.1)

        es(val_loss=1.0, model=model)
        # val_loss decreases from 1.0 to 0.95, improvement is 0.05 < delta (0.1)
        # score = -0.95 < best_score (-1.0) + delta (0.1) = -0.90
        es(val_loss=0.95, model=model)
        self.assertEqual(es.counter, 1)
        self.assertFalse(es.early_stop)

        # val_loss decreases to 0.85, improvement is 0.15 >= delta (0.1)
        es(val_loss=0.85, model=model)
        self.assertEqual(es.counter, 0)
        self.assertEqual(es.best_score, -0.85)

    def test_load_best_model_restores_parameters(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        with th.no_grad():
            model.weight.fill_(1.0)

        es = EarlyStopping(patience=2)
        es(val_loss=0.5, model=model)

        # Modify model weight in later worse epoch
        with th.no_grad():
            model.weight.fill_(99.0)
        es(val_loss=2.0, model=model)

        # Restore
        es.load_best_model(model)
        th.testing.assert_close(model.weight, th.tensor([[1.0, 1.0]]))

    def test_load_best_model_without_state_does_nothing(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        with th.no_grad():
            model.weight.fill_(42.0)
        es = EarlyStopping(patience=2)
        # best_model_state is None, should not raise or change model
        es.load_best_model(model)
        th.testing.assert_close(model.weight, th.tensor([[42.0, 42.0]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
