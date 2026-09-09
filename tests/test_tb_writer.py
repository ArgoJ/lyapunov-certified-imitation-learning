import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from lcil.imitation_learning.trainer import (
    PolicyEpochLossSummary,
    _tb_writer_add_scalar_if_finite,
    _tb_writer_add_summary,
    _tb_writer_build,
    _tb_writer_close,
)


class TestTensorboardWriterHelpers(unittest.TestCase):
    """Unit tests for TensorBoard writer logging helpers."""

    def test_tb_writer_build(self) -> None:
        self.assertIsNone(_tb_writer_build(None))

        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = _tb_writer_build(tmp_dir)
            self.assertIsNotNone(writer)
            _tb_writer_close(writer)

    def test_tb_writer_add_scalar_if_finite(self) -> None:
        mock_writer = MagicMock()

        # Finite value should be recorded
        _tb_writer_add_scalar_if_finite(mock_writer, "loss/train", 0.5, step=1)
        mock_writer.add_scalar.assert_called_once_with("loss/train", 0.5, 1)

        mock_writer.reset_mock()

        # NaN and Inf values should be ignored
        _tb_writer_add_scalar_if_finite(mock_writer, "loss/nan", float("nan"), step=2)
        _tb_writer_add_scalar_if_finite(mock_writer, "loss/inf", float("inf"), step=3)
        _tb_writer_add_scalar_if_finite(mock_writer, "loss/neginf", float("-inf"), step=4)
        mock_writer.add_scalar.assert_not_called()

    def test_tb_writer_add_summary(self) -> None:
        # None writer or None summary should do nothing
        _tb_writer_add_summary(None, None, prefix="Train", step=0)
        _tb_writer_add_summary(MagicMock(), None, prefix="Train", step=0)

        mock_writer = MagicMock()
        _tb_writer_add_summary(mock_writer, None, prefix="Train", step=0)
        mock_writer.add_scalar.assert_not_called()

        # Summary with mixed finite and NaN values
        summary = PolicyEpochLossSummary(
            total=1.5,
            base_raw=0.8,
            dynamics_raw=np.nan,
            base=0.8,
            dynamics=np.nan,
        )
        _tb_writer_add_summary(mock_writer, summary, prefix="Train", step=5)

        tags_called = [call.args[0] for call in mock_writer.add_scalar.call_args_list]
        self.assertIn("RawLoss/TrainBase", tags_called)
        self.assertIn("WeightedLoss/TrainTotal", tags_called)
        self.assertIn("WeightedLoss/TrainBase", tags_called)
        self.assertNotIn("RawLoss/TrainDynamics", tags_called)
        self.assertNotIn("WeightedLoss/TrainDynamics", tags_called)

    def test_tb_writer_close(self) -> None:
        # None writer should not error
        _tb_writer_close(None)

        mock_writer = MagicMock()
        _tb_writer_close(mock_writer)
        mock_writer.flush.assert_called_once()
        mock_writer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
