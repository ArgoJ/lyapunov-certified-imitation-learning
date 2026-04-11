import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lcil.certification.config import LyapunovCertificationConfig
from lcil.imitation_learning_mlp.config import ImitationTrainingConfig
from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.utils.config_io import JsonConfigMixin


class TestConfigRoundtrip(unittest.TestCase):
    def test_lyapunov_learning_config_save(self) -> None:
        training_cfg = LyapunovTrainingConfig(
            state_dim=2,
            state_bounds=np.array([[-2.0, -1.0], [2.0, 1.0]], dtype=float),
            batch_size=64,
            outer_epochs=2,
            steps_per_epoch=3,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            saved_path = training_cfg.save(out_dir)
            loaded_cfg = LyapunovTrainingConfig.load(saved_path)

        self.assertEqual(loaded_cfg.__class__, training_cfg.__class__)
        self.assertEqual(loaded_cfg.to_dict(), training_cfg.to_dict())

    def test_certification_config_save(self) -> None:
        certification_cfg = LyapunovCertificationConfig(
            state_dim=2,
            cert_bounds=np.array([[-3.0, -2.0], [3.0, 2.0]], dtype=float),
            bins_per_dim=4,
            center_refinement_factor=1.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            saved_path = certification_cfg.save(out_dir)
            loaded_cfg = LyapunovCertificationConfig.load(saved_path)

        self.assertEqual(loaded_cfg.__class__, certification_cfg.__class__)
        self.assertEqual(loaded_cfg.to_dict(), certification_cfg.to_dict())

    def test_imitation_training_config_save(self) -> None:
        training_cfg = ImitationTrainingConfig(
            epochs=12,
            restore_best_model=False,
            tb_log_dir="runs/test",
            learning_rate=5e-4,
            scheduler_type="plateau",
            scheduler_kwargs={
                "mode": "min",
                "factor": 0.5,
                "patience": 3,
                "min_lr": 1e-6,
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            saved_path = training_cfg.save(out_dir)
            loaded_cfg = ImitationTrainingConfig.load(saved_path)

        self.assertEqual(loaded_cfg.__class__, training_cfg.__class__)
        self.assertEqual(loaded_cfg.to_dict(), training_cfg.to_dict())


@dataclass(frozen=True)
class DummyConfig(JsonConfigMixin):
    arr: np.ndarray | None
    optional: float | None
    NP_ARRAY_FIELDS = ("arr",)


class TestJsonConfigMixin(unittest.TestCase):
    def test_mixin_roundtrip_with_empty_numpy_array(self) -> None:
        cfg = DummyConfig(arr=np.array([], dtype=float), optional=None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_path = cfg.save(tmp_dir)
            loaded_cfg = DummyConfig.load(saved_path)

        self.assertIsInstance(loaded_cfg.arr, np.ndarray)
        assert loaded_cfg.arr is not None
        self.assertEqual(loaded_cfg.arr.shape, (0,))
        self.assertEqual(loaded_cfg.optional, cfg.optional)
        self.assertEqual(loaded_cfg.to_dict(), cfg.to_dict())

    def test_mixin_roundtrip_with_none_numpy_field(self) -> None:
        cfg = DummyConfig(arr=None, optional=1.5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_path = cfg.save(tmp_dir)
            loaded_cfg = DummyConfig.load(saved_path)

        self.assertIsNone(loaded_cfg.arr)
        self.assertEqual(loaded_cfg.optional, cfg.optional)
        self.assertEqual(loaded_cfg.to_dict(), cfg.to_dict())

    def test_mixin_load_from_directory_path(self) -> None:
        cfg = DummyConfig(arr=np.array([1.0, 2.0], dtype=float), optional=None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            cfg.save(out_dir)
            loaded_cfg = DummyConfig.load(out_dir)

        self.assertEqual(loaded_cfg.to_dict(), cfg.to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)