import tempfile
import unittest
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lcil.certification.config import LyapunovCertificationConfig
from lcil.imitation_learning.config import ImitationTrainingConfig
from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.trainer import (
    LyapunovTrainingCurriculumResult,
    LyapunovTrainingCurriculumStage,
    LyapunovTrainingResult,
)
from lcil.utils import GridSearchHelper
from lcil.utils.base_config import ArgumentParserConfig, JsonDataclass, config_field


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

    def test_lyapunov_training_result_save(self) -> None:
        training_result = LyapunovTrainingResult(
            rho_estimate=1.25,
            num_mined_counterexamples=7,
            train_time=3.5,
            aborted=True,
            abort_reason="rho threshold reached",
            lyap_model_path=Path("checkpoints/lyapunov_model.pt"),
            policy_model_path=Path("checkpoints/policy_model.pt"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            saved_path = training_result.save(out_dir)
            loaded_result = LyapunovTrainingResult.load(saved_path)

        self.assertEqual(saved_path.name, "training_result.json")
        self.assertEqual(loaded_result.__class__, training_result.__class__)
        self.assertEqual(loaded_result.to_dict(), training_result.to_dict())
        self.assertIsInstance(loaded_result.lyap_model_path, Path)
        self.assertIsInstance(loaded_result.policy_model_path, Path)

    def test_lyapunov_training_curriculum_result_save(self) -> None:
        curriculum_result = LyapunovTrainingCurriculumResult(
            stages=[
                LyapunovTrainingCurriculumStage(
                    stage_index=0,
                    state_bounds=np.array([[-1.0, -0.5], [1.0, 0.5]], dtype=float),
                    scale=np.array([0.5, 0.5], dtype=float),
                    result=LyapunovTrainingResult(
                        rho_estimate=0.8,
                        num_mined_counterexamples=2,
                        train_time=1.0,
                    ),
                ),
            ],
            aborted_result=LyapunovTrainingResult(
                rho_estimate=1.1,
                num_mined_counterexamples=4,
                train_time=2.0,
                aborted=True,
                abort_reason="monitor triggered",
            ),
            aborted_stage_index=1,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            saved_path = curriculum_result.save(out_dir)
            loaded_result = LyapunovTrainingCurriculumResult.load(saved_path)

        self.assertEqual(saved_path.name, "training_curriculum_result.json")
        self.assertEqual(loaded_result.__class__, curriculum_result.__class__)
        self.assertEqual(loaded_result.to_dict(), curriculum_result.to_dict())
        self.assertEqual(len(loaded_result.stages), 1)
        self.assertIsInstance(loaded_result.stages[0].state_bounds, np.ndarray)
        self.assertIsInstance(loaded_result.stages[0].scale, np.ndarray)
        self.assertIsInstance(loaded_result.stages[0].result, LyapunovTrainingResult)

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
            dataset_path="data/imitation.h5",
            sequence_length=3,
            stride=2,
            target_mode="all",
            val_fraction=0.15,
            seed=123,
            split_strategy="trajectory",
            use_references=False,
            near_duplicate_radius=5e-4,
            batch_size=128,
            epochs=12,
            restore_best_model=False,
            tb_log_dir="runs/test",
            learning_rate=5e-4,
            weight_decay=1e-5,
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
class DummyConfig(JsonDataclass):
    arr: np.ndarray | None
    optional: float | None
    NP_ARRAY_FIELDS = ("arr",)


@dataclass(frozen=True)
class DummyCliConfig(ArgumentParserConfig):
    epochs: int = config_field(default=10, description="Number of optimization epochs.")
    restore_best_model: bool = config_field(default=True, help="Restore the best checkpoint after training.")
    tb_log_dir: str | Path | None = config_field(default=None, help="Optional TensorBoard log directory.")
    hidden_sizes: tuple[int, ...] = config_field(default=(32, 32), help="Hidden layer sizes.")
    internal_only: str = config_field(default="skip", cli=False)


@dataclass(frozen=True)
class DummySweepConfig(ArgumentParserConfig):
    learning_rate: float = config_field(
        default=1e-3,
        help="Optimizer learning rate.",
        display_alias="lr",
        argparse_kwargs={"nargs": "+"},
    )
    kappa: float = config_field(
        default=0.1,
        help="Lyapunov decrease margin.",
        argparse_kwargs={"nargs": "+"},
    )
    hidden_sizes: tuple[int, ...] = config_field(default=(32, 32), help="Hidden layer sizes.")
    label: str = config_field(default="baseline", help="Run label.")


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


class TestArgumentParserConfig(unittest.TestCase):
    def test_add_to_argparse_uses_field_metadata_for_help(self) -> None:
        parser = ArgumentParser()
        DummyCliConfig().add_to_argparse(parser)

        help_text = parser.format_help()

        self.assertIn("--epochs", help_text)
        self.assertIn("Number of optimization epochs.", help_text)
        self.assertIn("Restore the best checkpoint after training.", help_text)
        self.assertIn("Optional TensorBoard log directory.", help_text)
        self.assertIn("Hidden layer sizes.", help_text)
        self.assertNotIn("--internal-only", help_text)

    def test_add_to_argparse_parses_common_config_types(self) -> None:
        parser = ArgumentParser()
        DummyCliConfig().add_to_argparse(parser)

        args = parser.parse_args([
            "--epochs", "25",
            "--no-restore-best-model",
            "--tb-log-dir", "runs/test",
            "--hidden-sizes", "64", "32",
        ])

        self.assertEqual(args.epochs, 25)
        self.assertFalse(args.restore_best_model)
        self.assertEqual(args.tb_log_dir, "runs/test")
        self.assertEqual(args.hidden_sizes, [64, 32])

    def test_add_to_argparse_supports_prefix_and_exclude(self) -> None:
        parser = ArgumentParser()
        DummyCliConfig().add_to_argparse(
            parser,
            prefix="train-",
            exclude_fields={"internal_only", "tb_log_dir"},
        )

        help_text = parser.format_help()

        self.assertIn("--train-epochs", help_text)
        self.assertIn("--train-hidden-sizes", help_text)
        self.assertNotIn("--train-tb-log-dir", help_text)
        self.assertNotIn("--train-internal-only", help_text)

    def test_add_to_argparse_supports_nargs_fields_for_scalars(self) -> None:
        parser = ArgumentParser()
        defaults = DummyCliConfig()
        defaults.add_to_argparse(parser, nargs_fields={"epochs"})

        args = parser.parse_args([
            "--epochs", "25", "30",
        ])
        configs = defaults.iter_from_namespace(args)

        self.assertEqual(args.epochs, [25, 30])
        self.assertEqual(len(configs), 2)
        self.assertEqual([cfg.epochs for cfg in configs], [25, 30])

    def test_iter_from_namespace_expands_nargs_scalar_fields(self) -> None:
        parser = ArgumentParser()
        defaults = DummySweepConfig()
        defaults.add_to_argparse(parser)

        args = parser.parse_args([
            "--learning-rate", "1e-3", "5e-4",
            "--kappa", "0.1", "0.2",
            "--hidden-sizes", "64", "32",
            "--label", "grid",
        ])

        configs = defaults.iter_from_namespace(args)

        self.assertEqual(len(configs), 4)
        self.assertEqual({cfg.learning_rate for cfg in configs}, {1e-3, 5e-4})
        self.assertEqual({cfg.kappa for cfg in configs}, {0.1, 0.2})
        self.assertTrue(all(cfg.hidden_sizes == (64, 32) for cfg in configs))
        self.assertTrue(all(cfg.label == "grid" for cfg in configs))

    def test_from_namespace_rejects_multiple_configs(self) -> None:
        parser = ArgumentParser()
        defaults = DummySweepConfig()
        defaults.add_to_argparse(parser)

        args = parser.parse_args([
            "--learning-rate", "1e-3", "5e-4",
        ])

        with self.assertRaises(ValueError):
            defaults.from_namespace(args)

    def test_from_namespace_keeps_union_sequence_values_scalar(self) -> None:
        parser = ArgumentParser()
        defaults = LyapunovCertificationConfig(
            state_dim=4,
            cert_bounds=np.array([[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]], dtype=float),
            bins_per_dim=4,
            center_refinement_factor=1.0,
        )
        defaults.add_to_argparse(
            parser,
            prefix="cert-",
            include_fields={"bins_per_dim", "center_refinement_factor", "rho_scaling"},
        )

        args = parser.parse_args([
            "--cert-bins-per-dim", "2", "6", "10", "10",
            "--cert-center-refinement-factor", "1.0", "0.8", "0.4", "0.4",
            "--cert-rho-scaling", "1.3",
        ])

        config = defaults.from_namespace(args, prefix="cert-")

        self.assertEqual(config.bins_per_dim, (2, 6, 10, 10))
        self.assertEqual(config.center_refinement_factor, (1.0, 0.8, 0.4, 0.4))
        self.assertEqual(config.rho_scaling, 1.3)

    def test_from_namespace_collapses_single_union_sequence_value_to_scalar(self) -> None:
        parser = ArgumentParser()
        defaults = LyapunovCertificationConfig(
            state_dim=4,
            cert_bounds=np.array([[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]], dtype=float),
            bins_per_dim=4,
            center_refinement_factor=1.0,
            origin_exclusion=0.0,
        )
        defaults.add_to_argparse(
            parser,
            prefix="cert-",
            include_fields={"bins_per_dim", "center_refinement_factor", "origin_exclusion"},
        )

        args = parser.parse_args([
            "--cert-bins-per-dim", "3",
            "--cert-center-refinement-factor", "0.7",
            "--cert-origin-exclusion", "0.1",
        ])

        config = defaults.from_namespace(args, prefix="cert-")

        self.assertEqual(config.bins_per_dim, (3, 3, 3, 3))
        self.assertEqual(config.center_refinement_factor, (0.7, 0.7, 0.7, 0.7))
        self.assertEqual(config.origin_exclusion, 0.1)

    def test_from_namespace_parses_skip_boundary_core_cert_boolean_flag(self) -> None:
        parser = ArgumentParser()
        defaults = LyapunovCertificationConfig(
            state_dim=2,
            cert_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=float),
            bins_per_dim=4,
            center_refinement_factor=1.0,
        )
        defaults.add_to_argparse(
            parser,
            include_fields={"skip_boundary_core_cert"},
        )

        args = parser.parse_args([
            "--skip-boundary-core-cert",
        ])
        config = defaults.from_namespace(args)

        self.assertTrue(config.skip_boundary_core_cert)


class TestGridSearchHelper(unittest.TestCase):
    def test_helper_infers_varying_fields_and_creates_run_dirs(self) -> None:
        parser = ArgumentParser()
        defaults = DummySweepConfig()
        defaults.add_to_argparse(parser)

        args = parser.parse_args([
            "--learning-rate", "1e-3", "5e-4",
            "--kappa", "0.1", "0.2",
            "--hidden-sizes", "64", "32",
            "--label", "grid",
        ])

        with tempfile.TemporaryDirectory() as tmp_dir:
            helper = GridSearchHelper.from_namespace(
                defaults,
                args,
                output_root=tmp_dir,
                sweep_id="20260425_120000",
                field_aliases={
                    "training_bound_scales": "curr",
                },
                extra_name_parts={
                    "lyap_eps": 0.1,
                    "training_bound_scales": [0.3, 0.6, 1.0],
                },
            )

            runs = list(helper)
            self.assertTrue(all(run.output_dir.is_dir() for run in runs))

        self.assertEqual(len(helper), 4)
        self.assertEqual(helper.run_name_fields, ("learning_rate", "kappa"))
        self.assertEqual(helper._sweep_base_path.name, "20260425_120000")
        self.assertEqual(runs[0].run_name, "lr_0.001__kappa_0.1__lyap_eps_0.1__curr_0.3-0.6-1")
        self.assertEqual(runs[0].description, "lr: 0.001, kappa: 0.1, lyap_eps: 0.1, curr: [0.3, 0.6, 1]")
        self.assertEqual(runs[-1].progress_message(), "[4/4] lr: 0.0005, kappa: 0.2, lyap_eps: 0.1, curr: [0.3, 0.6, 1]")

    def test_helper_explicit_aliases_override_display_alias_metadata(self) -> None:
        configs = [
            DummySweepConfig(learning_rate=1e-3, kappa=0.1),
            DummySweepConfig(learning_rate=5e-4, kappa=0.1),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            helper = GridSearchHelper(
                configs,
                output_root=tmp_dir,
                sweep_id="20260425_120001",
                field_aliases={"learning_rate": "eta"},
            )
            runs = list(helper)

        self.assertEqual(runs[0].run_name, "eta_0.001")
        self.assertEqual(runs[0].description, "eta: 0.001")


if __name__ == "__main__":
    unittest.main(verbosity=2)