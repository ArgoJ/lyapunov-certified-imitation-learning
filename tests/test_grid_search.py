import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from lcil.utils.grid_search import GridSearchHelper, GridSearchRun # Import-Pfad anpassen!


@dataclass
class DummyModelConfig:
    hidden_size: int
    layers: int = 2


@dataclass
class DummyTrainConfig:
    lr: float
    batch_size: int = 32


class TestGridSearchHelper(unittest.TestCase):
    """Unit tests for the functional logic of the GridSearchHelper."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_dir = Path(tempfile.mkdtemp(prefix="grid_search_test_"))
        cls.models = [DummyModelConfig(64), DummyModelConfig(128)]
        cls.trains = [DummyTrainConfig(0.01), DummyTrainConfig(0.001)]


    @classmethod
    def tearDownClass(cls) -> None:
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)


    def test_initialization_and_cartesian_product(self) -> None:
        helper = GridSearchHelper(self.models, self.trains, output_root=self.test_dir)
        self.assertEqual(len(helper), 4)
        self.assertTrue(helper.sweep_base_path.exists())


    def test_infer_varying_fields(self) -> None:
        helper = GridSearchHelper(self.models, self.trains, output_root=self.test_dir)
        self.assertIn("0.hidden_size", helper.run_name_fields)
        self.assertIn("1.lr", helper.run_name_fields)
        self.assertNotIn("0.layers", helper.run_name_fields)
        self.assertNotIn("1.batch_size", helper.run_name_fields)


    def test_run_generation_and_index_stripping(self) -> None:
        helper = GridSearchHelper(self.models, self.trains, output_root=self.test_dir)
        runs = list(helper)

        self.assertEqual(len(runs), 4)

        for run in runs:
            self.assertIsInstance(run, GridSearchRun)
            self.assertIsInstance(run.config, tuple)
            self.assertEqual(len(run.config), 2)
            
            model_cfg, train_cfg = run.config
            self.assertIsInstance(model_cfg, DummyModelConfig)
            self.assertIsInstance(train_cfg, DummyTrainConfig)
            self.assertTrue(run.output_dir.exists())
            self.assertTrue(run.output_dir.is_dir())
            self.assertNotIn("0.hidden_size", run.run_name)
            self.assertNotIn("1.lr", run.run_name)
            self.assertNotIn("0.hidden_size", run.description)
            self.assertNotIn("1.lr", run.description)
            self.assertIn("hidden_size", run.run_name)
            self.assertIn("lr", run.run_name)


    def test_field_aliases_override(self) -> None:
        aliases = {"0.hidden_size": "hs", "1.lr": "learning_rate"}
        helper = GridSearchHelper(
            self.models, 
            self.trains, 
            output_root=self.test_dir,
            field_aliases=aliases
        )
        runs = list(helper)
        first_run = runs[0]
        
        self.assertIn("hs:", first_run.description)
        self.assertIn("learning_rate:", first_run.description)
        self.assertIn("hs_", first_run.run_name)
        self.assertIn("learning_rate_", first_run.run_name)


    def test_empty_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            GridSearchHelper(output_root=self.test_dir)

        with self.assertRaises(ValueError):
            GridSearchHelper([], output_root=self.test_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)