import importlib
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

# Ensure abcrown/auto_LiRPA runs without TorchScript JIT in tests.
os.environ["PYTORCH_JIT"] = "0"

import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm

certified_regions_2d = importlib.import_module("lcil.utils.plot").certified_regions_2d


class _ZeroPolicy(nn.Module):
    def __init__(self, nu: int = 1):
        super().__init__()
        self.nu = nu

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.zeros((x.shape[0], self.nu), dtype=x.dtype, device=x.device)


class _ZeroDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return th.zeros_like(x)


class _MixedLyapunov(nn.Module):
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: th.Tensor) -> th.Tensor:
        r2 = (x * x).sum(dim=1, keepdim=True)
        return r2 - self.alpha * (r2 * r2)


class TestABCrownCertifierIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("LCIL_RUN_ABCROWN_INTEGRATION", "0") != "1":
            raise unittest.SkipTest(
                "Set LCIL_RUN_ABCROWN_INTEGRATION=1 to run real ABCrown integration tests."
            )

        try:
            abcrown = importlib.import_module("abcrown")
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise unittest.SkipTest(f"abcrown is not importable: {exc}") from exc

        required_symbols = [
            "ABCrownSolver",
            "VerificationSpec",
            "ConfigBuilder",
            "input_vars",
            "output_vars",
        ]
        missing = [name for name in required_symbols if not hasattr(abcrown, name)]
        if missing:
            raise unittest.SkipTest(
                f"abcrown is missing required symbols for wrapper integration: {missing}"
            )

        try:
            abcrown_wrapper = importlib.import_module("lcil.certification.abcrown_wrapper")
            cls.ABCrownCertifier = abcrown_wrapper.ABCrownCertifier
            cls.LyapunovCertificationConfig = importlib.import_module(
                "lcil.certification.config"
            ).LyapunovCertificationConfig
            cls.certifier_base = importlib.import_module("lcil.certification.certifier_base")
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise unittest.SkipTest(f"Could not import certifier modules: {exc}") from exc

        cls._patchers = [
            mock.patch.object(cls.certifier_base.__logger__, "tqdm", tqdm),
        ]
        for patcher in cls._patchers:
            patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        for patcher in getattr(cls, "_patchers", []):
            patcher.stop()

    @classmethod
    def _make_certifier(cls, lyap_model: nn.Module):
        config = cls.LyapunovCertificationConfig(
            state_dim=2,
            state_bounds=np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32),
            kappa=0.1,
            invariance_weight=1.0,
            cert_step=1.0,
            cert_origin_exclusion=0.0,
            cert_max_scale_steps=6,
            cert_max_bisection_steps=6,
            cert_method="alpha-crown",
            condition_tolerance=1e-6,
        )
        return cls.ABCrownCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=lyap_model,
            dyn_model=_ZeroDynamics(),
            config=config,
            device=th.device("cpu"),
        )

    def _assert_region_plot_written(
        self,
        certified_regions: np.ndarray,
        uncertified_regions: np.ndarray,
        stem: str,
    ) -> None:
        plot_dir = os.environ.get("LCIL_TEST_PLOT_DIR")
        if plot_dir:
            output_dir = Path(plot_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / f"{stem}.html"
            certified_regions_2d(
                certified_regions=certified_regions,
                uncertified_regions=uncertified_regions,
                state_labels=["x0", "x1"],
                html_path=str(html_path),
            )
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / f"{stem}.html"
            certified_regions_2d(
                certified_regions=certified_regions,
                uncertified_regions=uncertified_regions,
                state_labels=["x0", "x1"],
                html_path=str(html_path),
            )
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)

    def test_mixed_lyapunov_pipeline_with_real_abcrown(self) -> None:
        certifier = self._make_certifier(_MixedLyapunov(alpha=0.3))

        rho_certified, result = certifier.certify(rho_estimate=1.0)

        self.assertIsInstance(float(rho_certified), float)
        self.assertGreaterEqual(rho_certified, certifier.config.rho_min)
        self.assertGreater(result.certified_regions.shape[0], 0)
        self.assertGreater(result.failed_regions.shape[0], 0)
        self.assertGreaterEqual(result.counter_examples.shape[0], result.failed_regions.shape[0])
        self._assert_region_plot_written(
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="mixed_regions_integration",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
