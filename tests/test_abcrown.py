import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm

plot_module = importlib.import_module("lcil.utils.plot")
certified_regions_2d = plot_module.certified_regions_2d
lyapunov = plot_module.lyapunov


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
    def __init__(self, alpha: float = 0.3, beta: float = 1.5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, x: th.Tensor) -> th.Tensor:
        r2 = (x * x).sum(dim=1, keepdim=True)
        return self.beta * r2 - self.alpha * (r2 * r2)


class TestABCrownCertifierIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("LCIL_RUN_ABCROWN_INTEGRATION", "0") != "1":
            raise unittest.SkipTest(
                "Set LCIL_RUN_ABCROWN_INTEGRATION=1 to run real ABCrown integration tests."
            )

        try:
            abcrown = importlib.import_module("abcrown")

            is_stubbed_module = (
                getattr(abcrown, "__file__", None) is None
                or not hasattr(getattr(abcrown, "ConfigBuilder", object), "from_defaults")
            )
            if is_stubbed_module:
                sys.modules.pop("abcrown", None)
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
            # Unit tests may import this module with stubbed abcrown symbols first.
            # Ensure we bind to the real abcrown package for integration testing.
            sys.modules.pop("lcil.certification.abcrown_wrapper", None)
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
            state_dim=3,
            state_bounds=np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=np.float32),
            kappa=0.1,
            invariance_weight=1.0,
            cert_bins_per_dim=4,
            origin_exclusion=0.0,
            max_scale_steps=6,
            max_bisection_steps=6,
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

    @staticmethod
    def _project_regions_to_2d(
        regions: np.ndarray,
        state_indices: tuple[int, int] = (0, 1),
    ) -> np.ndarray:
        return regions[:, :, list(state_indices)]

    def _assert_region_plot_written(
        self,
        certified_regions: np.ndarray,
        uncertified_regions: np.ndarray,
        stem: str,
    ) -> None:
        certified_regions_2d_view = self._project_regions_to_2d(certified_regions)
        uncertified_regions_2d_view = self._project_regions_to_2d(uncertified_regions)

        plot_dir = os.environ.get("LCIL_TEST_PLOT_DIR")
        if plot_dir:
            output_dir = Path(plot_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / f"{stem}.html"
            certified_regions_2d(
                certified_regions=certified_regions_2d_view,
                uncertified_regions=uncertified_regions_2d_view,
                state_labels=["x0", "x1"],
                html_path=str(html_path),
            )
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / f"{stem}.html"
            certified_regions_2d(
                certified_regions=certified_regions_2d_view,
                uncertified_regions=uncertified_regions_2d_view,
                state_labels=["x0", "x1"],
                html_path=str(html_path),
            )
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)

    @staticmethod
    def _to_numpy_lyapunov(
        lyap_model: nn.Module,
        state_dim: int,
        plot_state_indices: tuple[int, int],
    ):
        def _lyapunov_func(x: np.ndarray) -> np.ndarray | float:
            x_array = np.asarray(x, dtype=np.float32)

            if x_array.shape[-1] == len(plot_state_indices):
                x_lifted = np.zeros((*x_array.shape[:-1], state_dim), dtype=np.float32)
                x_lifted[..., plot_state_indices[0]] = x_array[..., 0]
                x_lifted[..., plot_state_indices[1]] = x_array[..., 1]
            elif x_array.shape[-1] == state_dim:
                x_lifted = x_array
            else:
                raise ValueError(
                    f"Lyapunov input has invalid shape {x_array.shape}; expected last dim "
                    f"{len(plot_state_indices)} or {state_dim}."
                )

            x_tensor = th.as_tensor(x_lifted, dtype=th.float32)
            if x_tensor.ndim == 1:
                x_tensor = x_tensor.unsqueeze(0)

            with th.no_grad():
                values = lyap_model(x_tensor).reshape(-1)

            values_np = values.detach().cpu().numpy()
            if x_lifted.ndim == 1:
                return float(values_np[0])
            return values_np

        return _lyapunov_func

    def _assert_lyapunov_plot_written(
        self,
        lyap_model: nn.Module,
        certified_regions: np.ndarray,
        uncertified_regions: np.ndarray,
        stem: str,
    ) -> None:
        plot_state_indices = (0, 1)
        lyap_func = self._to_numpy_lyapunov(
            lyap_model=lyap_model,
            state_dim=3,
            plot_state_indices=plot_state_indices,
        )
        certified_regions_2d_view = self._project_regions_to_2d(
            certified_regions,
            state_indices=plot_state_indices,
        )
        uncertified_regions_2d_view = self._project_regions_to_2d(
            uncertified_regions,
            state_indices=plot_state_indices,
        )

        plot_dir = os.environ.get("LCIL_TEST_PLOT_DIR")
        if plot_dir:
            output_dir = Path(plot_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / f"{stem}.html"
            lyapunov(
                dataset=None,
                lyapunov_func=lyap_func,
                state_labels=["x0", "x1"],
                certified_regions=certified_regions_2d_view,
                uncertified_regions=uncertified_regions_2d_view,
                html_path=str(html_path),
            )
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / f"{stem}.html"
            lyapunov(
                dataset=None,
                lyapunov_func=lyap_func,
                state_labels=["x0", "x1"],
                certified_regions=certified_regions_2d_view,
                uncertified_regions=uncertified_regions_2d_view,
                html_path=str(html_path),
            )
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)

    def test_mixed_lyapunov_pipeline_with_real_abcrown(self) -> None:
        certifier = self._make_certifier(_MixedLyapunov(alpha=0.3, beta=2.0))

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
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="mixed_lyapunov_integration",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
