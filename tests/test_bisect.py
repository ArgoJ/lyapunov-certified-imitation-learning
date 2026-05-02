import importlib
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn
from plot_assertions_mixin import PlotAssertionsMixin

plot_module = importlib.import_module("lcil.utils.plot")
certified_regions_2d = plot_module.certified_regions_2d
lyapunov_cert_regions = plot_module.lyapunov_cert_regions


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


class _IdentityDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return x


class _DirectionalScaleDynamics(nn.Module):
    def __init__(self, base_scale: float = 0.8, axis_gain: float = 0.4):
        super().__init__()
        self.base_scale = float(base_scale)
        self.axis_gain = float(axis_gain)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        scale = self.base_scale + self.axis_gain * x[:, :1]
        return scale * x


class _QuadraticLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x * x).sum(dim=1, keepdim=True)


class _MixedLyapunov(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 1.5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, x: th.Tensor) -> th.Tensor:
        r2 = (x * x).sum(dim=1, keepdim=True)
        return self.beta * r2 - self.alpha * (r2 * r2)


def _load_real_bisect_modules() -> tuple[type, type]:
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
            f"abcrown is missing required symbols for wrapper tests: {missing}"
        )

    try:
        # Unit tests may import this module with stubbed abcrown symbols first.
        # Ensure we bind to the real abcrown package before loading the certifier.
        sys.modules.pop("lcil.certification.bisect_certifier", None)
        bisect_certifier = importlib.import_module("lcil.certification.bisect_certifier")
        bisect_backend = bisect_certifier.BisectCertifier
        certification_config = importlib.import_module(
            "lcil.certification.config"
        ).LyapunovCertificationConfig
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise unittest.SkipTest(f"Could not import certifier modules: {exc}") from exc

    return bisect_backend, certification_config


class _BisectModuleLoaderMixin:
    @classmethod
    def _load_bisect_modules(cls) -> None:
        cls.BisectCertifier, cls.LyapunovCertificationConfig = _load_real_bisect_modules()


class TestBisectCertifierIntegration(_BisectModuleLoaderMixin, PlotAssertionsMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_bisect_integration = os.environ.get("LCIL_RUN_BISECT_INTEGRATION", "0") == "1"
        run_legacy_abcrown_integration = os.environ.get("LCIL_RUN_ABCROWN_INTEGRATION", "0") == "1"
        if not run_bisect_integration and not run_legacy_abcrown_integration:
            raise unittest.SkipTest(
                "Set LCIL_RUN_BISECT_INTEGRATION=1 to run real bisect integration tests. "
                "LCIL_RUN_ABCROWN_INTEGRATION=1 remains supported for compatibility."
            )

        cls._load_bisect_modules()

        cls._patchers = []

    @classmethod
    def tearDownClass(cls) -> None:
        for patcher in getattr(cls, "_patchers", []):
            patcher.stop()

    @classmethod
    def _make_certifier(
        cls,
        lyap_model: nn.Module,
        *,
        dyn_model: nn.Module | None = None,
        kappa: float = 0.1,
        rho_min: float = 1e-6,
    ):
        config = cls.LyapunovCertificationConfig(
            state_dim=3,
            cert_bounds=np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=np.float32),
            kappa=kappa,
            rho_min=rho_min,
            bins_per_dim=4,
            origin_exclusion=0.0,
            max_scale_steps=6,
            max_bisection_steps=6,
            cert_method="alpha-crown",
            condition_tolerance=1e-6,
            suppress_native_output=True,
            batch_size=512,
            max_recursion_depth=5,
        )
        return cls.BisectCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=lyap_model,
            dyn_model=_ZeroDynamics() if dyn_model is None else dyn_model,
            config=config,
            device=th.device("cpu"),
        )

    def _assert_region_plot_written(
        self,
        certification_result,
        stem: str,
    ) -> None:
        self._assert_plot_written(
            plot_fn=certified_regions_2d,
            stem=stem,
            plot_kwargs={
                "certification_result": certification_result,
                "state_labels": ["x0", "x1", "x2"],
            },
        )

    @staticmethod
    def _to_numpy_lyapunov(
        lyap_model: nn.Module,
        state_dim: int,
    ):
        def _lyapunov_func(x: np.ndarray) -> np.ndarray | float:
            x_array = np.asarray(x, dtype=np.float32)

            if x_array.shape[-1] == state_dim:
                x_lifted = x_array
            elif x_array.shape[-1] < state_dim:
                # Keep provided coordinates and pad trailing state dimensions with zeros.
                x_lifted = np.zeros((*x_array.shape[:-1], state_dim), dtype=np.float32)
                x_lifted[..., : x_array.shape[-1]] = x_array
            else:
                raise ValueError(
                    f"Lyapunov input has invalid shape {x_array.shape}; expected last dim "
                    f"<= {state_dim}."
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
        certification_result,
        stem: str,
    ) -> None:
        lyap_func = self._to_numpy_lyapunov(
            lyap_model=lyap_model,
            state_dim=3,
        )
        self._assert_plot_written(
            plot_fn=lyapunov_cert_regions,
            stem=stem,
            plot_kwargs={
                "dataset": None,
                "lyapunov_func": lyap_func,
                "state_labels": ["x0", "x1", "x2"],
                "certification_result": certification_result,
            },
        )

    def test_mixed_lyapunov_pipeline_with_real_bisect_backend(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_DirectionalScaleDynamics(base_scale=0.8, axis_gain=0.4),
            kappa=0.0,
            rho_min=0.9,
        )

        result = certifier.certify(rho_estimate=1.0)

        self.assertIsInstance(float(result.rho), float)
        self.assertGreaterEqual(result.rho, certifier.config.rho_min)
        self.assertGreater(result.certified_regions.shape[0], 0)
        self.assertGreater(result.failed_regions.shape[0], 0)
        certified_centers = result.certified_regions.mean(axis=1)
        failed_centers = result.failed_regions.mean(axis=1)
        self.assertTrue(np.any(certified_centers[:, 0] < 0.0))
        self.assertTrue(np.any(failed_centers[:, 0] > 0.0))
        self._assert_region_plot_written(
            certification_result=result,
            stem="mixed_regions_integration",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem="mixed_lyapunov_integration",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
