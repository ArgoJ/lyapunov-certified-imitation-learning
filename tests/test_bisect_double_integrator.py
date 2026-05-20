import importlib
import os
import sys
import unittest

import numpy as np
import torch as th
import torch.nn as nn
from plot_assertions_mixin import PlotAssertionsMixin
from shared_utils import analytical_quadratic_level_set_measure

from shared_double_integrator import (
    DoubleIntegratorConfig,
    DoubleIntegratorDynamics,
    RiccatiPolicy,
    RiccatiQuadraticLyapunov,
    riccati_gain_and_value_matrix,
)



from mpc_datagen.plots import roa
from lcil.utils.plot import certified_regions_2d, lyapunov_cert_regions
from lcil.certification.metrics import estimate_level_set_volume


class TestBisectDoubleIntegratorIntegration(PlotAssertionsMixin):
    @classmethod
    def setUpClass(cls) -> None:
        run_bisect_integration = os.environ.get("LCIL_RUN_BISECT_INTEGRATION", "0") == "1"
        run_legacy_abcrown_integration = os.environ.get("LCIL_RUN_ABCROWN_INTEGRATION", "0") == "1"
        if not run_bisect_integration and not run_legacy_abcrown_integration:
            raise unittest.SkipTest(
                "Set LCIL_RUN_BISECT_INTEGRATION=1 to run real bisect integration tests. "
                "LCIL_RUN_ABCROWN_INTEGRATION=1 remains supported for compatibility."
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
            sys.modules.pop("lcil.certification.bisect_certifier", None)
            bisect_certifier = importlib.import_module("lcil.certification.bisect_certifier")
            cls.BisectCertifier = bisect_certifier.BisectCertifier
            cls.LyapunovCertificationConfig = importlib.import_module(
                "lcil.certification.config"
            ).LyapunovCertificationConfig
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise unittest.SkipTest(f"Could not import certifier modules: {exc}") from exc

    @classmethod
    def _make_certifier(cls):
        sys_cfg = DoubleIntegratorConfig(dt=0.1)
        k_gain, p_value = riccati_gain_and_value_matrix(sys_cfg)
        lyap_model = RiccatiQuadraticLyapunov(p_value)
        config = cls.LyapunovCertificationConfig(
            state_dim=2,
            cert_bounds=np.array(
                [
                    [-2.0, -2.0],
                    [2.0, 2.0],
                ],
                dtype=np.float32,
            ),
            kappa=0.001,
            rho_min=1e-4,
            rho_scaling=1.1,
            bins_per_dim=(6, 6),
            center_refinement_factor=(0.6, 0.6),
            origin_exclusion=0.0,
            max_scale_steps=6,
            max_bisection_steps=6,
            bisection_tol=0.005,
            cert_method="alpha-crown",
            condition_tolerance=1e-6,
            max_recursion_depth=4,
            batch_size=2048,
            abcrown_timeout=60,
            suppress_native_output=True,
            skip_boundary_core_cert=True,
        )
        return cls.BisectCertifier(
            policy_model=RiccatiPolicy(k_gain),
            lyap_model=lyap_model,
            dyn_model=DoubleIntegratorDynamics(sys_cfg),
            config=config,
            device=th.device("cpu"),
        )

    def _assert_region_plot_written(
        self,
        certification_result,
        stem: str,
        state_labels: list[str],
    ) -> None:
        self._assert_plot_written(
            plot_fn=certified_regions_2d,
            stem=stem,
            plot_kwargs={
                "certification_result": certification_result,
                "state_labels": state_labels,
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
        state_labels: list[str],
    ) -> None:
        lyap_func = self._to_numpy_lyapunov(
            lyap_model=lyap_model,
            state_dim=2,
        )
        self._assert_plot_written(
            plot_fn=lyapunov_cert_regions,
            stem=stem,
            plot_kwargs={
                "lyapunov_func": lyap_func,
                "state_labels": state_labels,
                "certification_result": certification_result,
            },
        )

    def _assert_roa_plot_written(
        self,
        lyap_model: nn.Module,
        certification_result,
        stem: str,
        state_labels: list[str],
    ) -> None:
        lyap_func = self._to_numpy_lyapunov(
            lyap_model=lyap_model,
            state_dim=2,
        )
        self._assert_plot_written(
            plot_fn=roa,
            stem=stem,
            plot_kwargs={
                "lyapunov_func": lyap_func,
                "c_level": certification_result.rho,
                "nx": 2,
                "state_labels": state_labels,
            },
        )

    def test_double_integrator_lqr(self) -> None:
        certifier = self._make_certifier()
        result = certifier.certify(rho_estimate=50.48)
        level_set_estimate = estimate_level_set_volume(
            certifier.lyap_model,
            rho=float(result.rho),
            num_states=2,
            num_directions=2048,
            device=th.device("cpu"),
        )
        lqr_area = analytical_quadratic_level_set_measure(
            rho=float(result.rho),
            p_matrix=certifier.lyap_model.p_matrix.detach().cpu().numpy(),
        )
        base = "double_integrator_bisect_integration_lqr"
        state_labels = ["$x$", "$v$"]

        self.assertIsInstance(float(result.rho), float)
        self.assertGreaterEqual(result.rho, certifier.config.rho_min)
        self.assertFalse(level_set_estimate.truncated)
        self.assertAlmostEqual(level_set_estimate.volume / lqr_area, 1.0, delta=5e-2)
        self._assert_region_plot_written(
            certification_result=result,
            stem=f"{base}_regions",
            state_labels=state_labels,
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem=f"{base}_lyapunov",
            state_labels=state_labels,
        )
        self._assert_roa_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem=f"{base}_roa",
            state_labels=state_labels,
        )
        self.assertGreaterEqual(result.uncertified_regions.shape[0], 0)
        self.assertGreater(result.certified_sublevel_regions.shape[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)