import importlib
import os
import sys
import unittest

import numpy as np
import torch as th
import torch.nn as nn
from plot_assertions_mixin import PlotAssertionsMixin

from shared_cartpole import (
    PendulumOnCartConfig,
    riccati_gain_and_value_matrix,
    RiccatiPolicy,
    NonlinearInvertedPendulumOnCartDynamics,
    RiccatiQuadraticLyapunov
)

plot_module = importlib.import_module("lcil.utils.plot")
base_models_module = importlib.import_module("lcil.utils.base_models")
certified_regions_2d = plot_module.certified_regions_2d
lyapunov_cert_regions = plot_module.lyapunov_cert_regions


class TestABCrownInvertedPendulumOnCartIntegration(PlotAssertionsMixin, unittest.TestCase):
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
        sys_cfg = PendulumOnCartConfig()
        k_gain, p_value = riccati_gain_and_value_matrix(sys_cfg)
        lyap_model = RiccatiQuadraticLyapunov(p_value)
        theta_bound = np.pi * 1.4999
        config = cls.LyapunovCertificationConfig(
            state_dim=4,
            cert_bounds=np.array(
                [
                    [-2.0, -10.0, -theta_bound, -10.0],
                    [ 2.0,  10.0,  theta_bound,  10.0]
                ],
                dtype=np.float32,
            ),
            kappa=0.001,
            rho_scaling=1.3,
            bins_per_dim=(3, 4, 6, 7),
            center_refinement_factor=(0.6, 0.6, 0.5, 0.6),
            origin_exclusion=0.1,
            max_scale_steps=5,
            max_bisection_steps=5,
            bisection_tol=0.012,
            cert_method="alpha-crown",
            condition_tolerance=1e-5,
            max_recursion_depth=3,
            batch_size=4096,
        )
        return cls.BisectCertifier(
            policy_model=RiccatiPolicy(k_gain),
            lyap_model=lyap_model,
            dyn_model=NonlinearInvertedPendulumOnCartDynamics(sys_cfg),
            config=config,
            device=th.device("cpu"), # th.device("cuda" if th.cuda.is_available() else "cpu"),
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
                # mpc_datagen may evaluate pairwise projections with reduced tail dimensions.
                # Preserve existing coordinates and pad missing higher dimensions with zeros.
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
            state_dim=4,
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

    def test_inverted_pendulum_on_cart_lqr(self) -> None:
        certifier = self._make_certifier()
        result = certifier.certify(rho_estimate=32.835)
        base = "cartpole_abcrown_integration_dare"
        state_labels = ["$x$", "$v$", r"$\theta$", r"$\dot{\theta}$"]

        self.assertIsInstance(float(result.rho), float)
        self.assertGreaterEqual(result.rho, certifier.config.rho_min)
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
        self.assertGreaterEqual(result.failed_regions.shape[0], 0)
        self.assertGreater(result.certified_regions.shape[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
