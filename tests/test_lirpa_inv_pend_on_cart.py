import importlib
import os
import sys
import unittest

import numpy as np
import torch as th
import torch.nn as nn
from plot_assertions_mixin import PlotAssertionsMixin

from shared_inv_pend_on_cart import (
    PendulumOnCartConfig,
    riccati_gain_and_value_matrix,
    RiccatiPolicy,
    NonlinearInvertedPendulumOnCartDynamics,
    RiccatiQuadraticLyapunov
)

plot_module = importlib.import_module("lcil.utils.plot")
base_models_module = importlib.import_module("lcil.utils.base_models")
certified_regions_2d = plot_module.certified_regions_2d
lyapunov = plot_module.lyapunov


class TestLiRPAInvertedPendulumOnCartIntegration(PlotAssertionsMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("LCIL_RUN_LIRPA_INTEGRATION", "0") != "1":
            raise unittest.SkipTest(
                "Set LCIL_RUN_LIRPA_INTEGRATION=1 to run real auto_LiRPA integration tests."
            )

        try:
            auto_LiRPA = importlib.import_module("auto_LiRPA")
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise unittest.SkipTest(f"auto_LiRPA is not importable: {exc}") from exc

        required_symbols = [
            "BoundedModule",
            "BoundedTensor",
            "PerturbationLpNorm",
        ]
        missing = [name for name in required_symbols if not hasattr(auto_LiRPA, name)]
        if missing:
            raise unittest.SkipTest(
                f"auto_LiRPA is missing required symbols for wrapper integration: {missing}"
            )

        try:
            sys.modules.pop("lcil.certification.lirpa_wrapper", None)
            lirpa_wrapper = importlib.import_module("lcil.certification.lirpa_wrapper")
            cls.LiRPACertifier = lirpa_wrapper.LiRPACertifier
            cls.LyapunovCertificationConfig = importlib.import_module(
                "lcil.certification.config"
            ).LyapunovCertificationConfig
            cls.certifier_base = importlib.import_module("lcil.certification.certifier_base")
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise unittest.SkipTest(f"Could not import certifier modules: {exc}") from exc

    @classmethod
    def _make_certifier(cls):
        sys_cfg = PendulumOnCartConfig()
        k_gain, p_value = riccati_gain_and_value_matrix(sys_cfg)
        lyap_model = RiccatiQuadraticLyapunov(p_value)
        theta_bound = np.pi * 0.999 
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
            invariance_weight=1.0,
            rho_scaling=1.1,
            bins_per_dim=(12, 12, 12, 12),
            center_refinement_factor=(0.3, 0.3, 0.2, 0.3),
            origin_exclusion=0.15,
            max_scale_steps=20,
            max_bisection_steps=10,
            cert_method="alpha-crown",
            condition_tolerance=1e-5,
            suppress_native_output=True,
            batch_size=4096,
            use_ibp_filter=True,
            max_recursion_depth=10,
        )
        return cls.LiRPACertifier(
            policy_model=RiccatiPolicy(k_gain),
            lyap_model=lyap_model,
            dyn_model=NonlinearInvertedPendulumOnCartDynamics(sys_cfg),
            config=config,
            device=th.device("cuda" if th.cuda.is_available() else "cpu"),
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
        plot_state_indices = (2, 3)
        state_labels = ["theta", "theta_dot"]

        certified_regions_2d_view = self._project_regions_to_2d(
            certified_regions,
            state_indices=plot_state_indices,
        )
        uncertified_regions_2d_view = self._project_regions_to_2d(
            uncertified_regions,
            state_indices=plot_state_indices,
        )

        self._assert_plot_written(
            plot_fn=certified_regions_2d,
            stem=stem,
            plot_kwargs={
                "certified_regions": certified_regions_2d_view,
                "uncertified_regions": uncertified_regions_2d_view,
                "state_labels": state_labels,
                "state_indices": plot_state_indices
            },
        )

    @staticmethod
    def _to_numpy_lyapunov(
        lyap_model: nn.Module,
        state_dim: int,
        plot_state_indices: tuple[int, int],
    ):
        try:
            device = next(lyap_model.parameters()).device
        except StopIteration:
            device = next(lyap_model.buffers()).device

        def _lyapunov_func(x: np.ndarray) -> np.ndarray | float:
            x_array = np.asarray(x, dtype=np.float32)

            if x_array.shape[-1] == len(plot_state_indices):
                x_lifted = np.zeros((*x_array.shape[:-1], state_dim), dtype=np.float32, device=device)
                x_lifted[..., plot_state_indices[0]] = x_array[..., 0]
                x_lifted[..., plot_state_indices[1]] = x_array[..., 1]
            elif x_array.shape[-1] == state_dim:
                x_lifted = x_array
            else:
                raise ValueError(
                    f"Lyapunov input has invalid shape {x_array.shape}; expected last dim "
                    f"{len(plot_state_indices)} or {state_dim}."
                )

            x_tensor = th.as_tensor(x_lifted, dtype=th.float32, device=device)
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
        plot_state_indices = (2, 3)
        state_labels = ["theta", "theta_dot"]

        lyap_func = self._to_numpy_lyapunov(
            lyap_model=lyap_model,
            state_dim=4,
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

        self._assert_plot_written(
            plot_fn=lyapunov,
            stem=stem,
            plot_kwargs={
                "dataset": None,
                "lyapunov_func": lyap_func,
                "state_indices": list(plot_state_indices),
                "state_labels": state_labels,
                "certified_regions": certified_regions_2d_view,
                "uncertified_regions": uncertified_regions_2d_view,
            },
        )

    def test_inverted_pendulum_on_cart_lqr_lirpa(self) -> None:
        certifier = self._make_certifier()
        rho_certified, result = certifier.certify(rho_estimate=20.0)

        self.assertIsInstance(float(rho_certified), float)
        self.assertGreaterEqual(rho_certified, certifier.config.rho_min)
        self._assert_region_plot_written(
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="inverted_pendulum_on_cart_lirpa_regions_integration",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="inverted_pendulum_on_cart_lirpa_lyapunov_integration",
        )
        self.assertGreater(result.failed_regions.shape[0], 0)
        self.assertGreater(result.certified_regions.shape[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)