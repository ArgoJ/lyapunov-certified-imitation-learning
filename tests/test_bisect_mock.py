import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn
from plot_assertions_mixin import PlotAssertionsMixin

from certification_mock_common import (
    BisectCertifier,
    CertificationMockedABCrownTestCase,
    LyapunovRegionBounds,
    _MockVerificationResult,
    _StatusAwareMockRegionCertifier,
    _complete_candidate_partition,
)
from lcil.certification.abcrown_region_certifier import EarlyExitLevel
from lcil.certification.bisect_certifier import RegionCertificationResult
from lcil.utils.lcil_plt import certified_regions_2d, lyapunov_cert_regions
from shared_utils import (
    _DirectionalScaleDynamics,
    _IdentityDynamics,
    _NegativeQuadraticLyapunov,
    _QuadraticLyapunov,
    _ZeroDynamics,
    _ZeroPolicy,
)


class TestBisectCertifier(PlotAssertionsMixin, CertificationMockedABCrownTestCase):
    """Unit and mock tests specifically verifying the BisectCertifier component.

    Tests full bisection search (find_max_rho / certify), delegation of
    is_rho_certified with early exit, and persistence of RegionCertificationResult.
    """

    @classmethod
    def _make_certifier(
        cls,
        lyap_model: nn.Module,
        *,
        dyn_model: nn.Module | None = None,
        kappa: float = 0.1,
        rho_min: float = 1e-6,
        origin_exclusion: float | tuple[float, ...] = 0.1,
    ) -> BisectCertifier:
        config = cls.make_config(
            state_dim=3,
            cert_bounds=np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=np.float32),
            kappa=kappa,
            rho_min=rho_min,
            bins_per_dim=4,
            center_refinement_factor=0.7,
            origin_exclusion=origin_exclusion,
            max_scale_steps=6,
            max_bisection_steps=6,
            lirpa_method="alpha-crown",
            condition_tolerance=1e-6,
            max_recursion_depth=3,
        )
        return cls.make_bisect_certifier(
            policy_model=_ZeroPolicy(),
            lyap_model=lyap_model,
            dyn_model=_ZeroDynamics() if dyn_model is None else dyn_model,
            config=config,
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

    def test_quadratic_lyapunov_with_identity_dynamics_certifies_all_regions(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_ZeroDynamics(),
            kappa=1e-6,
        )
        result = certifier.certify(rho_estimate=1.0, collect_details_on_failed=True)

        self.assertTrue(result.global_success)
        self.assertTrue(result.partial_success)
        self.assertGreaterEqual(result.rho, 1.0)
        self.assertEqual(result.uncertified_regions.shape[0], 0)
        self.assertGreater(result.certified_sublevel_regions.shape[0], 0)
        self._assert_region_plot_written(
            certification_result=result,
            stem="quadratic_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem="quadratic_lyapunov",
        )

    def test_negative_quadratic_produces_counterexamples(self) -> None:
        certifier = self._make_certifier(_NegativeQuadraticLyapunov())
        result = certifier.certify(rho_estimate=1.0, collect_details_on_failed=True)

        self.assertFalse(result.global_success)
        self.assertFalse(result.partial_success)
        self.assertAlmostEqual(result.rho, certifier.config.rho_min, places=6)
        self.assertEqual(result.certified_sublevel_regions.shape[0], 0)
        self.assertGreaterEqual(result.uncertified_regions.shape[0], 0)
        self.assertGreaterEqual(result.outside_sublevel_regions.shape[0], 0)
        self._assert_region_plot_written(
            certification_result=result,
            stem="negative_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem="negative_lyapunov",
        )

    def test_mixed_lyapunov_has_safe_and_unsafe_regions(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_DirectionalScaleDynamics(base_scale=0.8, axis_gain=0.4),
            kappa=1e-6,
            rho_min=0.9,
        )
        result = certifier.certify(rho_estimate=1.0, collect_details_on_failed=True)

        self.assertFalse(result.global_success)
        self.assertTrue(result.partial_success)
        self.assertGreater(result.certified_sublevel_regions.shape[0], 0)
        self.assertGreater(result.uncertified_regions.shape[0], 0)

        certified_centers = result.certified_sublevel_regions.mean(axis=1)
        failed_centers = result.uncertified_regions.mean(axis=1)
        self.assertTrue(np.any(certified_centers[:, 0] < 0.0))
        self.assertTrue(np.any(failed_centers[:, 0] > 0.0))
        self._assert_region_plot_written(
            certification_result=result,
            stem="mixed_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem="mixed_lyapunov",
        )

    def test_is_rho_certified_stops_immediately_on_direct_counterexample(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=1e-6,
        )
        certifier.regions = th.tensor(
            [
                [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            ],
            dtype=th.float32,
        )
        region_certifier = _StatusAwareMockRegionCertifier(
            [
                _MockVerificationResult(
                    verified=True,
                    counterexample_found=False,
                    status="safe",
                ),
                _MockVerificationResult(
                    verified=False,
                    counterexample_found=True,
                    status="unsafe-pgd",
                ),
            ]
        )

        certifier.region_manager.cache_region_bounds(
            LyapunovRegionBounds(
                lower=th.zeros((len(certifier.regions),), dtype=th.float32),
                upper=th.zeros((len(certifier.regions),), dtype=th.float32),
            ),
            regions=certifier.regions,
            make_current=True,
        )

        with mock.patch.object(
            certifier.region_manager,
            "partition_certification_regions",
            return_value=_complete_candidate_partition(certifier.regions),
        ), mock.patch.object(
            certifier,
            "_get_region_certifier",
            return_value=region_certifier,
        ):
            self.assertFalse(certifier.is_rho_certified(rho=1.0))

        self.assertEqual(region_certifier.calls, 2)

    def test_is_rho_certified_delegates_to_recursive_regions_with_early_exit(self) -> None:
        certifier = self._make_certifier(_QuadraticLyapunov())
        mock_result = mock.MagicMock()
        mock_result.global_success = True

        with mock.patch.object(
            certifier, "_certify_recursive_regions", return_value=mock_result
        ) as mock_cert:
            self.assertTrue(certifier.is_rho_certified(rho=0.75))
            mock_cert.assert_called_once_with(
                rho=0.75, early_exit=EarlyExitLevel.ON_COUNTEREXAMPLE
            )

    def test_region_certification_result_save_and_load_roundtrip(self) -> None:
        regions = np.ones((2, 2, 3), dtype=np.float32)
        result = RegionCertificationResult(
            global_success=True,
            partial_success=True,
            rho=0.42,
            outside_sublevel_regions=regions,
            uncertified_regions=regions[:0],
            certified_sublevel_regions=regions,
            certified_boundary_regions=regions[:1],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "cert_result.npz"
            result.save(file_path)
            loaded = RegionCertificationResult.load(file_path)

            self.assertEqual(loaded.global_success, result.global_success)
            self.assertEqual(loaded.partial_success, result.partial_success)
            self.assertAlmostEqual(loaded.rho, result.rho, places=6)
            np.testing.assert_allclose(
                loaded.certified_sublevel_regions, result.certified_sublevel_regions
            )
            np.testing.assert_allclose(
                loaded.outside_sublevel_regions, result.outside_sublevel_regions
            )
            np.testing.assert_allclose(
                loaded.uncertified_regions, result.uncertified_regions
            )
            np.testing.assert_allclose(
                loaded.certified_boundary_regions, result.certified_boundary_regions
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
