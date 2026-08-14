import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch as th
from shared_utils import (
    _EllipticalLyapunov,
    _NonFiniteOutsideUnitBallLyapunov,
    _NonMonotonicRadialLyapunov,
    _OffsetLyapunov,
    _QuadraticLyapunov,
)

from lcil.certification.metrics import (
    LevelSetEstimate,
    _sample_unit_sphere_directions,
    estimate_level_set_measure,
)



class TestCertificationMetrics(unittest.TestCase):

    def test_sample_unit_sphere_directions_have_unit_norm(self) -> None:
        directions = _sample_unit_sphere_directions(num_states=4, num_directions=128)

        self.assertEqual(directions.shape, (128, 4))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)

    def test_origin_outside_sublevel_set_raises_error(self) -> None: 
        with self.assertRaises(ValueError):
            estimate_level_set_measure(_OffsetLyapunov(), rho=1.0, num_states=2)

    def test_initial_radius_above_max_radius_raises_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial_radius"):
            estimate_level_set_measure(
                _QuadraticLyapunov(),
                rho=1.0,
                num_states=2,
                initial_radius=3.0,
                max_radius=2.0,
            )

    def test_estimate_level_set_area_matches_unit_disk(self) -> None:
        estimate = estimate_level_set_measure(
            _QuadraticLyapunov(),
            rho=1.0,
            num_states=2,
            num_directions=256,
        )

        self.assertEqual(estimate.details.directions.shape, (256, 2))
        np.testing.assert_allclose(estimate.details.radii, 1.0, atol=5e-6)
        self.assertFalse(estimate.details.truncated)
        self.assertAlmostEqual(estimate.measure, np.pi, delta=1e-4)

    def test_estimate_level_set_area_matches_unit_ball(self) -> None:
        estimate = estimate_level_set_measure(
            _QuadraticLyapunov(),
            rho=1.0,
            num_states=3,
            num_directions=512,
        )

        np.testing.assert_allclose(estimate.details.radii, 1.0, atol=5e-6)
        self.assertFalse(estimate.details.truncated)
        self.assertAlmostEqual(estimate.measure, 4.0 * np.pi / 3.0, delta=1e-4)

    def test_estimate_level_set_area_truncation(self) -> None:
        # V(x) <= 100 -> radius should be 10.0
        # artificially set max_radius to 2.0.
        estimate = estimate_level_set_measure(
            _QuadraticLyapunov(),
            rho=100.0,
            num_states=2,
            num_directions=32,
            max_radius=2.0,
        )

        # All rays should be truncated at 2.0, so the area should be pi * 2^2 = 4pi.
        self.assertAlmostEqual(estimate.measure, 4.0 * np.pi, delta=1e-5)
        self.assertTrue(estimate.details.truncated)
        self.assertEqual(estimate.details.truncated_fraction, 1.0)
        np.testing.assert_allclose(estimate.details.radii, 2.0)

    def test_nonfinite_values_are_treated_as_ray_exit(self) -> None:
        estimate = estimate_level_set_measure(
            _NonFiniteOutsideUnitBallLyapunov(),
            rho=4.0,
            num_states=2,
            num_directions=64,
            initial_radius=0.5,
            growth_factor=2.0,
            max_radius=4.0,
        )

        self.assertFalse(estimate.details.truncated)
        np.testing.assert_allclose(estimate.details.radii, 1.0, atol=5e-5)
        self.assertAlmostEqual(estimate.measure, np.pi, delta=1e-4)

    def test_non_monotonic_rays_logs_warning(self) -> None:
        with self.assertLogs("lcil.certification.metrics", level="WARNING"):
            estimate_level_set_measure(
                _NonMonotonicRadialLyapunov(),
                rho=1.1,
                num_states=2,
                num_directions=64,
                initial_radius=0.25,
            )

    def test_estimate_level_set_area_matches_ellipse(self) -> None:
        estimate = estimate_level_set_measure(
            _EllipticalLyapunov(),
            rho=1.0,
            num_states=2,
            num_directions=2048,
        )
        
        # analytical area of an ellipse: pi * a * b
        expected_area = np.pi * 1.0 * 0.5 
        
        self.assertFalse(estimate.details.truncated)
        # Tolerance slightly higher due to Monte Carlo error in asymmetric 
        # shapes being larger than in perfect circles.
        self.assertAlmostEqual(estimate.measure, expected_area, delta=5e-3)

    def test_estimate_level_set_area_monte_carlo(self) -> None:
        # V(x) = x1^2 + x2^2 <= 1.0 -> unit circle area = pi
        estimate = estimate_level_set_measure(
            _QuadraticLyapunov(),
            rho=1.0,
            num_states=2,
            method="monte_carlo",
            bounds=2.0,
            num_samples=200_000,
        )

        self.assertEqual(estimate.method, "monte_carlo")
        self.assertAlmostEqual(estimate.measure, np.pi, delta=5e-2)

    def test_level_set_estimate_json_roundtrip(self) -> None:
        estimate = estimate_level_set_measure(
            _QuadraticLyapunov(),
            rho=1.0,
            num_states=2,
            num_directions=32,
        )

        with TemporaryDirectory() as tmp_dir:
            output_path = estimate.save(tmp_dir)
            reloaded = LevelSetEstimate.load(tmp_dir)

        self.assertEqual(output_path.name, "level_set_estimate.json")
        self.assertEqual(reloaded.rho, estimate.rho)
        self.assertEqual(reloaded.num_states, estimate.num_states)
        self.assertEqual(reloaded.details.num_directions, estimate.details.num_directions)
        self.assertEqual(reloaded.measure, estimate.measure)
        self.assertEqual(reloaded.details.unit_sphere_surface_area, estimate.details.unit_sphere_surface_area)
        self.assertEqual(reloaded.details.max_radius, estimate.details.max_radius)
        np.testing.assert_allclose(reloaded.details.directions, estimate.details.directions)
        np.testing.assert_allclose(reloaded.details.radii, estimate.details.radii)
        np.testing.assert_array_equal(reloaded.details.truncated_mask, estimate.details.truncated_mask)

    def test_monte_carlo_level_set_estimate_json_roundtrip(self) -> None:
        estimate = estimate_level_set_measure(
            _QuadraticLyapunov(),
            rho=1.0,
            num_states=2,
            method="monte_carlo",
            bounds=2.0,
            num_samples=1000,
        )

        with TemporaryDirectory() as tmp_dir:
            output_path = estimate.save(tmp_dir)
            reloaded = LevelSetEstimate.load(tmp_dir)

        self.assertEqual(output_path.name, "level_set_estimate.json")
        self.assertEqual(reloaded.rho, estimate.rho)
        self.assertEqual(reloaded.num_states, estimate.num_states)
        self.assertEqual(reloaded.details.num_samples, estimate.details.num_samples)
        self.assertEqual(reloaded.measure, estimate.measure)
        self.assertEqual(reloaded.details.box_volume, estimate.details.box_volume)
        self.assertEqual(reloaded.details.inside_fraction, estimate.details.inside_fraction)
        np.testing.assert_allclose(reloaded.details.bounds, estimate.details.bounds)


if __name__ == "__main__":
    unittest.main(verbosity=2)