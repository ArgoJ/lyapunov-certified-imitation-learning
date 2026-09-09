import unittest
from lcil.utils.search_utils import (
    bisect_rho,
    scale_rho_down,
    scale_rho_up,
    search_and_bisect_value,
)


class TestSearchUtils(unittest.TestCase):
    """Unit tests for bisection and scaling utilities in search_utils."""

    def test_scale_rho_up_advances_when_certified(self) -> None:
        eval_fn = lambda x: x <= 5.0
        stop, lo, up = scale_rho_up(lo=1.0, up=1.0, eval_fn=eval_fn, scaling=2.0)
        self.assertFalse(stop)
        self.assertEqual(lo, 2.0)
        self.assertEqual(up, 2.0)

    def test_scale_rho_up_stops_when_not_certified(self) -> None:
        eval_fn = lambda x: x <= 3.0
        stop, lo, up = scale_rho_up(lo=2.0, up=2.0, eval_fn=eval_fn, scaling=2.0)
        self.assertTrue(stop)
        self.assertEqual(lo, 2.0)
        self.assertEqual(up, 4.0)

    def test_scale_rho_down_advances_when_not_certified_above_min(self) -> None:
        eval_fn = lambda x: x <= 0.2
        stop, lo, up = scale_rho_down(
            lo=None, up=1.0, eval_fn=eval_fn, min_val=0.1, scaling=2.0
        )
        self.assertFalse(stop)
        self.assertIsNone(lo)
        self.assertEqual(up, 0.5)

    def test_scale_rho_down_stops_when_certified(self) -> None:
        eval_fn = lambda x: x <= 0.6
        stop, lo, up = scale_rho_down(
            lo=None, up=1.0, eval_fn=eval_fn, min_val=0.1, scaling=2.0
        )
        self.assertTrue(stop)
        self.assertEqual(lo, 0.5)
        self.assertEqual(up, 1.0)

    def test_scale_rho_down_stops_at_min_val_when_uncertified(self) -> None:
        eval_fn = lambda x: False
        stop, lo, up = scale_rho_down(
            lo=None, up=0.15, eval_fn=eval_fn, min_val=0.1, scaling=2.0
        )
        self.assertTrue(stop)
        self.assertIsNone(lo)
        self.assertEqual(up, 0.1)

    def test_bisect_rho_updates_lower_bound_when_mid_certified(self) -> None:
        eval_fn = lambda x: x <= 1.6
        stop, lo, up = bisect_rho(lo=1.0, up=2.0, eval_fn=eval_fn, bisection_tol=0.01)
        self.assertFalse(stop)
        self.assertEqual(lo, 1.5)
        self.assertEqual(up, 2.0)

    def test_bisect_rho_updates_upper_bound_when_mid_uncertified(self) -> None:
        eval_fn = lambda x: x <= 1.4
        stop, lo, up = bisect_rho(lo=1.0, up=2.0, eval_fn=eval_fn, bisection_tol=0.01)
        self.assertFalse(stop)
        self.assertEqual(lo, 1.0)
        self.assertEqual(up, 1.5)

    def test_bisect_rho_stops_when_tolerance_reached(self) -> None:
        eval_fn = lambda x: x <= 1.5
        stop, lo, up = bisect_rho(lo=1.0, up=1.05, eval_fn=eval_fn, bisection_tol=0.1)
        self.assertTrue(stop)
        self.assertEqual(lo, 1.025)
        self.assertEqual(up, 1.05)

    def test_search_and_bisect_value_scale_up_and_bisect(self) -> None:
        target_max = 3.65
        eval_fn = lambda x: x <= target_max
        tol = 1e-4

        result = search_and_bisect_value(
            initial_estimate=1.0,
            eval_fn=eval_fn,
            min_val=0.01,
            scaling_factor=2.0,
            bisection_tol=tol,
            max_scale_steps=10,
            max_bisection_steps=20,
        )

        self.assertTrue(eval_fn(result))
        self.assertAlmostEqual(result, target_max, delta=tol)

    def test_search_and_bisect_value_scale_down_and_bisect(self) -> None:
        target_max = 0.35
        eval_fn = lambda x: x <= target_max
        tol = 1e-4

        result = search_and_bisect_value(
            initial_estimate=2.0,
            eval_fn=eval_fn,
            min_val=0.01,
            scaling_factor=2.0,
            bisection_tol=tol,
            max_scale_steps=10,
            max_bisection_steps=20,
        )

        self.assertTrue(eval_fn(result))
        self.assertAlmostEqual(result, target_max, delta=tol)

    def test_search_and_bisect_value_returns_zero_when_min_val_not_certified(self) -> None:
        eval_fn = lambda x: False

        result = search_and_bisect_value(
            initial_estimate=1.0,
            eval_fn=eval_fn,
            min_val=0.05,
            scaling_factor=2.0,
            bisection_tol=1e-3,
            max_scale_steps=5,
            max_bisection_steps=10,
        )

        self.assertEqual(result, 0.0)

    def test_search_and_bisect_value_handles_initial_estimate_below_min_val(self) -> None:
        target_max = 0.08
        eval_fn = lambda x: x <= target_max
        tol = 1e-4

        result = search_and_bisect_value(
            initial_estimate=0.005,
            eval_fn=eval_fn,
            min_val=0.01,
            scaling_factor=2.0,
            bisection_tol=tol,
            max_scale_steps=10,
            max_bisection_steps=20,
        )

        self.assertTrue(eval_fn(result))
        self.assertAlmostEqual(result, target_max, delta=tol)

    def test_search_and_bisect_value_rejects_non_positive_initial_estimate(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial_estimate must be positive"):
            search_and_bisect_value(
                initial_estimate=0.0,
                eval_fn=lambda x: True,
                min_val=0.01,
                scaling_factor=2.0,
                bisection_tol=1e-3,
                max_scale_steps=5,
                max_bisection_steps=5,
            )

        with self.assertRaisesRegex(ValueError, "initial_estimate must be positive"):
            search_and_bisect_value(
                initial_estimate=-1.5,
                eval_fn=lambda x: True,
                min_val=0.01,
                scaling_factor=2.0,
                bisection_tol=1e-3,
                max_scale_steps=5,
                max_bisection_steps=5,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
