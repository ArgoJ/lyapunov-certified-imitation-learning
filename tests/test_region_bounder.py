import unittest

import torch as th

from lcil.certification.lirpa_lyapunov_bounds import LyapunovRegionBounds


class TestLyapunovRegionBounds(unittest.TestCase):
    def test_inside_mask_includes_regions_touching_the_threshold(self) -> None:
        bounds = LyapunovRegionBounds(
            lower=th.tensor([-2.0, 0.5, 1.0], dtype=th.float32),
            upper=th.tensor([0.0, 1.0, 1.5], dtype=th.float32),
        )

        inside = bounds.inside_mask(threshold=1.0)

        self.assertTrue(th.equal(inside, th.tensor([True, True, False], dtype=th.bool)))

    def test_outside_mask_requires_strictly_greater_lower_bound(self) -> None:
        bounds = LyapunovRegionBounds(
            lower=th.tensor([0.5, 1.0, 1.1], dtype=th.float32),
            upper=th.tensor([0.75, 1.5, 2.0], dtype=th.float32),
        )

        outside = bounds.outside_mask(threshold=1.0)

        self.assertTrue(th.equal(outside, th.tensor([False, False, True], dtype=th.bool)))

    def test_sublevel_masks_partition_inside_boundary_and_outside_regions(self) -> None:
        bounds = LyapunovRegionBounds(
            lower=th.tensor([-2.0, 0.5, 1.0, 2.0], dtype=th.float32),
            upper=th.tensor([0.0, 1.0, 1.5, 3.0], dtype=th.float32),
        )

        inside, boundary, outside = bounds.sublevel_masks(threshold=1.0)

        self.assertTrue(th.equal(inside, th.tensor([True, True, False, False], dtype=th.bool)))
        self.assertTrue(th.equal(boundary, th.tensor([False, False, True, False], dtype=th.bool)))
        self.assertTrue(th.equal(outside, th.tensor([False, False, False, True], dtype=th.bool)))
        self.assertTrue(th.equal(inside | boundary | outside, th.ones_like(inside)))
        self.assertFalse(bool(th.any(inside & boundary).item()))
        self.assertFalse(bool(th.any(boundary & outside).item()))
        self.assertFalse(bool(th.any(inside & outside).item()))

    def test_masks_handle_empty_bounds(self) -> None:
        empty = th.empty((0,), dtype=th.float32)
        bounds = LyapunovRegionBounds(lower=empty, upper=empty)

        inside, boundary, outside = bounds.sublevel_masks(threshold=1.0)

        self.assertEqual(inside.numel(), 0)
        self.assertEqual(boundary.numel(), 0)
        self.assertEqual(outside.numel(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)