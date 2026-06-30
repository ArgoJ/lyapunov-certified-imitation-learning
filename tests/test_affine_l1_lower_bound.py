import itertools
import unittest

import torch as th

from lcil.certification.lirpa_lyapunov_bounds import (
    LiRPALyapunovRegionBounds,
    affine_l1_lower_bound,
)
from lcil.lyapunov_learning.models import NeuralLyapunovCandidate
from lcil.utils.base_models import MLP


def _brute_force_l1_min(
    region: th.Tensor,
    pd_matrix: th.Tensor,
    x_star: th.Tensor,
    samples_per_dim: int = 9,
) -> float:
    """Grid-evaluate min of ||M (x - x*)||_1 over a single box (corners + grid)."""
    lb = region[0]
    ub = region[1]
    axes = [th.linspace(float(lb[i]), float(ub[i]), samples_per_dim) for i in range(lb.numel())]
    best = float("inf")
    for combo in itertools.product(*axes):
        x = th.tensor(combo, dtype=pd_matrix.dtype)
        delta = x - x_star
        value = (pd_matrix @ delta).abs().sum().item()
        best = min(best, value)
    return best


class TestAffineL1LowerBound(unittest.TestCase):
    def setUp(self) -> None:
        th.manual_seed(0)
        self.nx = 4
        # A generic symmetric positive-definite M = eps I + R^T R.
        r = th.randn(self.nx, self.nx)
        self.pd_matrix = 1e-3 * th.eye(self.nx) + r.transpose(0, 1) @ r
        self.x_star = th.zeros(self.nx)

    def _random_regions(self, n: int) -> th.Tensor:
        centers = th.randn(n, self.nx) * 1.5
        half_widths = th.rand(n, self.nx) * 2.0 + 0.05
        lb = centers - half_widths
        ub = centers + half_widths
        return th.stack((lb, ub), dim=1)

    def test_lower_bound_is_sound_against_grid_minimum(self) -> None:
        regions = self._random_regions(40)
        analytic = affine_l1_lower_bound(regions, self.pd_matrix, self.x_star)

        for idx in range(len(regions)):
            grid_min = _brute_force_l1_min(regions[idx], self.pd_matrix, self.x_star)
            # Soundness: the analytic bound must never exceed the true minimum.
            self.assertLessEqual(
                float(analytic[idx].item()),
                grid_min + 1e-4,
                msg=f"Analytic lower bound exceeded grid minimum at region {idx}.",
            )

    def test_lower_bound_is_nonnegative(self) -> None:
        regions = self._random_regions(20)
        analytic = affine_l1_lower_bound(regions, self.pd_matrix, self.x_star)
        self.assertTrue(bool((analytic >= -1e-6).all().item()))

    def test_region_around_origin_has_zero_bound(self) -> None:
        # A box straddling x* in every coordinate must yield a zero lower bound.
        lb = -th.ones(self.nx)
        ub = th.ones(self.nx)
        region = th.stack((lb, ub), dim=0).unsqueeze(0)
        analytic = affine_l1_lower_bound(region, self.pd_matrix, self.x_star)
        self.assertAlmostEqual(float(analytic.item()), 0.0, places=5)

    def test_far_region_has_positive_bound(self) -> None:
        lb = th.tensor([5.0, 5.0, 5.0, 5.0])
        ub = th.tensor([6.0, 6.0, 6.0, 6.0])
        region = th.stack((lb, ub), dim=0).unsqueeze(0)
        analytic = affine_l1_lower_bound(region, self.pd_matrix, self.x_star)
        self.assertGreater(float(analytic.item()), 0.0)

    def test_diagonal_matrix_matches_closed_form(self) -> None:
        # With a diagonal M the per-row min decouples exactly per coordinate.
        diag = th.tensor([2.0, 0.5, 1.0, 3.0])
        m = th.diag(diag)
        lb = th.tensor([1.0, -2.0, 0.5, -4.0])
        ub = th.tensor([3.0, -1.0, 2.0, -2.0])
        region = th.stack((lb, ub), dim=0).unsqueeze(0)
        analytic = affine_l1_lower_bound(region, m, th.zeros(4))

        expected = 0.0
        for i in range(4):
            lo = float(diag[i]) * float(lb[i])
            hi = float(diag[i]) * float(ub[i])
            interval_lo, interval_hi = min(lo, hi), max(lo, hi)
            if interval_lo <= 0.0 <= interval_hi:
                expected += 0.0
            else:
                expected += min(abs(interval_lo), abs(interval_hi))
        self.assertAlmostEqual(float(analytic.item()), expected, places=5)


class TestAffineL1Integration(unittest.TestCase):
    def setUp(self) -> None:
        th.manual_seed(1)
        self.nx = 4
        feature_net = MLP(layer_dims=[self.nx, 16, 1], activations=["tanh", "identity"])
        self.lyap_model = NeuralLyapunovCandidate(
            feature_net=feature_net,
            state_dim=self.nx,
            eps=1e-3,
        ).eval()

    def test_extracts_term_and_tightens_outside_region(self) -> None:
        bounder = LiRPALyapunovRegionBounds(
            lyap_model=self.lyap_model,
            state_dim=self.nx,
            batch_size=8,
            default_bound_method="crown",
            use_affine_l1_lower_bound=True,
        )
        self.assertIsNotNone(bounder._affine_l1_term)

        # A region far from the origin: the analytic L1 term should give a
        # strong positive lower bound enabling outside-sublevel pruning.
        lb = th.full((self.nx,), 4.0)
        ub = th.full((self.nx,), 5.0)
        region = th.stack((lb, ub), dim=0).unsqueeze(0)

        bounds = bounder.compute_bounds_for_regions(region)

        # Lower bound must remain sound (<= true V at any sampled interior point).
        center = 0.5 * (lb + ub).unsqueeze(0)
        with th.no_grad():
            v_center = float(self.lyap_model(center).item())
        self.assertLessEqual(float(bounds.lower.item()), v_center + 1e-4)
        self.assertGreater(float(bounds.lower.item()), 0.0)

    def test_lower_bound_never_exceeds_true_v_on_grid(self) -> None:
        bounder = LiRPALyapunovRegionBounds(
            lyap_model=self.lyap_model,
            state_dim=self.nx,
            batch_size=8,
            default_bound_method="crown",
            use_affine_l1_lower_bound=True,
        )

        th.manual_seed(2)
        centers = th.randn(10, self.nx)
        half = th.rand(10, self.nx) + 0.1
        regions = th.stack((centers - half, centers + half), dim=1)
        bounds = bounder.compute_bounds_for_regions(regions)

        for idx in range(len(regions)):
            lb = regions[idx, 0]
            ub = regions[idx, 1]
            grid = th.stack(
                th.meshgrid(
                    *[th.linspace(float(lb[i]), float(ub[i]), 4) for i in range(self.nx)],
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, self.nx)
            with th.no_grad():
                v_min = float(self.lyap_model(grid).min().item())
            self.assertLessEqual(
                float(bounds.lower[idx].item()),
                v_min + 1e-3,
                msg=f"Tightened lower bound exceeded sampled V minimum at region {idx}.",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
