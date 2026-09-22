import unittest
import torch as th

from lcil.lyapunov_learning.buffer import (
    BoundaryStateBuffer,
    CEGISBuffer,
    get_spatial_diversity_indices,
)


class TestBufferSpatialDiversity(unittest.TestCase):
    def test_get_spatial_diversity_indices_empty_and_single(self) -> None:
        # Empty tensor
        empty_states = th.empty((0, 2))
        empty_vals = th.empty((0,))
        idx = get_spatial_diversity_indices(empty_states, empty_vals, filter_eps=0.1)
        self.assertEqual(idx.numel(), 0)

        # Single element
        single_states = th.tensor([[1.0, 2.0]])
        single_vals = th.tensor([0.5])
        idx = get_spatial_diversity_indices(single_states, single_vals, filter_eps=0.1)
        self.assertEqual(idx.tolist(), [0])

    def test_get_spatial_diversity_indices_filter_eps_zero(self) -> None:
        states = th.tensor([[0.0], [1.0], [2.0], [3.0]])
        values = th.tensor([10.0, 40.0, 20.0, 30.0])

        # Descending: indices ordered by value: 1 (40), 3 (30), 2 (20), 0 (10)
        idx = get_spatial_diversity_indices(states, values, filter_eps=0.0, descending=True)
        self.assertEqual(idx.tolist(), [1, 3, 2, 0])

        # Truncation with max_elements
        idx_limited = get_spatial_diversity_indices(states, values, filter_eps=0.0, descending=True, max_elements=2)
        self.assertEqual(idx_limited.tolist(), [1, 3])

    def test_get_spatial_diversity_indices_suppresses_close_neighbors(self) -> None:
        # Points: [0.0, 0.0] (val 10), [0.02, 0.02] (val 5 - close to first), [1.0, 1.0] (val 8 - far)
        states = th.tensor([
            [0.0, 0.0],
            [0.02, 0.02],
            [1.0, 1.0],
        ])
        values = th.tensor([10.0, 5.0, 8.0])

        # filter_eps = 0.1 -> [0.0, 0.0] and [0.02, 0.02] fall into the same cell.
        # [0.0, 0.0] has higher value (10 > 5), so it should be kept and [0.02, 0.02] suppressed.
        idx = get_spatial_diversity_indices(states, values, filter_eps=0.1, descending=True)
        self.assertEqual(len(idx), 2)
        # Should keep index 0 (val 10) and index 2 (val 8)
        self.assertEqual(set(idx.tolist()), {0, 2})

    def test_get_spatial_diversity_indices_ascending(self) -> None:
        # Descending = False: lowest value wins within a voxel cell
        states = th.tensor([
            [0.0, 0.0],
            [0.02, 0.02],
            [1.0, 1.0],
        ])
        values = th.tensor([10.0, 5.0, 8.0])

        idx = get_spatial_diversity_indices(states, values, filter_eps=0.1, descending=False)
        self.assertEqual(len(idx), 2)
        # Index 1 has value 5.0, lower than index 0 (10.0), so 1 is kept over 0
        self.assertEqual(set(idx.tolist()), {1, 2})

    def test_get_spatial_diversity_indices_cuda(self) -> None:
        if not th.cuda.is_available():
            self.skipTest("CUDA not available")

        device = th.device("cuda")
        states = th.randn(100, 3, device=device)
        values = th.randn(100, device=device)

        idx = get_spatial_diversity_indices(states, values, filter_eps=0.2, descending=True, max_elements=20)
        self.assertEqual(idx.device.type, "cuda")
        self.assertLessEqual(len(idx), 20)

    def test_boundary_state_buffer_with_diversity(self) -> None:
        buf = BoundaryStateBuffer(
            state_dim=2,
            max_size=5,
            filter_eps=0.2,
            device=th.device("cpu"),
        )
        # Add clusters of points
        cluster1 = th.tensor([[0.0, 0.0], [0.05, 0.05], [0.08, 0.08]])
        cluster2 = th.tensor([[2.0, 2.0], [2.05, 2.05]])
        buf.update(th.cat([cluster1, cluster2], dim=0), value_fn=lambda x: x.norm(dim=1))

        # Only diverse points should be kept
        self.assertLessEqual(len(buf), 5)

    def test_get_spatial_diversity_indices_normalized_bounds(self) -> None:
        # Dimension 0 width is 100.0, Dimension 1 width is 0.01
        lb = th.tensor([0.0, 0.0])
        ub = th.tensor([100.0, 0.01])

        # Point 0 and Point 1 are at the same x0=50.0, but differ by 0.008 in x1 (80% of dim 1)
        states = th.tensor([
            [50.0, 0.001],
            [50.0, 0.009],
        ])
        values = th.tensor([10.0, 5.0])

        # With normalized filter_eps=0.1 (10% of box width in each dimension):
        # 0.008 is 80% > 10%, so they must NOT suppress each other!
        idx = get_spatial_diversity_indices(states, values, filter_eps=0.1, lb=lb, ub=ub)
        self.assertEqual(len(idx), 2)

    def test_cegis_buffer_filters_only_new_cexs(self) -> None:
        lb = th.tensor([0.0])
        ub = th.tensor([1.0])
        buf = CEGISBuffer(
            initial_states=th.tensor([[0.5]]),
            state_buffer_limit=10,
            cex_buffer_limit=5,
            lb=lb,
            ub=ub,
            filter_eps=0.2,  # 20% bin size: [0.0, 0.2), [0.2, 0.4), etc.
            max_cex_age=5,
            device=th.device("cpu"),
        )

        # Batch 1: two new points in the SAME bin [0.0, 0.2): 0.05 (val 2) and 0.08 (val 10)
        # Spatial filtering among new CEXs keeps only 0.08 (higher violation score)
        buf.register_cex(
            th.tensor([[0.05], [0.08]]),
            objective=lambda x: -x,  # score = x
        )
        self.assertEqual(buf.cex_count, 1)
        self.assertAlmostEqual(buf.cexs[0, 0].item(), 0.08, places=5)

        # Batch 2: new point at 0.06 (also in bin [0.0, 0.2)).
        # Since ONLY new CEXs are filtered, the existing point 0.08 is NOT removed!
        buf.register_cex(
            th.tensor([[0.06]]),
            objective=lambda x: -x,
        )
        self.assertEqual(buf.cex_count, 2)
        retained_vals = {round(x, 2) for x in buf.cexs.flatten().tolist()}
        self.assertEqual(retained_vals, {0.08, 0.06})

    def test_dynamic_state_buffer_keeps_most_violating_counterexamples(self) -> None:
        initial_states = th.zeros((4, 1), dtype=th.float32)
        state_buffer = CEGISBuffer(
            lb=th.tensor([-10.0]),
            ub=th.tensor([10.0]),
            initial_states=initial_states,
            state_buffer_limit=16,
            cex_buffer_limit=3,
            filter_eps=0.0,
            device=th.device("cpu"),
        )

        state_buffer.register_cex(
            th.tensor([[0.2], [0.4]], dtype=th.float32),
            objective=lambda x: -x,
        )
        state_buffer.register_cex(
            th.tensor([[0.1], [0.9]], dtype=th.float32),
            objective=lambda x: -x,
        )

        retained = state_buffer.cexs.flatten()
        expected = th.tensor([0.9, 0.4, 0.2], dtype=th.float32)

        self.assertEqual(state_buffer.state_count, 4)
        self.assertEqual(state_buffer.cex_count, 3)
        self.assertEqual(len(state_buffer), 7)
        self.assertTrue(th.allclose(retained, expected))

    def test_dynamic_state_buffer_sample_returns_requested_batch_size(self) -> None:
        state_buffer = CEGISBuffer(
            lb=th.tensor([-10.0]),
            ub=th.tensor([10.0]),
            initial_states=th.tensor([[1.0], [2.0]], dtype=th.float32),
            state_buffer_limit=4,
            cex_buffer_limit=3,
            device=th.device("cpu"),
        )

        batch = state_buffer.sample(batch_size=5)

        self.assertEqual(batch.shape, (5, 1))
        self.assertTrue(th.all((batch == 1.0) | (batch == 2.0)).item())

    def test_dynamic_state_buffer_sample_uses_regular_and_cex_pools_separately(self) -> None:
        state_buffer = CEGISBuffer(
            lb=th.tensor([-10.0]),
            ub=th.tensor([10.0]),
            initial_states=th.tensor([[1.0], [2.0]], dtype=th.float32),
            state_buffer_limit=8,
            cex_buffer_limit=3,
            device=th.device("cpu"),
        )
        state_buffer.register_cex(th.tensor([[10.0]], dtype=th.float32), objective=lambda x: -x)

        batch = state_buffer.sample(batch_size=4, cex_fraction=0.5)
        batch_values = batch.flatten()

        self.assertEqual(batch.shape, (4, 1))
        self.assertEqual(int((batch_values == 10.0).sum().item()), 1)
        self.assertTrue(th.all((batch_values != 10.0) <= ((batch_values == 1.0) | (batch_values == 2.0))).item())

    def test_dynamic_state_buffer_rejects_empty_initial_states(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial_states cannot be empty"):
            CEGISBuffer(
                lb=th.tensor([-10.0]),
                ub=th.tensor([10.0]),
                initial_states=th.empty((0, 1), dtype=th.float32),
                state_buffer_limit=4,
                cex_buffer_limit=3,
                device=th.device("cpu"),
            )

    def test_dynamic_state_buffer_sample_clamps_out_of_range_cex_fraction(self) -> None:
        state_buffer = CEGISBuffer(
            lb=th.tensor([-10.0]),
            ub=th.tensor([10.0]),
            initial_states=th.tensor([[1.0], [2.0]], dtype=th.float32),
            state_buffer_limit=4,
            cex_buffer_limit=3,
            device=th.device("cpu"),
        )
        state_buffer.register_cex(th.tensor([[10.0]], dtype=th.float32), objective=lambda x: -x)

        batch = state_buffer.sample(batch_size=4, cex_fraction=2.0)
        batch_values = batch.flatten()

        self.assertEqual(batch.shape, (4, 1))
        self.assertEqual(int((batch_values == 10.0).sum().item()), 1)
        self.assertTrue(th.all((batch_values == 10.0) | (batch_values == 1.0) | (batch_values == 2.0)).item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
