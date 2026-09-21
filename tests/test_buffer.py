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
        self.assertGreaterEqual(len(buf), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
