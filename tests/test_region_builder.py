import unittest

import numpy as np
import plotly.graph_objects as go
import torch as th

from plot_assertions_mixin import PlotAssertionsMixin

from lcil.utils.region_builder import RegionBuilder


def _regions_to_numpy(regions: th.Tensor | np.ndarray) -> np.ndarray:
    return th.as_tensor(regions, dtype=th.float32).detach().cpu().numpy()


def _region_key(region: th.Tensor | np.ndarray, decimals: int = 6) -> tuple[float, ...]:
    region_np = np.asarray(region, dtype=np.float64)
    return tuple(np.round(region_np.reshape(-1), decimals=decimals).tolist())


def plot_region_groups_2d(
    region_groups: list[tuple[str, th.Tensor | np.ndarray, str]],
    html_path: str,
    title: str = "RegionBuilder regions",
) -> None:
    fig = go.Figure()

    for name, regions, color in region_groups:
        regs_np = _regions_to_numpy(regions)
        if regs_np.size == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": color, "width": 3},
                name=name,
            )
        )
        for region in regs_np:
            x0, y0 = region[0]
            x1, y1 = region[1]
            fig.add_shape(
                type="rect",
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                line={"color": color, "width": 2},
                fillcolor=color,
                opacity=0.25,
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title=r"$x_0$",
        yaxis_title=r"$x_1$",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
    fig.write_html(html_path)


class TestRegionBuilder(PlotAssertionsMixin, unittest.TestCase):
    def _make_builder(
        self,
        *,
        bounds: list[list[float]] | np.ndarray | None = None,
        bins_per_dim: int | tuple[int, ...] = 2,
        center_refinement_factor: float | tuple[float, ...] = 1.0,
        origin_exclusion: float | tuple[float, ...] | None = 0.0,
        split_dim_weights: float | tuple[float, ...] = 1.0,
    ) -> RegionBuilder:
        resolved_bounds = bounds
        if resolved_bounds is None:
            resolved_bounds = [[-1.0, -1.0], [1.0, 1.0]]
        return RegionBuilder(
            bounds=resolved_bounds,
            bins_per_dim=bins_per_dim,
            center_refinement_factor=center_refinement_factor,
            origin_exclusion=origin_exclusion,
            split_dim_weights=split_dim_weights,
            device=th.device("cpu"),
        )

    def _assert_regions_equal_unordered(
        self,
        actual: th.Tensor | np.ndarray,
        expected: th.Tensor | np.ndarray,
    ) -> None:
        actual_np = _regions_to_numpy(actual)
        expected_np = _regions_to_numpy(expected)
        actual_keys = {_region_key(region) for region in actual_np}
        expected_keys = {_region_key(region) for region in expected_np}
        self.assertSetEqual(actual_keys, expected_keys)

    def _assert_region_plot_written(
        self,
        *,
        stem: str,
        region_groups: list[tuple[str, th.Tensor | np.ndarray, str]],
        title: str,
    ) -> None:
        self._assert_plot_written(
            plot_fn=plot_region_groups_2d,
            stem=stem,
            plot_kwargs={
                "region_groups": region_groups,
                "title": title,
            },
        )

    def test_init_normalizes_constructor_arguments(self) -> None:
        builder = self._make_builder(
            bounds=[[-2.0, -1.0], [2.0, 3.0]],
            bins_per_dim=4,
            center_refinement_factor=0.5,
            origin_exclusion=0.25,
        )

        self.assertEqual(builder.state_dim, 2)
        self.assertEqual(builder.bins_per_dim, (4, 4))
        self.assertEqual(builder.center_refinement_factor, (0.5, 0.5))
        th.testing.assert_close(
            builder.bounds,
            th.tensor([[-2.0, -1.0], [2.0, 3.0]], dtype=th.float32),
        )
        th.testing.assert_close(
            builder.origin_exclusion,
            th.tensor([0.25, 0.25], dtype=th.float32),
        )

    def test_default_settings(self) -> None:
        builder = self._make_builder(
            bounds=[[-1.0, -1.0], [1.0, 1.0]],
            bins_per_dim=1,
            origin_exclusion=0.25,
        )

        self.assertEqual(builder.state_dim, 2)
        self.assertEqual(builder.bins_per_dim, (1, 1))
        th.testing.assert_close(
            builder.origin_exclusion,
            th.tensor([0.25, 0.25], dtype=th.float32),
        )
        regions = builder.build_regions()
        self._assert_region_plot_written(
            stem="region_builder_build_regions_default",
            region_groups=[("regions", regions, "#ff7f0e")],
            title="RegionBuilder build_regions output",
        )


    def test_resolve_bounds_validates_shape_and_order(self) -> None:
        bounds = RegionBuilder._resolve_bounds([[-2.0, -1.0], [2.0, 3.0]], th.device("cpu"))
        th.testing.assert_close(
            bounds,
            th.tensor([[-2.0, -1.0], [2.0, 3.0]], dtype=th.float32),
        )

        with self.assertRaisesRegex(ValueError, "bounds must have shape"):
            RegionBuilder._resolve_bounds([-1.0, 1.0], th.device("cpu"))

        with self.assertRaisesRegex(ValueError, "strictly greater"):
            RegionBuilder._resolve_bounds([[-1.0, 0.0], [-1.0, 1.0]], th.device("cpu"))

    def test_normalize_bins_handles_scalar_sequence_and_invalid_values(self) -> None:
        builder = self._make_builder(bounds=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

        self.assertEqual(builder._normalize_bins(3), (3, 3, 3))
        self.assertEqual(builder._normalize_bins((1, 2, 3)), (1, 2, 3))

        with self.assertRaisesRegex(ValueError, "match state_dim"):
            builder._normalize_bins((1, 2))

        with self.assertRaisesRegex(ValueError, "positive integers"):
            builder._normalize_bins((1, 0, 3))

    def test_normalize_refinement_factors_handles_scalar_sequence_and_invalid_values(self) -> None:
        builder = self._make_builder(bounds=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

        self.assertEqual(builder._normalize_refinement_factors(0.5), (0.5, 0.5, 0.5))
        self.assertEqual(
            builder._normalize_refinement_factors((0.25, 0.5, 1.0)),
            (0.25, 0.5, 1.0),
        )

        with self.assertRaisesRegex(ValueError, "match state_dim"):
            builder._normalize_refinement_factors((0.5, 0.5))

        with self.assertRaisesRegex(ValueError, "values in \(0, 1\]"):
            builder._normalize_refinement_factors((0.5, 1.5, 0.5))

    def test_resolve_origin_exclusion_defaults_clamps_and_validates(self) -> None:
        builder = self._make_builder(bounds=[[-2.0, -0.5], [1.0, 3.0]])

        default_exclusion = builder._resolve_origin_exclusion(None)
        th.testing.assert_close(
            default_exclusion,
            th.tensor([0.02, 0.03], dtype=th.float32),
            atol=1e-7,
            rtol=0.0,
        )

        th.testing.assert_close(
            builder._resolve_origin_exclusion(10.0),
            th.tensor([1.0, 0.5], dtype=th.float32),
        )
        th.testing.assert_close(
            builder._resolve_origin_exclusion((0.1, 0.2)),
            th.tensor([0.1, 0.2], dtype=th.float32),
        )

        with self.assertRaisesRegex(ValueError, "match state_dim"):
            builder._resolve_origin_exclusion((0.1, 0.2, 0.3))

        with self.assertRaisesRegex(ValueError, "non-negative"):
            builder._resolve_origin_exclusion((-0.1, 0.2))

    def test_build_center_refined_axis_edges_refines_toward_the_origin(self) -> None:
        refined = RegionBuilder._build_center_refined_axis_edges(
            th.tensor(-2.0),
            th.tensor(2.0),
            4,
            0.5,
        )

        th.testing.assert_close(
            refined,
            th.tensor([-2.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, 2.0], dtype=th.float32),
            atol=1e-6,
            rtol=0.0,
        )

    def test_build_center_refined_axis_edges_uses_uniform_edges_when_refinement_does_not_apply(self) -> None:
        non_crossing = RegionBuilder._build_center_refined_axis_edges(
            th.tensor(1.0),
            th.tensor(3.0),
            3,
            0.5,
        )
        th.testing.assert_close(
            non_crossing,
            th.linspace(1.0, 3.0, steps=4),
            atol=1e-6,
            rtol=0.0,
        )

        uniform = RegionBuilder._build_center_refined_axis_edges(
            th.tensor(-2.0),
            th.tensor(2.0),
            4,
            1.0,
        )
        th.testing.assert_close(
            uniform,
            th.linspace(-2.0, 2.0, steps=5),
            atol=1e-6,
            rtol=0.0,
        )

    def test_pack_regions_stacks_bounds_along_the_region_axis(self) -> None:
        lbs = th.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=th.float32)
        ubs = th.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=th.float32)

        packed = RegionBuilder._pack_regions(lbs, ubs)

        th.testing.assert_close(
            packed,
            th.tensor(
                [
                    [[0.0, 1.0], [1.0, 2.0]],
                    [[2.0, 3.0], [3.0, 4.0]],
                ],
                dtype=th.float32,
            ),
        )

    def test_subtract_origin_exclusion_returns_original_region_when_no_hole_is_hit(self) -> None:
        builder = self._make_builder(origin_exclusion=0.25)

        subregions = builder._subtract_origin_exclusion_from_region(
            th.tensor([0.5, -1.0], dtype=th.float32),
            th.tensor([1.0, 1.0], dtype=th.float32),
        )

        self.assertEqual(len(subregions), 1)
        th.testing.assert_close(subregions[0][0], th.tensor([0.5, -1.0], dtype=th.float32))
        th.testing.assert_close(subregions[0][1], th.tensor([1.0, 1.0], dtype=th.float32))

    def test_subtract_origin_exclusion_splits_region_around_the_central_hole(self) -> None:
        builder = self._make_builder(origin_exclusion=0.25)

        subregions = builder._subtract_origin_exclusion_from_region(
            th.tensor([-1.0, -1.0], dtype=th.float32),
            th.tensor([1.0, 1.0], dtype=th.float32),
        )

        actual = th.stack([th.stack([lb, ub], dim=0) for lb, ub in subregions], dim=0)
        expected = th.tensor(
            [
                [[-1.0, -1.0], [-0.25, 1.0]],
                [[0.25, -1.0], [1.0, 1.0]],
                [[-0.25, -1.0], [0.25, -0.25]],
                [[-0.25, 0.25], [0.25, 1.0]],
            ],
            dtype=th.float32,
        )
        self._assert_regions_equal_unordered(actual, expected)

    def test_validate_regions_casts_dtype_and_rejects_invalid_inputs(self) -> None:
        builder = self._make_builder()

        validated = builder._validate_regions(
            th.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=th.float64)
        )
        self.assertEqual(validated.dtype, th.float32)

        with self.assertRaisesRegex(ValueError, "regions must have shape"):
            builder._validate_regions(th.zeros((2, 2), dtype=th.float32))

        with self.assertRaisesRegex(ValueError, "strictly greater"):
            builder._validate_regions(
                th.tensor([[[0.0, 0.0], [0.0, 1.0]]], dtype=th.float32)
            )

    def test_split_regions_uses_the_widest_dimension_by_default(self) -> None:
        builder = self._make_builder(origin_exclusion=0.0)
        regions = th.tensor(
            [
                [[0.0, 0.0], [4.0, 1.0]],
                [[0.0, 0.0], [1.0, 4.0]],
            ],
            dtype=th.float32,
        )

        split = builder.split_regions(regions)

        expected = th.tensor(
            [
                [[0.0, 0.0], [2.0, 1.0]],
                [[0.0, 0.0], [1.0, 2.0]],
                [[2.0, 0.0], [4.0, 1.0]],
                [[0.0, 2.0], [1.0, 4.0]],
            ],
            dtype=th.float32,
        )
        th.testing.assert_close(split, expected)

    def test_split_dim_weights_bias_selection_toward_prioritized_dimension(self) -> None:
        # Dimension 0 is wider, but a large weight on dimension 1 must flip the
        # anisotropic argmax(weight * width) selection to split dimension 1.
        builder = self._make_builder(
            origin_exclusion=0.0,
            split_dim_weights=(1.0, 5.0),
        )
        regions = th.tensor(
            [[[0.0, 0.0], [4.0, 2.0]]],
            dtype=th.float32,
        )

        split = builder.split_regions(regions)

        expected = th.tensor(
            [
                [[0.0, 0.0], [4.0, 1.0]],
                [[0.0, 1.0], [4.0, 2.0]],
            ],
            dtype=th.float32,
        )
        th.testing.assert_close(split, expected)

    def test_split_dim_weights_reduce_to_widest_dimension_when_uniform(self) -> None:
        builder = self._make_builder(origin_exclusion=0.0, split_dim_weights=1.0)
        regions = th.tensor([[[0.0, 0.0], [4.0, 1.0]]], dtype=th.float32)

        split = builder.split_regions(regions)

        expected = th.tensor(
            [
                [[0.0, 0.0], [2.0, 1.0]],
                [[2.0, 0.0], [4.0, 1.0]],
            ],
            dtype=th.float32,
        )
        th.testing.assert_close(split, expected)

    def test_normalize_split_dim_weights_handles_scalar_sequence_and_invalid_values(self) -> None:
        builder = self._make_builder(bounds=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

        th.testing.assert_close(
            builder._normalize_split_dim_weights(2.0),
            th.tensor([2.0, 2.0, 2.0], dtype=th.float32),
        )
        th.testing.assert_close(
            builder._normalize_split_dim_weights((1.0, 2.0, 3.0)),
            th.tensor([1.0, 2.0, 3.0], dtype=th.float32),
        )

        with self.assertRaisesRegex(ValueError, "match state_dim"):
            builder._normalize_split_dim_weights((1.0, 2.0))

        with self.assertRaisesRegex(ValueError, "positive values"):
            builder._normalize_split_dim_weights((1.0, 0.0, 3.0))

    def test_split_regions_respects_explicit_split_dimensions_and_validates_them(self) -> None:
        builder = self._make_builder(origin_exclusion=0.0)
        regions = th.tensor(
            [
                [[0.0, 0.0], [4.0, 1.0]],
                [[0.0, 0.0], [1.0, 4.0]],
            ],
            dtype=th.float32,
        )

        split = builder.split_regions(regions, split_dims=th.tensor([1, 0]))
        expected = th.tensor(
            [
                [[0.0, 0.0], [4.0, 0.5]],
                [[0.0, 0.0], [0.5, 4.0]],
                [[0.0, 0.5], [4.0, 1.0]],
                [[0.5, 0.0], [1.0, 4.0]],
            ],
            dtype=th.float32,
        )
        th.testing.assert_close(split, expected)

        with self.assertRaisesRegex(ValueError, "exactly one split dimension"):
            builder.split_regions(regions, split_dims=th.tensor([0]))

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            builder.split_regions(regions, split_dims=th.tensor([0, 2]))

    def test_face_adjacency_mask_detects_shared_faces_but_not_corners(self) -> None:
        builder = self._make_builder(bounds=[[-3.0, -3.0], [3.0, 3.0]])
        regions = th.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[1.0, 0.0], [2.0, 1.0]],
                [[1.0, 1.0], [2.0, 2.0]],
                [[2.0, 1.0], [3.0, 2.0]],
                [[3.0, 0.0], [4.0, 1.0]],
            ],
            dtype=th.float32,
        )
        reference = th.tensor([[[1.0, 0.0], [2.0, 1.0]]], dtype=th.float32)

        adjacent = builder._face_adjacency_mask(regions, reference)

        th.testing.assert_close(
            adjacent,
            th.tensor([True, False, True, False, False], dtype=th.bool),
        )

    def test_split_regions_adjacent_to_reference_returns_only_frontier_children(self) -> None:
        builder = self._make_builder(bounds=[[-3.0, -3.0], [3.0, 3.0]])
        regions = th.tensor(
            [
                [[0.0, 0.0], [2.0, 1.0]],
                [[-2.0, -1.0], [-1.0, 0.0]],
            ],
            dtype=th.float32,
        )
        reference = th.tensor([[[2.0, 0.0], [4.0, 1.0]]], dtype=th.float32)

        frontier_children, terminal_regions = builder.split_regions_adjacent_to_reference(
            regions,
            reference,
        )

        expected_frontier = th.tensor([[[1.0, 0.0], [2.0, 1.0]]], dtype=th.float32)
        expected_terminal = th.tensor(
            [
                [[-2.0, -1.0], [-1.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ],
            dtype=th.float32,
        )
        self._assert_regions_equal_unordered(frontier_children, expected_frontier)
        self._assert_regions_equal_unordered(terminal_regions, expected_terminal)
        self._assert_region_plot_written(
            stem="region_builder_frontier_split",
            region_groups=[
                ("reference", reference, "#1f77b4"),
                ("frontier_children", frontier_children, "#2ca02c"),
                ("terminal", terminal_regions, "#d62728"),
            ],
            title="RegionBuilder frontier split",
        )

    def test_split_regions_adjacent_to_reference_splits_everything_when_reference_is_empty(self) -> None:
        builder = self._make_builder(bounds=[[-3.0, -3.0], [3.0, 3.0]])
        regions = th.tensor(
            [
                [[0.0, 0.0], [2.0, 1.0]],
                [[-2.0, -1.0], [-1.0, 0.0]],
            ],
            dtype=th.float32,
        )

        split_children, terminal_regions = builder.split_regions_adjacent_to_reference(
            regions,
            th.empty((0, 2, 2), dtype=th.float32),
        )

        expected_children = th.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[-2.0, -1.0], [-1.5, 0.0]],
                [[1.0, 0.0], [2.0, 1.0]],
                [[-1.5, -1.0], [-1.0, 0.0]],
            ],
            dtype=th.float32,
        )
        self._assert_regions_equal_unordered(split_children, expected_children)
        self.assertEqual(tuple(terminal_regions.shape), (0, 2, 2))

    def test_split_region_matches_the_single_region_batch_variant(self) -> None:
        builder = self._make_builder(origin_exclusion=0.0)
        region = th.tensor([[0.0, 0.0], [2.0, 1.0]], dtype=th.float32)

        single = builder.split_region(region, split_dim=1)
        batched = builder.split_regions(region.unsqueeze(0), split_dims=th.tensor([1]))

        th.testing.assert_close(single, batched)

    def test_build_regions_tiles_the_outer_box_minus_the_origin_hole(self) -> None:
        builder = self._make_builder(
            bounds=[[-2.0, -1.0], [2.0, 3.0]],
            bins_per_dim=(4, 2),
            center_refinement_factor=(0.5, 1.0),
            origin_exclusion=0.25,
        )

        regions = builder.build_regions()
        self.assertEqual(tuple(regions.shape), (12, 2, 2))
        self.assertTrue(bool(th.all(regions[:, 1] > regions[:, 0]).item()))
        self.assertTrue(bool(th.all(regions[:, 0] >= builder.bounds[0]).item()))
        self.assertTrue(bool(th.all(regions[:, 1] <= builder.bounds[1]).item()))

        regions_np = _regions_to_numpy(regions)
        hole_lb = np.array([-0.25, -0.25], dtype=np.float32)
        hole_ub = np.array([0.25, 0.25], dtype=np.float32)
        overlap_lb = np.maximum(regions_np[:, 0, :], hole_lb)
        overlap_ub = np.minimum(regions_np[:, 1, :], hole_ub)
        overlap_width = np.maximum(overlap_ub - overlap_lb, 0.0)
        overlap_area = np.prod(overlap_width, axis=1)
        self.assertTrue(np.all(overlap_area <= 1e-8))

        region_widths = regions_np[:, 1, :] - regions_np[:, 0, :]
        total_area = float(np.prod(region_widths, axis=1).sum())
        self.assertAlmostEqual(total_area, 16.0 - 0.25, places=5)

        self._assert_region_plot_written(
            stem="region_builder_build_regions",
            region_groups=[("regions", regions, "#ff7f0e")],
            title="RegionBuilder build_regions output",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)