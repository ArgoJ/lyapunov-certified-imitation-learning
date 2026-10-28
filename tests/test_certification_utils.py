import math
import os
import types
import unittest
from collections.abc import Sequence
from typing import Any

os.environ.setdefault("PYTORCH_JIT", "0")

import numpy as np
import plotly.graph_objects as go
import torch as th
from mpc_datagen import mdg_plt

from plot_assertions_mixin import PlotAssertionsMixin

from lcil.certification import build_blended_lqr_policy, compute_blended_lqr_c0_c1
from lcil.imitation_learning import BlendedLQRPolicy, BoundedPolicy, LQRController
from lcil.utils import MLP, extract_p_matrix, quad_form
from lcil.utils.lcil_plt.lyapunov import lyapunov_with_exclusion
from lcil.utils.lcil_plt.utils import add_regions, create_origin_exclusion_region


def plot_blended_lqr_c0_c1(
    exclusion_box: Sequence[float],
    P: Any,
    c0: float,
    c1: float,
    html_path: str,
    factor: float = 4.0,
    box_samples: th.Tensor | np.ndarray | None = None,
    expanded_samples: th.Tensor | np.ndarray | None = None,
    title: str | None = None,
) -> None:
    p_th = extract_p_matrix(P, as_tensor=True, dtype=th.float64)
    num_states = len(exclusion_box)
    scale = math.sqrt(factor)
    expanded_exclusion = [scale * float(r) for r in exclusion_box]

    def lyapunov_func(x: np.ndarray) -> np.ndarray:
        x_th = th.as_tensor(np.atleast_2d(x), dtype=th.float64, device=p_th.device)
        return quad_form(x_th, p_th).squeeze(-1).detach().cpu().numpy()

    results = lyapunov_with_exclusion(
        lyapunov_func=lyapunov_func,
        roa_level=c1,
        origin_exclusion=exclusion_box,
        num_states=num_states,
        html_path=None,
    )

    for result in results:
        fig = result.figure
        landscape_trace = fig.data[0]

        if len(fig.data) > 1:
            fig.data[1].update(
                name=mdg_plt.utils._to_latex(f"c1 = {c1:.4g} (alpha = 1)"),
                showlegend=True,
            )

        fig.add_trace(
            go.Contour(
                z=landscape_trace.z,
                x=landscape_trace.x,
                y=landscape_trace.y,
                contours=dict(
                    type="constraint",
                    operation="<=",
                    value=c0,
                ),
                fillcolor="rgba(0, 229, 255, 0.15)",
                line=dict(color="#00E5FF", width=3, dash="dash", smoothing=1.1),
                showscale=False,
                name=mdg_plt.utils._to_latex(f"c0 = {c0:.4g} (alpha = 0)"),
                showlegend=True,
            )
        )

        expanded_box = create_origin_exclusion_region(
            expanded_exclusion,
            idx_x=result.idx_x,
            idx_y=result.idx_y,
            num_states=num_states,
        )
        if expanded_box.shape[0] > 0:
            add_regions(
                expanded_box,
                fig,
                color="#FFA500",
                name=mdg_plt.utils._to_latex(f"{scale:.2g}x Exclusion Box"),
                fill=False,
                dash="dash",
            )

        if expanded_samples is not None:
            exp_np = th.as_tensor(expanded_samples).detach().cpu().numpy()
            fig.add_trace(
                go.Scatter(
                    x=exp_np[:, result.idx_x],
                    y=exp_np[:, result.idx_y],
                    mode="markers",
                    marker=dict(size=5, color="#FFA500", symbol="x", opacity=0.75),
                    name=mdg_plt.utils._to_latex(f"Samples in {scale:.2g}x box"),
                )
            )

        if box_samples is not None:
            box_np = th.as_tensor(box_samples).detach().cpu().numpy()
            fig.add_trace(
                go.Scatter(
                    x=box_np[:, result.idx_x],
                    y=box_np[:, result.idx_y],
                    mode="markers",
                    marker=dict(size=5, color="#FFD700", opacity=0.85),
                    name=mdg_plt.utils._to_latex("Samples in exclusion box"),
                )
            )

        if title is not None:
            fig.update_layout(title_text=mdg_plt.utils._to_latex(title))

    mdg_plt.utils._save_pair_figures(results, html_path, kind="Lyapunov", tikz=False)


class TestCertificationUtils(PlotAssertionsMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.P_2d = th.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=th.float64)
        self.r_2d = (0.1, 0.2)

    def test_2d_analytical_value(self) -> None:
        # For P = [[2, 0.5], [0.5, 1]], r = (0.1, 0.2)
        # v1 = (0.1, 0.2) -> 2*0.01 + 0.04 + 2*0.5*0.1*0.2 = 0.02 + 0.04 + 0.02 = 0.08
        # v2 = (0.1, -0.2) -> 2*0.01 + 0.04 - 0.02 = 0.04
        # max on box is 0.08. Double box is 4 * 0.08 = 0.32
        c0, c1 = compute_blended_lqr_c0_c1(self.r_2d, self.P_2d)
        self.assertIsInstance(c0, float)
        self.assertIsInstance(c1, float)
        self.assertAlmostEqual(c0, 0.08, places=7)
        self.assertAlmostEqual(c1, 0.32, places=7)
        self.assertAlmostEqual(c1, 4.0 * c0, places=7)

        self._assert_plot_written(
            plot_fn=plot_blended_lqr_c0_c1,
            stem="blended_lqr_c0_c1_analytical",
            plot_kwargs={
                "exclusion_box": self.r_2d,
                "P": self.P_2d,
                "c0": c0,
                "c1": c1,
                "title": "BlendedLQRPolicy c0 & c1 (Analytical 2D P)",
            },
        )

    def test_numpy_array_p(self) -> None:
        p_np = np.array([[3.0, 0.0], [0.0, 2.0]])
        r = (0.5, 0.5)
        c0, c1 = compute_blended_lqr_c0_c1(r, p_np)
        self.assertAlmostEqual(c0, 1.25, places=7)
        self.assertAlmostEqual(c1, 5.0, places=7)

    def test_p_from_lqr_controller(self) -> None:
        dt = 0.1
        A = th.tensor([[1.0, dt], [0.0, 1.0]])
        B = th.tensor([[0.0], [dt]])
        Q = th.tensor([[10.0, 0.0], [0.0, 1.0]])
        R = th.tensor([[0.5]])
        lqr = LQRController(A, B, R, Q=Q)

        r = (0.1, 0.1)
        c0, c1 = compute_blended_lqr_c0_c1(r, lqr)
        self.assertGreater(c0, 0.0)
        self.assertAlmostEqual(c1, 4.0 * c0, places=5)

        self._assert_plot_written(
            plot_fn=plot_blended_lqr_c0_c1,
            stem="blended_lqr_c0_c1_double_integrator_lqr",
            plot_kwargs={
                "exclusion_box": r,
                "P": lqr,
                "c0": c0,
                "c1": c1,
                "title": "BlendedLQRPolicy c0 & c1 (Double Integrator DARE P)",
            },
        )

    def test_p_from_blended_lqr_policy(self) -> None:
        dt = 0.1
        A = th.tensor([[1.0, dt], [0.0, 1.0]])
        B = th.tensor([[0.0], [dt]])
        Q = th.tensor([[10.0, 0.0], [0.0, 1.0]])
        R = th.tensor([[0.5]])
        lqr = LQRController(A, B, R, Q=Q)
        nn_policy = BoundedPolicy(
            feature_net=MLP(layer_dims=[2, 8, 1], activations=["relu", "identity"])
        )
        blended = BlendedLQRPolicy(nn_policy, lqr, c0=0.1, c1=0.5)

        c0, c1 = compute_blended_lqr_c0_c1((0.2, 0.1), blended)
        self.assertGreater(c0, 0.0)
        self.assertAlmostEqual(c1, 4.0 * c0, places=5)

    def test_points_in_box_satisfy_bounds(self) -> None:
        c0, c1 = compute_blended_lqr_c0_c1(self.r_2d, self.P_2d)
        torch_gen = th.Generator().manual_seed(42)
        u = th.rand(200, 2, generator=torch_gen, dtype=th.float64) * 2.0 - 1.0
        x_in_box = u * th.tensor(self.r_2d, dtype=th.float64)
        v_box = quad_form(x_in_box, self.P_2d)
        self.assertTrue((v_box <= c0 + 1e-9).all().item())

        x_in_double = u * (2.0 * th.tensor(self.r_2d, dtype=th.float64))
        v_double = quad_form(x_in_double, self.P_2d)
        self.assertTrue((v_double <= c1 + 1e-9).all().item())

        self._assert_plot_written(
            plot_fn=plot_blended_lqr_c0_c1,
            stem="blended_lqr_c0_c1_sampled_points",
            plot_kwargs={
                "exclusion_box": self.r_2d,
                "P": self.P_2d,
                "c0": c0,
                "c1": c1,
                "box_samples": x_in_box,
                "expanded_samples": x_in_double,
                "title": "BlendedLQRPolicy c0 & c1 with Sampled Box Points",
            },
        )

    def test_dimension_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_blended_lqr_c0_c1((0.1, 0.2, 0.3), self.P_2d)

    def test_non_square_p_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_blended_lqr_c0_c1((0.1, 0.2), th.randn(2, 3))

    def test_factor_custom(self) -> None:
        c0, c1_default = compute_blended_lqr_c0_c1(self.r_2d, self.P_2d)
        self.assertAlmostEqual(c1_default, 4.0 * c0, places=7)

        # Positional factor
        _, c1_pos = compute_blended_lqr_c0_c1(self.r_2d, self.P_2d, 2.5)
        self.assertAlmostEqual(c1_pos, 2.5 * c0, places=7)

        # Keyword factor
        _, c1_kw = compute_blended_lqr_c0_c1(self.r_2d, self.P_2d, factor=3.0)
        self.assertAlmostEqual(c1_kw, 3.0 * c0, places=7)

    def test_invalid_factor_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_blended_lqr_c0_c1(self.r_2d, self.P_2d, factor=1.0)
        with self.assertRaises(ValueError):
            compute_blended_lqr_c0_c1(self.r_2d, self.P_2d, factor=0.5)

    def test_build_blended_lqr_policy_from_dyn_and_mpc_config(self) -> None:
        dt = 0.1
        A_true = th.tensor([[1.0, dt], [0.0, 1.0]])
        B_true = th.tensor([[0.0], [dt]])
        Q_true = np.diag([10.0, 1.0])
        R_true = np.diag([0.5])
        W = np.block([[Q_true, np.zeros((2, 1))], [np.zeros((1, 2)), R_true]])

        def dyn_model(x: th.Tensor, u: th.Tensor) -> th.Tensor:
            return x @ A_true.T + u @ B_true.T

        mock_mpc_cfg = types.SimpleNamespace(
            nx=2,
            nu=1,
            cost=types.SimpleNamespace(W=W),
            constraints=types.SimpleNamespace(lbu=np.array([-1.0]), ubu=np.array([1.0])),
        )

        nn_policy = BoundedPolicy(
            feature_net=MLP(layer_dims=[2, 8, 1], activations=["relu", "identity"]),
            u_min=-1.0,
            u_max=1.0,
        )
        exclusion = (0.1, 0.1)

        blended = build_blended_lqr_policy(
            policy_model=nn_policy,
            dyn_model=dyn_model,
            mpc_config=mock_mpc_cfg,
            origin_exclusion=exclusion,
            factor=4.0,
        )
        self.assertIsInstance(blended, BlendedLQRPolicy)
        expected_c0, expected_c1 = compute_blended_lqr_c0_c1(exclusion, blended.lqr, factor=4.0)
        self.assertAlmostEqual(float(blended.c0.item()), expected_c0, places=5)
        self.assertAlmostEqual(float(blended.c1.item()), expected_c1, places=5)

        # At origin, blended policy must equal LQR policy (0)
        x_origin = th.zeros(1, 2)
        u_origin = blended(x_origin)
        self.assertTrue(th.allclose(u_origin, th.zeros(1, 1), atol=1e-6))

    def test_lqr_controller_save_load(self) -> None:
        import tempfile
        from pathlib import Path

        dt = 0.1
        A = th.tensor([[1.0, dt], [0.0, 1.0]])
        B = th.tensor([[0.0], [dt]])
        Q = th.tensor([[10.0, 0.0], [0.0, 1.0]])
        R = th.tensor([[0.5]])
        lqr = LQRController(A, B, R, Q=Q, u_min=-2.0, u_max=2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "lqr_controller.pt"
            lqr.save(save_path)
            self.assertTrue(save_path.is_file())

            loaded = LQRController.load(save_path)
            self.assertIsInstance(loaded, LQRController)

            x = th.randn(5, 2)
            self.assertTrue(th.allclose(lqr(x), loaded(x), atol=1e-6))
            self.assertTrue(th.allclose(lqr.K, loaded.K, atol=1e-6))
            self.assertTrue(th.allclose(lqr.P, loaded.P, atol=1e-6))

    def test_blended_lqr_policy_save_load(self) -> None:
        import tempfile
        from pathlib import Path

        dt = 0.1
        A = th.tensor([[1.0, dt], [0.0, 1.0]])
        B = th.tensor([[0.0], [dt]])
        Q = th.tensor([[10.0, 0.0], [0.0, 1.0]])
        R = th.tensor([[0.5]])
        lqr = LQRController(A, B, R, Q=Q, u_min=-1.0, u_max=1.0)
        nn_policy = BoundedPolicy(
            feature_net=MLP(layer_dims=[2, 8, 1], activations=["relu", "identity"]),
            u_min=-1.0,
            u_max=1.0,
        )
        c0, c1 = 0.05, 0.25
        blended = BlendedLQRPolicy(nn_policy, lqr, c0=c0, c1=c1)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "blended_policy.pt"
            blended.save(save_path)
            self.assertTrue(save_path.is_file())
            self.assertTrue((Path(tmpdir) / "blended_policy_inner.pt").is_file())
            self.assertTrue((Path(tmpdir) / "blended_policy_lqr.pt").is_file())

            loaded = BlendedLQRPolicy.load(save_path)
            self.assertIsInstance(loaded, BlendedLQRPolicy)
            self.assertAlmostEqual(float(loaded.c0.item()), c0, places=5)
            self.assertAlmostEqual(float(loaded.c1.item()), c1, places=5)

            x = th.randn(20, 2)
            self.assertTrue(th.allclose(blended(x), loaded(x), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
