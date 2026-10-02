import unittest
import torch as th
import plotly.graph_objects as go
from typing import Any

from plot_assertions_mixin import PlotAssertionsMixin
from lcil.lyapunov_learning.sampling import (
    sample_uniform_box,
    sample_boundary_points,
    sample_box_shell,
    sample_ellipsoid_boundary,
    sample_sobol_box,
    sample_box_rejection_states,
    sample_axis_antiphase_states,
    sample_eigenvector_antiphase_states,
)
from lcil.lyapunov_learning.utils import (
    compute_closed_loop_jacobian,
    compute_antiphase_eigenvectors,
)

def plot_sampling_methods(
    uniform_pts: th.Tensor,
    boundary_pts: th.Tensor,
    box_shell_pts: th.Tensor,
    ellipsoid_pts: th.Tensor,
    sobol_pts: th.Tensor,
    rejection_pts: th.Tensor,
    html_path: str,
) -> None:
    fig = go.Figure()
    
    def add_trace(pts: th.Tensor, name: str, marker_symbol: str = "circle", opacity: float = 0.7):
        pts_np = pts.cpu().numpy()
        fig.add_trace(go.Scatter(
            x=pts_np[:, 0],
            y=pts_np[:, 1],
            mode="markers",
            name=name,
            marker=dict(size=5, symbol=marker_symbol, opacity=opacity),
        ))
        
    add_trace(uniform_pts, "Uniform Box (sample_uniform_box)")
    add_trace(boundary_pts, "Box Boundary (sample_boundary_points)", marker_symbol="cross")
    add_trace(box_shell_pts, "Box Shell (sample_box_shell)")
    add_trace(ellipsoid_pts, "Ellipsoid Boundary (sample_ellipsoid_boundary)")
    add_trace(sobol_pts, "Sobol Box (sample_sobol_box)", marker_symbol="diamond", opacity=0.9)
    add_trace(rejection_pts, "Box Rejection (Center weighted)")

    fig.update_layout(
        title="Comparison of Lyapunov Sampling Methods (2D Projection)",
        xaxis_title="State 0",
        yaxis_title="State 1",
        width=800,
        height=800,
    )
    fig.write_html(html_path)

class TestSamplingMethods(PlotAssertionsMixin):
    def test_sampling_methods_plot(self):
        device = th.device("cpu")
        lb = th.tensor([-1.0, -1.0], device=device)
        ub = th.tensor([1.0, 1.0], device=device)
        center = th.tensor([0.0, 0.0], device=device)
        half_width = th.tensor([1.0, 1.0], device=device)
        
        sample_size = 800
        
        # 1. Uniform Box
        uniform_pts = sample_uniform_box(sample_size, lb, ub, device=device)
        
        # 2. Boundary Points
        boundary_pts, _, _ = sample_boundary_points(sample_size, lb, ub, device=device)
        
        # 3. Box Shell
        box_shell_pts = sample_box_shell(
            sample_size=sample_size,
            state_dim=2,
            center=center,
            half_width=half_width,
            device=device,
        )
        
        # 4. Ellipsoid Boundary
        ellipsoid_pts = sample_ellipsoid_boundary(
            sample_size=sample_size,
            state_dim=2,
            center=center,
            half_width=half_width,
            device=device,
        )
        
        # 5. Sobol Box
        sobol_engine = th.quasirandom.SobolEngine(dimension=2, scramble=True)
        sobol_pts = sample_sobol_box(sample_size, lb, ub, sobol_engine, device=device)
        
        # 5. Box Rejection States
        # Real-world scenario: Quadratic Lyapunov function V(x) = x^T P x
        P = th.tensor([[0.9, 1.4], [1.4, 1.8]], device=device)
        def value_fn(x: th.Tensor) -> th.Tensor:
            return (x @ P * x).sum(dim=1)
            
        rho_estimate = 0.5
        rho_margin = 2.0
        rho_target = rho_margin * rho_estimate
        rho_scale = max(rho_estimate, 1e-9)
        
        def score_fn(x: th.Tensor) -> th.Tensor:
            return (rho_target - value_fn(x)) / rho_scale
            
        rejection_pts = sample_box_rejection_states(
            lb=lb,
            ub=ub,
            target_count=sample_size,
            score_fn=score_fn,
            oversample_factor=5,
            sharpness=2.0,
            device=device,
        )
        
        self._assert_plot_written(
            plot_fn=plot_sampling_methods,
            stem="sampling_methods_comparison",
            plot_kwargs={
                "uniform_pts": uniform_pts,
                "boundary_pts": boundary_pts,
                "box_shell_pts": box_shell_pts,
                "ellipsoid_pts": ellipsoid_pts,
                "sobol_pts": sobol_pts,
                "rejection_pts": rejection_pts,
            }
        )

    def test_axis_antiphase_sampling(self):
        device = th.device("cpu")
        lb = th.tensor([-1.0, -3.0, -0.75, -3.0], device=device)
        ub = th.tensor([1.0, 3.0, 0.75, 3.0], device=device)
        exclusion = [0.002, 0.005, 0.001, 0.005]
        sample_size = 200
        scale_factor = 1.0

        samples = sample_axis_antiphase_states(
            sample_size=sample_size,
            lb=lb,
            ub=ub,
            scale_factor=scale_factor,
            device=device,
        )

        self.assertEqual(samples.shape, (sample_size, 4))

        # 1. No sample should be all-zeros (the origin)
        active_counts = (samples != 0.0).sum(dim=-1)
        self.assertTrue((active_counts >= 1).all().item())

        # 2. Sparsity: some samples should be on 1D axes (active count 1) or 2D planes (active count 2)
        self.assertTrue((active_counts == 1).any().item())
        self.assertTrue((active_counts == 2).any().item())

        # 3. Active dimensions must be within bounds
        abs_samples = samples.abs()
        for i in range(4):
            active_i = abs_samples[:, i] > 0.0
            if active_i.any():
                self.assertTrue((samples[active_i, i] >= lb[i] - 1e-7).all().item())
                self.assertTrue((samples[active_i, i] <= ub[i] + 1e-7).all().item())

        # 4. Both positive and negative signs should be represented
        has_positive = (samples > 0.0).any(dim=0)
        has_negative = (samples < 0.0).any(dim=0)
        self.assertTrue(has_positive.all().item())
        self.assertTrue(has_negative.all().item())

        # 5. Empty sample size edge case
        empty = sample_axis_antiphase_states(0, lb=lb, ub=ub, device=device)
        self.assertEqual(empty.shape, (0, 4))

    def test_sample_eigenvector_antiphase_states(self) -> None:
        device = th.device("cpu")
        lb = th.tensor([-1.0, -3.0, -0.75, -3.0], device=device)
        ub = th.tensor([1.0, 3.0, 0.75, 3.0], device=device)
        vec = th.tensor([0.3264, -0.7884, 0.1385, -0.1158], device=device)
        sample_size = 200

        # 1. Single eigenvector
        samples = sample_eigenvector_antiphase_states(
            sample_size=sample_size,
            lb=lb,
            ub=ub,
            eigenvector=vec,
            scale_factor=1.0,
            device=device,
        )
        self.assertEqual(samples.shape, (sample_size, 4))

        # Check bounds
        self.assertTrue((samples >= lb - 1e-6).all().item())
        self.assertTrue((samples <= ub + 1e-6).all().item())

        # Check collinearity on the eigenvector ray: x_i / v_i should be identical for all nonzero dimensions
        v_unit = vec / th.linalg.norm(vec)
        ratios = samples / v_unit.unsqueeze(0)
        # Difference between any two coordinate ratios should be ~0
        collinear_diff = (ratios[:, 0] - ratios[:, 1]).abs()
        self.assertTrue((collinear_diff < 1e-4).all().item())

        # Check both positive and negative directions along ray are present
        has_pos_ray = (ratios[:, 0] > 0.0).any().item()
        has_neg_ray = (ratios[:, 0] < 0.0).any().item()
        self.assertTrue(has_pos_ray)
        self.assertTrue(has_neg_ray)

        # 2. Multiple eigenvectors
        vecs = th.stack([vec, th.tensor([-0.0697, 0.5277, 0.1109, -0.8393], device=device)], dim=0)
        samples_multi = sample_eigenvector_antiphase_states(
            sample_size=sample_size,
            lb=lb,
            ub=ub,
            eigenvector=vecs,
            device=device,
        )
        self.assertEqual(samples_multi.shape, (sample_size, 4))
        self.assertTrue((samples_multi >= lb - 1e-6).all().item())
        self.assertTrue((samples_multi <= ub + 1e-6).all().item())

        # 3. Empty sample size
        empty = sample_eigenvector_antiphase_states(0, lb=lb, ub=ub, eigenvector=vec, device=device)
        self.assertEqual(empty.shape, (0, 4))

    def test_compute_antiphase_eigenvectors(self) -> None:
        # 1. 2x2 matrix with known antiphase mode
        A = th.tensor([[0.9, -0.2], [-0.1, 0.95]], dtype=th.float32)
        antiphase_vecs = compute_antiphase_eigenvectors(A)
        self.assertGreater(len(antiphase_vecs), 0)
        # Vector must be unit norm
        self.assertAlmostEqual(th.linalg.norm(antiphase_vecs[0]).item(), 1.0, places=5)
        # Must have opposing signs
        self.assertTrue((antiphase_vecs[0, 0] * antiphase_vecs[0, 1] < 0).item())

        # 2. From dynamics and policy callables
        A_dyn = th.tensor([[1.0, 0.1], [0.0, 1.0]])
        B_dyn = th.tensor([[0.0], [0.1]])
        K_pol = th.tensor([[-2.0, -1.5]])  # u = K x

        def dyn(x: th.Tensor, u: th.Tensor) -> th.Tensor:
            return x @ A_dyn.T + u @ B_dyn.T

        def pol(x: th.Tensor) -> th.Tensor:
            return x @ K_pol.T

        jac = compute_closed_loop_jacobian(dyn, pol, state_dim=2)
        expected_jac = A_dyn + B_dyn @ K_pol
        self.assertTrue(th.allclose(jac, expected_jac, atol=1e-5))

        vecs = compute_antiphase_eigenvectors(dynamics=dyn, policy=pol, state_dim=2)
        self.assertGreater(len(vecs), 0)


if __name__ == "__main__":
    unittest.main()

