import unittest
import torch as th
import plotly.graph_objects as go
from typing import Any

from plot_assertions_mixin import PlotAssertionsMixin
from lcil.lyapunov_learning.sampling import (
    sample_uniform_box,
    sample_boundary_points,
    sample_ellipsoid_boundary,
    sample_sobol_box,
    sample_box_rejection_states,
)

def plot_sampling_methods(
    uniform_pts: th.Tensor,
    boundary_pts: th.Tensor,
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
        
        # 3. Ellipsoid Boundary
        ellipsoid_pts = sample_ellipsoid_boundary(
            sample_size=sample_size,
            state_dim=2,
            center=center,
            half_width=half_width,
            device=device,
        )
        
        # 4. Sobol Box
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
                "ellipsoid_pts": ellipsoid_pts,
                "sobol_pts": sobol_pts,
                "rejection_pts": rejection_pts,
            }
        )

if __name__ == "__main__":
    unittest.main()
