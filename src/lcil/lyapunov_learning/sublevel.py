from __future__ import annotations

import logging
import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Sequence
from numpy.typing import NDArray

from .config import LyapunovTrainingConfig
from .sampling import (
    _bounds_tensor,
    sample_boundary_points,
    project_to_boundary_faces,
)

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BoundaryTermDiagnostics:
    feature_term_quantile: float
    pd_term_quantile: float
    feature_term_mean: float
    pd_term_mean: float
    feature_term_mean_share: float
    pd_term_mean_share: float

    @classmethod
    def nan(cls) -> "BoundaryTermDiagnostics":
        nan = float("nan")
        return cls(
            feature_term_quantile=nan,
            pd_term_quantile=nan,
            feature_term_mean=nan,
            pd_term_mean=nan,
            feature_term_mean_share=nan,
            pd_term_mean_share=nan,
        )


@dataclass(frozen=True, slots=True)
class BoundaryRhoEstimate:
    rho: float
    boundary_quantile: float
    boundary_mean: float


@dataclass(frozen=True, slots=True)
class BoundaryRhoEvaluation:
    rho: BoundaryRhoEstimate
    terms: BoundaryTermDiagnostics


@dataclass(frozen=True, slots=True)
class RhoEstimationConfig:
    """Configuration for Lyapunov sublevel set (rho) estimation."""

    train_bounds: NDArray | Sequence[float] | th.Tensor
    samples: int = 1024
    step_size: float = 0.05
    descent_steps: int = 10
    estimate_quantile: float = 0.05
    rho_min: float = 1e-4
    rho_growth_gamma: float = 1.0
    enable_diagnosis: bool = False

    @classmethod
    def from_training_config(
        cls,
        config: LyapunovTrainingConfig,
        **overrides,
    ) -> "RhoEstimationConfig":
        """Build a RhoEstimationConfig from a LyapunovTrainingConfig."""
        bounds = config.train_bounds if config.train_bounds is not None else config.state_bounds
        data = {
            "train_bounds": bounds,
            "samples": config.rho_estimation_samples,
            "step_size": config.rho_step_size,
            "descent_steps": config.rho_descent_steps,
            "estimate_quantile": config.rho_estimate_quantile,
            "rho_min": config.rho_min,
            "rho_growth_gamma": config.rho_growth_gamma,
            "enable_diagnosis": config.enable_diagnosis,
        }
        data.update(overrides)
        return cls(**data)


def _boundary_term_diagnostics(
    lyap_model: nn.Module,
    boundary_x: th.Tensor,
    quantile: float,
) -> BoundaryTermDiagnostics:
    if not all(
        hasattr(lyap_model, attr)
        for attr in ("feature_net", "x_star", "_pd_weight", "get_feature_term", "get_pd_term")
    ):
        return BoundaryTermDiagnostics.nan()

    pd_weight_fn = getattr(lyap_model, "_pd_weight")
    get_feature_term_fn = getattr(lyap_model, "get_feature_term")
    get_pd_term_fn = getattr(lyap_model, "get_pd_term")
    if not callable(pd_weight_fn) or not callable(get_feature_term_fn) or not callable(get_pd_term_fn):
        return BoundaryTermDiagnostics.nan()

    feature_term = get_feature_term_fn(boundary_x)
    pd_term = get_pd_term_fn(boundary_x)

    feature_term_mean = float(feature_term.mean().item())
    pd_term_mean = float(pd_term.mean().item())
    total_mean = feature_term_mean + pd_term_mean

    if total_mean <= 0.0:
        feature_term_mean_share = float("nan")
        pd_term_mean_share = float("nan")
    else:
        feature_term_mean_share = feature_term_mean / total_mean
        pd_term_mean_share = pd_term_mean / total_mean

    return BoundaryTermDiagnostics(
        feature_term_quantile=float(th.quantile(feature_term, q=quantile).item()),
        pd_term_quantile=float(th.quantile(pd_term, q=quantile).item()),
        feature_term_mean=feature_term_mean,
        pd_term_mean=pd_term_mean,
        feature_term_mean_share=feature_term_mean_share,
        pd_term_mean_share=pd_term_mean_share,
    )


def estimate_rho_from_boundary(
    lyap_model: nn.Module,
    config: RhoEstimationConfig,
    device: th.device | int | str | None = None,
    generator: th.Generator | None = None,
) -> tuple[BoundaryRhoEvaluation, th.Tensor]:
    """Estimate rho and expose boundary-term diagnostics for logging."""
    bounds = _bounds_tensor(config.train_bounds, device)
    lbx, ubx = bounds[0], bounds[1]
    boundary_x, face_dims, is_ub = sample_boundary_points(
        sample_size=config.samples,
        lb=lbx,
        ub=ubx,
        device=device,
        generator=generator,
    )
    
    step = config.step_size * (ubx - lbx).unsqueeze(0)
    with th.no_grad():
        best_boundary_x = boundary_x.clone()
        best_boundary_values = lyap_model(boundary_x).flatten()

    for _ in range(config.descent_steps):
        boundary_x.requires_grad_(True)
        boundary_values = lyap_model(boundary_x)
        grad = th.autograd.grad(
            boundary_values.mean(),
            boundary_x,
            retain_graph=False,
            create_graph=False,
        )[0]

        with th.no_grad():
            candidate_x = boundary_x - step * grad.sign()
            candidate_x = project_to_boundary_faces(
                candidate_x,
                lb=lbx,
                ub=ubx,
                face_dims=face_dims,
                is_ub=is_ub,
            )
            
            candidate_values = lyap_model(candidate_x).flatten()
            improved = candidate_values < best_boundary_values
            best_boundary_x[improved] = candidate_x[improved]
            best_boundary_values[improved] = candidate_values[improved]

    boundary_eval_x = best_boundary_x.detach()
    boundary_values = best_boundary_values.detach()

    with th.no_grad():
        boundary_values = lyap_model(boundary_eval_x).flatten()
        boundary_quantile = float(th.quantile(boundary_values, q=float(config.estimate_quantile)).item())
        boundary_mean = float(boundary_values.mean().item())
        
        if config.enable_diagnosis:
            term_diagnostics = _boundary_term_diagnostics(
                lyap_model=lyap_model,
                boundary_x=boundary_eval_x,
                quantile=float(config.estimate_quantile),
            )
        else:
            term_diagnostics = BoundaryTermDiagnostics.nan()

    rho_boundary = max(config.rho_min, config.rho_growth_gamma * boundary_quantile)
    evaluation = BoundaryRhoEvaluation(
        rho=BoundaryRhoEstimate(
            rho=float(rho_boundary),
            boundary_quantile=boundary_quantile,
            boundary_mean=boundary_mean,
        ),
        terms=term_diagnostics,
    )
    return evaluation, boundary_eval_x