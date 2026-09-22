from __future__ import annotations

import logging
import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Any, Callable, Sequence
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
    linear_term_quantile: float
    feature_term_mean: float
    linear_term_mean: float
    feature_term_mean_share: float
    linear_term_mean_share: float

    @classmethod
    def nan(cls) -> "BoundaryTermDiagnostics":
        nan = float("nan")
        return cls(
            feature_term_quantile=nan,
            linear_term_quantile=nan,
            feature_term_mean=nan,
            linear_term_mean=nan,
            feature_term_mean_share=nan,
            linear_term_mean_share=nan,
        )


@dataclass(frozen=True, slots=True)
class BoundaryRhoEstimate:
    rho: float
    boundary_quantile: float
    boundary_mean: float
    cex_cap: float | None = None


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
    origin_exclusion: float | Sequence[float] | NDArray | None = None
    enable_diagnosis: bool = False
    cex_quantile: float = 0.05

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
            "origin_exclusion": config.origin_exclusion,
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
        for attr in ("feature_net", "x_star", "_pd_matrix", "get_feature_term", "get_linear_term")
    ):
        return BoundaryTermDiagnostics.nan()

    pd_matrix_fn = getattr(lyap_model, "_pd_matrix")
    get_feature_term_fn = getattr(lyap_model, "get_feature_term")
    get_linear_term_fn = getattr(lyap_model, "get_linear_term")
    if not callable(pd_matrix_fn) or not callable(get_feature_term_fn) or not callable(get_linear_term_fn):
        return BoundaryTermDiagnostics.nan()

    feature_term = get_feature_term_fn(boundary_x)
    linear_term = get_linear_term_fn(boundary_x)

    feature_term_mean = float(feature_term.mean().item())
    linear_term_mean = float(linear_term.mean().item())
    total_mean = feature_term_mean + linear_term_mean

    if total_mean <= 0.0:
        feature_term_mean_share = float("nan")
        linear_term_mean_share = float("nan")
    else:
        feature_term_mean_share = feature_term_mean / total_mean
        linear_term_mean_share = linear_term_mean / total_mean

    return BoundaryTermDiagnostics(
        feature_term_quantile=float(th.quantile(feature_term, q=quantile).item()),
        linear_term_quantile=float(th.quantile(linear_term, q=quantile).item()),
        feature_term_mean=feature_term_mean,
        linear_term_mean=linear_term_mean,
        feature_term_mean_share=feature_term_mean_share,
        linear_term_mean_share=linear_term_mean_share,
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


def estimate_rho(
    lyap_model: nn.Module,
    config: RhoEstimationConfig,
    condition_evaluator: Callable[[th.Tensor], th.Tensor] | None = None,
    state_buffer: Any | None = None,
    device: th.device | int | str | None = None,
    generator: th.Generator | None = None,
    cex_quantile: float | None = None,
) -> tuple[BoundaryRhoEvaluation, th.Tensor]:
    """Estimate rho using boundary analysis and dynamic counterexample capping."""
    effective_cex_quantile = cex_quantile if cex_quantile is not None else config.cex_quantile

    eval_result, boundary_x = estimate_rho_from_boundary(
        lyap_model=lyap_model,
        config=config,
        device=device,
        generator=generator,
    )
    rho_boundary = eval_result.rho.rho
    rho_effective = rho_boundary
    cex_cap_val = None

    if state_buffer is not None and len(state_buffer) > 0 and condition_evaluator is not None:
        states_to_check = []
        if hasattr(state_buffer, "states") and state_buffer.states.numel() > 0:
            states_to_check.append(state_buffer.states)
        if hasattr(state_buffer, "cexs") and state_buffer.cexs.numel() > 0:
            states_to_check.append(state_buffer.cexs)

        if states_to_check:
            all_buffer_states = th.cat(states_to_check, dim=0)
            if config.origin_exclusion is not None:
                exclusion = th.as_tensor(config.origin_exclusion, dtype=all_buffer_states.dtype, device=all_buffer_states.device)
                outside_exclusion = ~th.all(th.abs(all_buffer_states) <= exclusion, dim=-1)
                all_buffer_states = all_buffer_states[outside_exclusion]

            if all_buffer_states.numel() > 0:
                with th.no_grad():
                    violations = condition_evaluator(all_buffer_states)
                    violations = violations.flatten()

                    violating_mask = violations > 1e-6

                    if violating_mask.any():
                        violating_states = all_buffer_states[violating_mask]
                        violating_v = lyap_model(violating_states).flatten()
                        cex_cap_val = float(th.quantile(violating_v, q=float(effective_cex_quantile)).item())
                        rho_effective = max(float(config.rho_min), min(rho_boundary, cex_cap_val))

    updated_eval = BoundaryRhoEvaluation(
        rho=BoundaryRhoEstimate(
            rho=float(rho_effective),
            boundary_quantile=eval_result.rho.boundary_quantile,
            boundary_mean=eval_result.rho.boundary_mean,
            cex_cap=cex_cap_val,
        ),
        terms=eval_result.terms,
    )
    return updated_eval, boundary_x
