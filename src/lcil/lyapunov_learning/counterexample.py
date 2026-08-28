from __future__ import annotations

import logging
import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .config import LyapunovTrainingConfig
from .sampling import (
    _bounds_tensor,
    project_to_box,
    sample_uniform_box,
    sample_boundary_points,
    project_to_boundary_faces,
)
from ..utils import timeit

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
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
    generator: th.Generator | None = None,
) -> tuple[BoundaryRhoEvaluation, th.Tensor]:
    """Estimate rho and expose boundary-term diagnostics for logging."""
    bounds = _bounds_tensor(config.train_bounds, device)
    lbx, ubx = bounds[0], bounds[1]
    boundary_x, face_dims, is_ub = sample_boundary_points(
        sample_size=config.rho_estimation_samples,
        lb=lbx,
        ub=ubx,
        device=device,
        generator=generator,
    )

    # n = ubx.shape[0]
    # center = (lbx + ubx) / 2.0
    # face_centers_tensor = center.repeat(2 * n, 1)
    # bounds_interleaved = th.stack([lbx, ubx], dim=1).flatten()
    # mask = th.eye(n, dtype=th.bool, device=device).repeat_interleave(2, dim=0)
    # face_centers_tensor[mask] = bounds_interleaved
    
    # # Batch aus Zufallspunkten und exakten Mittelpunkten zusammenbauen
    # boundary_eval_x = th.cat([boundary_x, face_centers_tensor], dim=0)

    step = config.rho_step_size * (ubx - lbx).unsqueeze(0)
    with th.no_grad():
        best_boundary_x = boundary_x.clone()
        best_boundary_values = lyap_model(boundary_x).flatten()

    for _ in range(config.rho_descent_steps):
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
        boundary_quantile = float(th.quantile(boundary_values, q=float(config.rho_estimate_quantile)).item())
        boundary_mean = float(boundary_values.mean().item())
        
        if config.enable_diagnosis:
            term_diagnostics = _boundary_term_diagnostics(
                lyap_model=lyap_model,
                boundary_x=boundary_eval_x,
                quantile=float(config.rho_estimate_quantile),
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
    config: LyapunovTrainingConfig,
    condition_evaluator: Callable[[th.Tensor], th.Tensor] | None = None,
    state_buffer: Any | None = None,
    device: th.device = th.device("cpu"),
    generator: th.Generator | None = None,
    cex_quantile: float = 0.05,
) -> tuple[BoundaryRhoEvaluation, th.Tensor]:
    """Estimate rho using boundary analysis and dynamic counterexample capping."""
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
            with th.no_grad():
                violations = condition_evaluator(all_buffer_states)
                violations = violations.flatten()

                violating_mask = violations > 1e-6

                if violating_mask.any():
                    violating_states = all_buffer_states[violating_mask]
                    violating_v = lyap_model(violating_states).flatten()
                    cex_cap_val = float(th.quantile(violating_v, q=float(cex_quantile)).item())
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



def find_counter_examples(
    objective: Callable[[th.Tensor], th.Tensor],
    condition_evaluator: Callable[[th.Tensor], tuple[th.Tensor, th.Tensor]],
    config: LyapunovTrainingConfig,
    initial_states: th.Tensor,
    device: th.device = th.device("cpu"),
) -> tuple[th.Tensor, th.Tensor]:
    """Find rho-gated training counterexamples via PGD.

    Performs adversarial mining using the provided objective,
    then filters the results using the condition evaluator to return 
    only true counterexamples strictly inside the current rho-sublevel set.
    """
    bounds = _bounds_tensor(config.train_bounds, device)
    lbx, ubx = bounds[0], bounds[1]
    adv_states = initial_states.clone().to(device=device)
    step = config.cex_step_size * (ubx - lbx).unsqueeze(0)

    with th.no_grad():
        best_states = adv_states.clone()
        best_violations = th.zeros(
            adv_states.shape[0],
            dtype=adv_states.dtype,
            device=device,
        )
        init_violations, init_mask = condition_evaluator(adv_states)
        init_violations = init_violations.flatten()
        best_violations[init_mask] = init_violations[init_mask]

    for _ in range(config.cex_descent_steps):
        adv_states.requires_grad_(True)
        raw_objective = objective(adv_states)

        grad = th.autograd.grad(
            raw_objective.mean(),
            adv_states,
            retain_graph=False,
            create_graph=False,
        )[0]

        with th.no_grad():
            candidate_states = adv_states - step * grad.sign()
            candidate_states = project_to_box(candidate_states, lbx, ubx)
            candidate_violations, candidate_mask = condition_evaluator(candidate_states)
            candidate_violations = candidate_violations.flatten()

            improved = candidate_mask & (candidate_violations > best_violations)

            best_states[improved] = candidate_states[improved]
            best_violations[improved] = candidate_violations[improved]

            adv_states = candidate_states

    with th.no_grad():
        exclusion = th.as_tensor(config.origin_exclusion, dtype=best_states.dtype, device=device)
        inside_exclusion = th.all(th.abs(best_states) <= exclusion, dim=-1)
        counter_mask = (best_violations > 0.0) & (~inside_exclusion)

    cex_states = best_states[counter_mask].clone().detach()
    cex_violations = best_violations[counter_mask].clone().detach()
    return cex_states, cex_violations