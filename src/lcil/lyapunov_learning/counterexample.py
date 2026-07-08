from __future__ import annotations

import logging
import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Callable, Sequence

from .config import LyapunovTrainingConfig
from ..utils import timeit

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundaryRhoDiagnostics:
    """Diagnostics for the boundary-based rho estimate."""

    rho: float
    boundary_quantile: float
    boundary_mean: float
    feature_term_quantile: float
    linear_term_quantile: float
    feature_term_mean: float
    linear_term_mean: float
    feature_term_mean_share: float
    linear_term_mean_share: float


def _bounds_tensor(state_bounds: Sequence[float], device: th.device) -> th.Tensor:
    bounds = th.as_tensor(state_bounds, dtype=th.float32, device=device)
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("state_bounds must be a sequence of shape (2, nx) [lb, ub].")
    return bounds


def project_to_box(state: th.Tensor, lb: th.Tensor, ub: th.Tensor) -> th.Tensor:
    """Project states to the asymmetric box B = {x | lb <= x <= ub}."""
    return th.maximum(th.minimum(state, ub), lb)


def sample_uniform_box(
    sample_size: int,
    lb: th.Tensor,
    ub: th.Tensor,
    device: th.device,
    generator: th.Generator | None = None,
) -> th.Tensor:
    """Sample uniformly from the asymmetric box B = {x | lb <= x <= ub}."""
    u = th.rand(sample_size, lb.numel(), device=device, generator=generator)
    return u * (ub - lb) + lb


def sample_boundary_points(
    sample_size: int,
    lb: th.Tensor,
    ub: th.Tensor,
    device: th.device,
    generator: th.Generator | None = None,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Sample points uniformly distributed over the actual surface area of the box."""
    points = sample_uniform_box(sample_size, lb, ub, device, generator)
    widths = ub - lb
    face_areas = th.prod(widths) / widths
    probs = face_areas / th.sum(face_areas)
    
    # Choose the dimensions weighted by their actual geometric area
    face_dims = th.multinomial(probs, sample_size, replacement=True, generator=generator)
    
    # 50/50 Chance for Upper or Lower Bound
    is_ub = th.rand(sample_size, device=device, generator=generator) >= 0.5
    batch_idx = th.arange(sample_size, device=device)
    points[batch_idx, face_dims] = th.where(is_ub, ub[face_dims], lb[face_dims])
    
    return points, face_dims, is_ub


def project_to_boundary_faces(
    points: th.Tensor,
    lb: th.Tensor,
    ub: th.Tensor,
    face_dims: th.Tensor,
    is_ub: th.Tensor,
) -> th.Tensor:
    """Project points onto the original boundary faces after a gradient step."""
    points = project_to_box(points, lb, ub)
    batch_idx = th.arange(points.shape[0], device=points.device)
    points[batch_idx, face_dims] = th.where(is_ub, ub[face_dims], lb[face_dims])
    return points


def _boundary_term_diagnostics(
    lyap_model: nn.Module,
    boundary_x: th.Tensor,
    quantile: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Return Lyapunov term diagnostics when the model exposes them."""
    nan = float("nan")
    if not all(hasattr(lyap_model, attr) 
        for attr in ("feature_net", "x_star", "_pd_matrix", "get_feature_term", "get_linear_term")):
        return nan, nan, nan, nan, nan, nan, nan

    pd_matrix_fn = getattr(lyap_model, "_pd_matrix")
    get_feature_term_fn = getattr(lyap_model, "get_feature_term")
    get_linear_term_fn = getattr(lyap_model, "get_linear_term")
    if not callable(pd_matrix_fn) or not callable(get_feature_term_fn) or \
       not callable(get_linear_term_fn):
        return nan, nan, nan, nan, nan, nan, nan

    feature_term = get_feature_term_fn(boundary_x)
    linear_term = get_linear_term_fn(boundary_x)
    total_mean = float((feature_term + linear_term).mean().item())
    feature_term_mean = float(feature_term.mean().item())
    linear_term_mean = float(linear_term.mean().item())

    if total_mean <= 0.0:
        feature_term_mean_share = nan
        linear_term_mean_share = nan
    else:
        feature_term_mean_share = feature_term_mean / total_mean
        linear_term_mean_share = linear_term_mean / total_mean

    return (
        float(th.quantile(feature_term, q=quantile).item()),
        float(th.quantile(linear_term, q=quantile).item()),
        feature_term_mean,
        linear_term_mean,
        feature_term_mean_share,
        linear_term_mean_share,
    )


def estimate_rho_from_boundary(
    lyap_model: nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
    generator: th.Generator | None = None,
) -> tuple[BoundaryRhoDiagnostics, th.Tensor]:
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

    step = config.rho_step_size * (ubx - lbx).unsqueeze(0)
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
            boundary_x = boundary_x - step * grad.sign()
            boundary_x = project_to_boundary_faces(
                boundary_x,
                lb=lbx,
                ub=ubx,
                face_dims=face_dims,
                is_ub=is_ub,
            )

    boundary_eval_x = boundary_x.detach()
    with th.no_grad():
        boundary_values = lyap_model(boundary_eval_x).flatten()
        boundary_quantile = float(th.quantile(boundary_values, q=float(config.rho_estimate_quantile)).item())
        boundary_mean = float(boundary_values.mean().item())
        
        term_diagnostics = _boundary_term_diagnostics(
            lyap_model=lyap_model,
            boundary_x=boundary_eval_x,
            quantile=float(config.rho_estimate_quantile),
        )

    rho_boundary = max(config.rho_min, config.rho_growth_gamma * boundary_quantile)
    diagnostics = BoundaryRhoDiagnostics(
        rho=float(rho_boundary),
        boundary_quantile=boundary_quantile,
        boundary_mean=boundary_mean,
        feature_term_quantile=term_diagnostics[0],
        linear_term_quantile=term_diagnostics[1],
        feature_term_mean=term_diagnostics[2],
        linear_term_mean=term_diagnostics[3],
        feature_term_mean_share=term_diagnostics[4],
        linear_term_mean_share=term_diagnostics[5],
    )
    return diagnostics, boundary_eval_x


def find_counter_examples(
    objective: Callable[[th.Tensor], th.Tensor] | nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
    generator: th.Generator | None = None,
) -> tuple[th.Tensor, th.Tensor]:
    """Find rho-gated training counterexamples via PGD on a minimization objective.

    The objective should follow the external training semantics: it must be
    negative on violating states within the current rho-sublevel set, zero on
    safe states inside the set, and positive outside the set.
    """
    bounds = _bounds_tensor(config.train_bounds, device)
    lbx, ubx = bounds[0], bounds[1]

    adv_states = sample_uniform_box(config.adversarial_samples, lbx, ubx, device, generator)
    step = config.adversarial_step_size * (ubx - lbx).unsqueeze(0)

    with th.no_grad():
        init_objective = objective(adv_states).flatten()
        best_states = adv_states.clone()
        best_objective = init_objective.clone()

    for _ in range(config.cex_steps):
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

            candidate_objective = objective(candidate_states).flatten()

            improved = candidate_objective < best_objective
            best_states[improved] = candidate_states[improved]
            best_objective[improved] = candidate_objective[improved]

            adv_states = candidate_states

    with th.no_grad():
        # Filter cex origin exclusion
        exclusion = th.as_tensor(config.origin_exclusion, dtype=best_states.dtype, device=device)
        inside_exclusion = th.all(th.abs(best_states) <= exclusion, dim=-1)
        cex_tol = 1e-9
        counter_mask = (best_objective < -cex_tol) & (~inside_exclusion)

    cex_states = best_states[counter_mask].clone().detach()
    cex_violations = (-best_objective[counter_mask]).clamp_min(0.0).clone().detach()
    return cex_states, cex_violations