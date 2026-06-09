from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch as th
import torch.nn as nn

from .buffer import BoundaryStateBuffer, DynamicStateBuffer
from .config import LyapunovTrainingConfig


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
    r_factor_fro_norm: float


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
    """Sample points on the boundary ∂B of the asymmetric box."""
    points = sample_uniform_box(sample_size, lb, ub, device, generator)
    
    # Randomly select a face dimension for each point to lie on.
    face_dims = th.randint(0, lb.numel(), (sample_size,), device=device, generator=generator)
    
    # 50/50 Chance: ub or lb
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
    if not all(hasattr(lyap_model, attr) for attr in ("feature_net", "x_star", "_pd_matrix")):
        return nan, nan, nan, nan, nan, nan, nan

    pd_matrix_fn = getattr(lyap_model, "_pd_matrix")
    if not callable(pd_matrix_fn):
        return nan, nan, nan, nan, nan, nan, nan

    x_star = getattr(lyap_model, "x_star")
    feature_net = getattr(lyap_model, "feature_net")

    x_star = x_star.to(dtype=boundary_x.dtype, device=boundary_x.device)
    x_star_batch = x_star.expand(boundary_x.shape[0], -1)
    phi_x = feature_net(boundary_x)
    phi_x_star = feature_net(x_star_batch)
    feature_term = th.abs(phi_x - phi_x_star).sum(dim=1)

    delta = boundary_x - x_star_batch
    pd_matrix = pd_matrix_fn()
    linear_term = th.abs(delta @ pd_matrix.transpose(0, 1)).sum(dim=1)
    total_mean = float((feature_term + linear_term).mean().item())
    feature_term_mean = float(feature_term.mean().item())
    linear_term_mean = float(linear_term.mean().item())

    if total_mean <= 0.0:
        feature_term_mean_share = nan
        linear_term_mean_share = nan
    else:
        feature_term_mean_share = feature_term_mean / total_mean
        linear_term_mean_share = linear_term_mean / total_mean

    r_factor_fro_norm = nan
    if hasattr(lyap_model, "r_factor"):
        r_factor_fro_norm = float(th.linalg.norm(getattr(lyap_model, "r_factor"), ord="fro").item())

    return (
        float(th.quantile(feature_term, q=quantile).item()),
        float(th.quantile(linear_term, q=quantile).item()),
        feature_term_mean,
        linear_term_mean,
        feature_term_mean_share,
        linear_term_mean_share,
        r_factor_fro_norm,
    )


def estimate_rho_from_boundary_diagnostics(
    lyap_model: nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
    boundary_buffer: BoundaryStateBuffer | None = None,
    cex_buffer: DynamicStateBuffer | None = None,
    generator: th.Generator | None = None,
) -> BoundaryRhoDiagnostics:
    """Estimate rho and expose boundary-term diagnostics for logging."""
    bounds = _bounds_tensor(config.state_bounds, device)
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

    if boundary_buffer is not None:
        boundary_buffer.update(boundary_x, value_fn=lyap_model)
        boundary_eval_x = boundary_buffer.states
    else:
        boundary_eval_x = boundary_x

    with th.no_grad():
        boundary_values = lyap_model(boundary_eval_x).flatten()
        boundary_quantile = float(th.quantile(boundary_values, q=float(config.rho_estimate_quantile)).item())
        boundary_mean = float(boundary_values.mean().item())
        (
            feature_term_quantile,
            linear_term_quantile,
            feature_term_mean,
            linear_term_mean,
            feature_term_mean_share,
            linear_term_mean_share,
            r_factor_fro_norm,
        ) = _boundary_term_diagnostics(
            lyap_model=lyap_model,
            boundary_x=boundary_eval_x,
            quantile=float(config.rho_estimate_quantile),
        )

    rho_boundary = max(config.rho_min, config.rho_growth_gamma * boundary_quantile)

    rho_cex = float("inf")
    if cex_buffer is not None and cex_buffer.cex_count > 0:
        with th.no_grad():
            v_cexs = lyap_model(cex_buffer.cexs)
            rho_cex = float(v_cexs.min().item()) * 0.99 
            
    rho = min(rho_boundary, rho_cex)
    return BoundaryRhoDiagnostics(
        rho=float(rho),
        boundary_quantile=boundary_quantile,
        boundary_mean=boundary_mean,
        feature_term_quantile=feature_term_quantile,
        linear_term_quantile=linear_term_quantile,
        feature_term_mean=feature_term_mean,
        linear_term_mean=linear_term_mean,
        feature_term_mean_share=feature_term_mean_share,
        linear_term_mean_share=linear_term_mean_share,
        r_factor_fro_norm=r_factor_fro_norm,
    )


def find_counter_examples(
    objective: Callable[[th.Tensor], th.Tensor] | nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
    generator: th.Generator | None = None,
) -> th.Tensor:
    """Find rho-gated training counterexamples via PGD on a minimization objective.

    The objective should follow the external training semantics: it must be
    negative on violating states within the current rho-sublevel set, zero on
    safe states inside the set, and positive outside the set.
    """
    bounds = _bounds_tensor(config.state_bounds, device)
    lbx, ubx = bounds[0], bounds[1]

    adv_states = sample_uniform_box(config.adversarial_samples, lbx, ubx, device, generator)
    step = config.adversarial_step_size * (ubx - lbx).unsqueeze(0)

    for _ in range(config.counterexample_steps):
        adv_states.requires_grad_(True)
        raw_objective = objective(adv_states)
        grad = th.autograd.grad(
            raw_objective.mean(),
            adv_states,
            retain_graph=False,
            create_graph=False,
        )[0]

        with th.no_grad():
            adv_states = adv_states - step * grad.sign()
            adv_states = project_to_box(adv_states, lbx, ubx)

    with th.no_grad():
        raw_objective = objective(adv_states)
        counter_mask = raw_objective.flatten() < -config.condition_tolerance

    return adv_states[counter_mask].clone().detach()