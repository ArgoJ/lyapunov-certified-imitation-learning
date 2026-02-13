from __future__ import annotations

from typing import Sequence

import torch as th
import torch.nn as nn

from .verifier_models import ClosedLoopLyapunovConditionVerifier
from ..lyapunov_learning.train_config import LyapunovTrainingConfig


def _bounds_tensor(state_bounds: Sequence[float], device: th.device) -> th.Tensor:
    bounds = th.as_tensor(state_bounds, dtype=th.float32, device=device)
    if bounds.ndim != 1:
        raise ValueError("state_bounds must be a flat sequence of per-dimension bounds.")
    return bounds


def project_to_box(state: th.Tensor, bounds: th.Tensor) -> th.Tensor:
    """Project states to the box B = {x | -bounds <= x <= bounds}."""
    return th.maximum(th.minimum(state, bounds), -bounds)


def sample_uniform_box(
    sample_size: int,
    bounds: th.Tensor,
    device: th.device,
) -> th.Tensor:
    """Sample uniformly from the box B = {x | -bounds <= x <= bounds}."""
    delta = th.zeros(sample_size, bounds.numel(), device=device).uniform_(-1.0, 1.0)
    return delta * bounds


def sample_boundary_points(
    sample_size: int,
    bounds: th.Tensor,
    device: th.device,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Sample points on the boundary ∂B and return fixed boundary faces."""
    points = sample_uniform_box(sample_size, bounds, device)
    face_dims = th.randint(0, bounds.numel(), (sample_size,), device=device)
    face_sign = th.where(
        th.rand(sample_size, device=device) >= 0.5,
        th.ones(sample_size, device=device),
        -th.ones(sample_size, device=device),
    )
    points[th.arange(sample_size, device=device), face_dims] = (
        face_sign * bounds[face_dims]
    )
    return points, face_dims, face_sign


def project_to_boundary_faces(
    points: th.Tensor,
    bounds: th.Tensor,
    face_dims: th.Tensor,
    face_sign: th.Tensor,
) -> th.Tensor:
    """Project points onto the original boundary faces after a gradient step."""
    points = project_to_box(points, bounds)
    points[th.arange(points.shape[0], device=points.device), face_dims] = (
        face_sign * bounds[face_dims]
    )
    return points


def estimate_rho_from_boundary(
    lyap_model: nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
) -> float:
    """Estimate a sublevel value rho from low-Lyapunov points on the boundary of B."""
    if len(config.state_bounds) != config.state_dim:
        raise ValueError("state_bounds must match state_dim.")

    bounds = _bounds_tensor(config.state_bounds, device)
    boundary_points, face_dims, face_sign = sample_boundary_points(
        sample_size=config.rho_boundary_samples,
        bounds=bounds,
        device=device,
    )

    step = config.rho_step_size * bounds.unsqueeze(0)
    for _ in range(config.rho_descent_steps):
        boundary_points.requires_grad_(True)
        boundary_values = lyap_model(boundary_points)
        grad = th.autograd.grad(
            boundary_values.mean(),
            boundary_points,
            retain_graph=False,
            create_graph=False,
        )[0]

        with th.no_grad():
            boundary_points = boundary_points - step * grad.sign()
            boundary_points = project_to_boundary_faces(
                boundary_points,
                bounds=bounds,
                face_dims=face_dims,
                face_sign=face_sign,
            )

    with th.no_grad():
        min_boundary_value = lyap_model(boundary_points).min().item()
    rho = max(config.rho_min, config.rho_growth_gamma * min_boundary_value)
    return float(rho)


def find_counter_examples(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    config: LyapunovTrainingConfig,
    rho: float,
    device: th.device = th.device("cpu"),
) -> th.Tensor:
    """Find counterexamples via PGD on ReLU(verifier(x))."""
    if len(config.state_bounds) != config.state_dim:
        raise ValueError("state_bounds must match state_dim.")

    bounds = _bounds_tensor(config.state_bounds, device)
    rho_value = max(config.rho_min, float(rho))
    verifier = ClosedLoopLyapunovConditionVerifier(
        policy_model=policy_model,
        lyap_model=lyap_model,
        dyn_model=dyn_model,
        bounds=bounds,
        kappa=config.kappa,
        invariance_weight=config.invariance_weight,
        rho=rho_value,
    ).to(device)
    verifier.set_rho(rho_value)

    adv_states = sample_uniform_box(config.adversarial_samples, bounds, device)
    step = config.adversarial_step_size * bounds.unsqueeze(0)

    for _ in range(config.counterexample_steps):
        adv_states.requires_grad_(True)
        violation = th.relu(verifier(adv_states))
        grad = th.autograd.grad(
            violation.mean(),
            adv_states,
            retain_graph=False,
            create_graph=False,
        )[0]

        with th.no_grad():
            adv_states = adv_states + step * grad.sign()
            adv_states = project_to_box(adv_states, bounds)

    with th.no_grad():
        violation = th.relu(verifier(adv_states))
        counter_mask = violation.flatten() > config.condition_tolerance

    return adv_states[counter_mask].clone().detach()
