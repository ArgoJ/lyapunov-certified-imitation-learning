from __future__ import annotations

from typing import Sequence

import torch as th
import torch.nn as nn

from .config import LyapunovTrainingConfig


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
) -> th.Tensor:
    """Sample uniformly from the asymmetric box B = {x | lb <= x <= ub}."""
    u = th.rand(sample_size, lb.numel(), device=device)
    return u * (ub - lb) + lb


def sample_boundary_points(
    sample_size: int,
    lb: th.Tensor,
    ub: th.Tensor,
    device: th.device,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Sample points on the boundary ∂B of the asymmetric box."""
    points = sample_uniform_box(sample_size, lb, ub, device)
    
    # Randomly select a face dimension for each point to lie on.
    face_dims = th.randint(0, lb.numel(), (sample_size,), device=device)
    
    # 50/50 Chance: ub or lb
    is_ub = th.rand(sample_size, device=device) >= 0.5
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


def estimate_rho_from_boundary(
    lyap_model: nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
) -> float:
    """Estimate a sublevel value rho from low-Lyapunov points on the boundary of B."""
    bounds = _bounds_tensor(config.state_bounds, device)
    lbx, ubx = bounds[0], bounds[1]
    boundary_x, face_dims, is_ub = sample_boundary_points(
        sample_size=config.rho_boundary_samples,
        lb=lbx,
        ub=ubx,
        device=device,
    )
    refs = th.zeros_like(boundary_x) # TODO placeholder for reference state if needed in the future

    step = config.rho_step_size * (ubx - lbx).unsqueeze(0)
    for _ in range(config.rho_descent_steps):
        boundary_x.requires_grad_(True)
        
        e = boundary_x - refs
        boundary_values = lyap_model(e)
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

    with th.no_grad():
        min_boundary_value = lyap_model(boundary_x).min().item()
    rho = max(config.rho_min, config.rho_growth_gamma * min_boundary_value)
    return float(rho)


def find_counter_examples(
    verifier: nn.Module,
    config: LyapunovTrainingConfig,
    rho: float,
    kappa: float,
    device: th.device = th.device("cpu"),
) -> th.Tensor:
    """Find counterexamples via PGD on ReLU(verifier(x, rho, kappa))."""
    bounds = _bounds_tensor(config.state_bounds, device)
    lbx, ubx = bounds[0], bounds[1]
    
    # Convert scalar rho and kappa to tensors for the verifier's forward method
    rho_t = th.tensor([max(config.rho_min, float(rho))], dtype=th.float32, device=device)
    kappa_t = th.tensor([kappa], dtype=th.float32, device=device)

    adv_states = sample_uniform_box(config.adversarial_samples, lbx, ubx, device)
    step = config.adversarial_step_size * (ubx - lbx).unsqueeze(0)

    for _ in range(config.counterexample_steps):
        adv_states.requires_grad_(True)
        violation = th.relu(verifier(adv_states, rho_t, kappa_t))
        grad = th.autograd.grad(
            violation.mean(),
            adv_states,
            retain_graph=False,
            create_graph=False,
        )[0]

        with th.no_grad():
            adv_states = adv_states + step * grad.sign()
            adv_states = project_to_box(adv_states, lbx, ubx)

    with th.no_grad():
        violation = th.relu(verifier(adv_states, rho_t, kappa_t))
        counter_mask = violation.flatten() > config.condition_tolerance

    return adv_states[counter_mask].clone().detach()