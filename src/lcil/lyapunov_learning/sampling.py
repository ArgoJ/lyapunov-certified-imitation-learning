import math
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
import torch as th

def _bounds_tensor(
    state_bounds: NDArray | Sequence[float] | th.Tensor | Any, 
    device: th.device | int | str | None,
) -> th.Tensor:
    """Convert state bounds to a PyTorch tensor."""
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
    device: th.device | int | str | None,
    generator: th.Generator | None = None,
) -> th.Tensor:
    """Sample uniformly from the asymmetric box B = {x | lb <= x <= ub}."""
    u = th.rand(sample_size, lb.numel(), device=device, generator=generator)
    return u * (ub - lb) + lb


def sample_boundary_points(
    sample_size: int,
    lb: th.Tensor,
    ub: th.Tensor,
    device: th.device | int | str | None,
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


def sample_rejection_states(
    candidates: th.Tensor,
    scores: th.Tensor,
    target_count: int,
    sharpness: float = 10.0,
    generator: th.Generator | None = None,
) -> th.Tensor:
    """Subsamples candidates using rejection sampling based on scores."""
    weights = th.sigmoid(sharpness * scores)
    weights = weights + 1e-6 

    idx = th.multinomial(weights, target_count, replacement=False, generator=generator)
    return candidates[idx]


def sample_box_rejection_states(
    lb: th.Tensor,
    ub: th.Tensor,
    target_count: int,
    score_fn: Callable[[th.Tensor], th.Tensor],
    oversample_factor: int = 4,
    sharpness: float = 10.0,
    device: th.device | str = "cpu",
    generator: th.Generator | None = None,
) -> th.Tensor:
    """Sample states uniformly from a box and subsample based on a score function."""
    n_candidates = target_count * oversample_factor
    candidates = sample_uniform_box(n_candidates, lb, ub, device, generator)
    scores = score_fn(candidates)
    
    return sample_rejection_states(
        candidates=candidates, 
        scores=scores, 
        target_count=target_count, 
        sharpness=sharpness,
        generator=generator
    )


def sample_mixed_batch(
    regular_states: th.Tensor,
    cexs: th.Tensor,
    batch_size: int,
    cex_fraction: float,
    device: th.device | str = "cpu",
    generator: th.Generator | None = None,
) -> th.Tensor:
    """
    Uniformly samples a batch of states from the buffer, injecting a 
    portion of recent counterexamples if available.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    state_count = regular_states.shape[0]
    cex_count = cexs.shape[0] if cexs is not None else 0

    if cex_count == 0:
        batch_idx = th.randint(
            low=0,
            high=state_count,
            size=(batch_size,),
            device=device,
            generator=generator
        )
        return regular_states[batch_idx]

    max_inject = int(batch_size * cex_fraction)
    n_inject = min(cex_count, max_inject)

    cex_idx = th.randint(
        low=0,
        high=cex_count,
        size=(n_inject,),
        device=device,
        generator=generator
    )
    injected_cexs = cexs[cex_idx]

    n_regular = batch_size - n_inject
    reg_idx = th.randint(
        low=0,
        high=state_count,
        size=(n_regular,),
        device=device,
        generator=generator
    )
    sampled_regular_states = regular_states[reg_idx]

    batch = th.cat((injected_cexs, sampled_regular_states), dim=0)
    perm = th.randperm(batch.shape[0], device=batch.device, generator=generator)
    return batch[perm]


def sample_sobol_box(
    sample_size: int,
    lb: th.Tensor,
    ub: th.Tensor,
    sobol_engine: th.quasirandom.SobolEngine,
    device: th.device | str = "cpu",
) -> th.Tensor:
    """Sample a batch of states using Sobol sequence from the bounds."""
    rand_sobol = sobol_engine.draw(sample_size).to(device)
    return lb + rand_sobol * (ub - lb)

def sample_box_shell(
    sample_size: int,
    state_dim: int,
    center: th.Tensor,
    half_width: th.Tensor,
    min_scale: float = 0.6,
    max_scale: float = 1.0,
    device: th.device | str = "cpu",
    generator: th.Generator | None = None,
) -> th.Tensor:
    """Create diverse candidate states in a box-shaped shell."""
    # Uniformly sample in the unit hypercube [-1, 1]^N
    directions = th.rand(
        sample_size,
        state_dim,
        device=device,
        generator=generator,
    ) * 2.0 - 1.0
    
    # Project to the surface of the unit hypercube (L-infinity norm = 1)
    directions = directions / directions.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    
    # Scale randomly between min_scale and max_scale
    scales = th.rand(
        sample_size,
        1,
        device=device,
        generator=generator
    ) * (max_scale - min_scale) + min_scale
    
    z_candidates = directions * scales
    return z_candidates * half_width + center


def sample_ellipsoid_boundary(
    sample_size: int,
    state_dim: int,
    center: th.Tensor,
    half_width: th.Tensor,
    min_radius: float = 0.6,
    max_radius: float = 1.0,
    device: th.device | str = "cpu",
    generator: th.Generator | None = None,
) -> th.Tensor:
    """Create diverse candidate states near the boundary of the asymmetric box (ellipsoidal)."""
    directions = th.randn(
        sample_size,
        state_dim,
        device=device,
        generator=generator,
    )
    directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
    radii = th.rand(
        sample_size,
        1,
        device=device,
        generator=generator
    ) * (max_radius - min_radius) + min_radius
    z_candidates = directions * radii  # between min_radius and max_radius in random directions
    return z_candidates * half_width + center


def sample_axis_antiphase_states(
    sample_size: int,
    lb: th.Tensor | Sequence[float],
    ub: th.Tensor | Sequence[float],
    scale_factor: float = 1.0,
    device: th.device | int | str | None = "cpu",
    generator: th.Generator | None = None,
) -> th.Tensor:
    """Sample states on coordinate axes and planes with alternating signs from training bounds.

    Generates sparse, axis-aligned states with Rademacher (opposing) signs
    and amplitudes scaled from the training bounds.
    This directly targets coordinate cancellation vulnerabilities and axis-aligned decrease
    violations without requiring system-specific domain knowledge.

    Parameters
    ----------
    sample_size : int
        Number of states to sample.
    lb : th.Tensor | Sequence[float]
        Lower bounds of the training domain.
    ub : th.Tensor | Sequence[float]
        Upper bounds of the training domain.
    scale_factor : float, optional
        Scale factor relative to training bounds defining the upper sampling amplitude,
        by default 1.0.
    device : th.device | int | str | None, optional
        Torch device on which to generate states, by default "cpu".
    generator : th.Generator | None, optional
        Random number generator for reproducible sampling, by default None.

    Returns
    -------
    th.Tensor
        Sampled states of shape (sample_size, state_dim).
    """
    lb_t = th.as_tensor(lb, dtype=th.float32, device=device)
    ub_t = th.as_tensor(ub, dtype=th.float32, device=device)
    nx = lb_t.numel()

    if sample_size <= 0:
        return th.empty((0, nx), dtype=th.float32, device=device)

    half_width = 0.5 * (ub_t - lb_t)
    max_r = float(scale_factor) * half_width

    effective_min = 1e-4 * max_r
    max_r = th.maximum(max_r, effective_min * 1.01)

    # 1. Random Rademacher signs (+1 / -1) covering all 2^nx orthants
    signs = (th.randint(0, 2, (sample_size, nx), device=device, generator=generator) * 2 - 1).float()

    # 2. Magnitudes scaled logarithmically from effective_min to max_r
    #    This ensures scale-invariant coverage across all decades from the origin
    #    exclusion boundary up to the full training bounds.
    u = th.rand(sample_size, nx, device=device, generator=generator)
    radii = effective_min * ((max_r / effective_min) ** u)

    # 3. Structured sparsity: sample across 1D axes, 2D planes, and higher-D subspaces
    if nx <= 2:
        k_vals = th.randint(1, nx + 1, (sample_size,), device=device, generator=generator)
    else:
        probs = th.zeros(nx, device=device)
        probs[0] = 0.25  # 25% pure 1D axes
        probs[1] = 0.50  # 50% 2D coordinate interaction planes
        probs[2:] = 0.25 / (nx - 2)  # 25% 3D / higher-D subspaces
        k_vals = th.multinomial(probs, sample_size, replacement=True, generator=generator) + 1

    perms = th.argsort(th.rand(sample_size, nx, device=device, generator=generator), dim=-1)
    masks = perms < k_vals.unsqueeze(-1)

    states = masks.float() * radii * signs
    return th.clamp(states, min=lb_t, max=ub_t)


def sample_eigenvector_antiphase_states(
    sample_size: int,
    lb: th.Tensor | Sequence[float],
    ub: th.Tensor | Sequence[float],
    eigenvector: th.Tensor | Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    scale_factor: float = 1.0,
    min_radius: float = 1e-4,
    center: th.Tensor | Sequence[float] | None = None,
    device: th.device | int | str | None = "cpu",
    generator: th.Generator | None = None,
) -> th.Tensor:
    """Sample states along closed-loop eigenvector ray directions (antiphase modes).

    Generates states distributed along the ray defined by the provided eigenvector,
    with logarithmic spacing from near the origin out to the domain boundary. Both
    positive and negative ray directions (+v and -v) are sampled.

    Parameters
    ----------
    sample_size : int
        Number of states to sample.
    lb : th.Tensor | Sequence[float]
        Lower bounds of the state domain.
    ub : th.Tensor | Sequence[float]
        Upper bounds of the state domain.
    eigenvector : th.Tensor | Sequence[float] | Sequence[Sequence[float]] | np.ndarray
        Eigenvector direction(s) of shape (nx,) or (num_vecs, nx).
    scale_factor : float, optional
        Scale factor relative to domain boundary defining the upper sampling amplitude,
        by default 1.0.
    min_radius : float, optional
        Fraction of maximum radius defining the lower logarithmic sampling boundary,
        by default 1e-4.
    center : th.Tensor | Sequence[float] | None, optional
        Center of expansion (e.g. equilibrium state x*), by default the origin (0).
    device : th.device | int | str | None, optional
        Torch device on which to generate states, by default "cpu".
    generator : th.Generator | None, optional
        Random number generator for reproducible sampling, by default None.

    Returns
    -------
    th.Tensor
        Sampled states of shape (sample_size, nx).
    """
    lb_t = th.as_tensor(lb, dtype=th.float32, device=device)
    ub_t = th.as_tensor(ub, dtype=th.float32, device=device)
    nx = lb_t.numel()

    if sample_size <= 0:
        return th.empty((0, nx), dtype=th.float32, device=device)

    vecs_t = th.as_tensor(eigenvector, dtype=th.float32, device=device)
    if vecs_t.ndim == 1:
        vecs_t = vecs_t.unsqueeze(0)

    norms = th.linalg.norm(vecs_t, dim=-1, keepdim=True)
    vecs_t = vecs_t / th.where(norms > 1e-8, norms, th.ones_like(norms))
    num_vecs = vecs_t.shape[0]

    c_t = th.zeros(nx, dtype=th.float32, device=device) if center is None else th.as_tensor(center, dtype=th.float32, device=device)
    delta_ub = ub_t - c_t
    delta_lb = lb_t - c_t

    indices = th.randint(0, num_vecs, (sample_size,), device=device, generator=generator)
    signs = (th.randint(0, 2, (sample_size, 1), device=device, generator=generator) * 2 - 1).float()
    dirs = signs * vecs_t[indices]

    pos_limits = th.where(dirs > 1e-8, delta_ub / dirs, th.full_like(dirs, float("inf")))
    neg_limits = th.where(dirs < -1e-8, delta_lb / dirs, th.full_like(dirs, float("inf")))
    limits = th.minimum(pos_limits, neg_limits)
    r_boundary = th.amin(limits, dim=-1)

    r_max = float(scale_factor) * r_boundary
    effective_min = float(min_radius) * r_max
    effective_min = th.maximum(effective_min, th.full_like(effective_min, 1e-9))
    r_max = th.maximum(r_max, effective_min * 1.01)

    u = th.rand(sample_size, device=device, generator=generator)
    radii = effective_min * ((r_max / effective_min) ** u)

    states = c_t + radii.unsqueeze(-1) * dirs
    return th.clamp(states, min=lb_t, max=ub_t)


