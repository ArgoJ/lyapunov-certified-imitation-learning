import torch as th
from typing import Sequence, Callable

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
