from __future__ import annotations

import logging

from collections.abc import Callable
from dataclasses import dataclass
from math import gamma, pi

import numpy as np
import torch as th
from numpy.typing import NDArray
from scipy.stats import norm, qmc

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class LevelSetEstimate:
    """Deterministic star-shaped size estimate for ``V(x) <= rho``.

    The estimate is computed from deterministic rays whose directions lie on the
    unit sphere. Along each direction, the method approximates the first radial
    exit of the origin-connected component of the sublevel set. The resulting
    measure is therefore a star-shaped approximation around the origin, not the
    exact geometric volume of an arbitrary neural-network sublevel set.
    """

    rho: float
    num_states: int
    num_directions: int
    directions: NDArray
    radii: NDArray
    measure: float
    unit_sphere_surface_area: float
    max_radius: float
    truncated_mask: NDArray

    @property
    def area(self) -> float:
        """Alias for ``measure`` retained for historical naming."""
        return self.measure

    @property
    def truncated(self) -> bool:
        """Whether any direction failed to leave the sublevel set before ``max_radius``."""
        return bool(np.any(self.truncated_mask))

    @property
    def truncated_fraction(self) -> float:
        """Fraction of directions truncated at ``max_radius``."""
        return float(np.mean(self.truncated_mask))


def _unit_sphere_surface_area(num_states: int) -> float:
    """Return the surface area of the unit sphere ``S^(n-1)`` in ``R^n``.
    $$ |S^{n-1}| = \frac{2 \pi^{\frac{n}{2}}}{\Gamma(\frac{n}{2})} $$
    
    Parameters
    ----------
    num_states: int
        The number of dimensions of the state space (i.e., n).
    
    Returns
    -------
    float
        The surface area of the unit sphere in the given number of dimensions.
    """
    if num_states < 1:
        raise ValueError(f"num_states must be positive, got {num_states}.")
    return 2.0 * pi ** (0.5 * num_states) / gamma(0.5 * num_states)


def _sublevel_surface_area(
    radii: NDArray,
    unit_sphere_surface_area: float,
    num_states: int,
) -> float:
    r"""Compute the nD measure of a star-shaped set from ray lengths and unit sphere surface area.
    $$ \hat{\mu}(\Omega_{\rho}) = \frac{|S^{n-1}|}{n} \frac{1}{N} \sum_{i=1}^{N} r_i^n $$
    
    Parameters
    ----------
    radii: NDArray
        The lengths of the rays from the origin to the boundary of the set in each direction.
    unit_sphere_surface_area: float
        The surface area of the unit sphere in the given number of dimensions.
    num_states: int
        The number of dimensions of the state space.
    
    Returns
    -------
    float
        The nD measure of the star-shaped set defined by the given ray lengths.
    """
    return unit_sphere_surface_area * float(np.mean(radii**num_states)) / float(num_states)


def _sample_unit_sphere_directions(num_states: int, num_directions: int) -> NDArray:
    """Sample deterministic directions on the unit sphere via a Sobol design."""
    if num_states < 1:
        raise ValueError(f"num_states must be positive, got {num_states}.")
    if num_directions < 1:
        raise ValueError(f"num_directions must be positive, got {num_directions}.")

    eps = np.finfo(float).eps
    m = int(np.ceil(np.log2(max(num_directions + 4, 2))))

    while True:
        engine = qmc.Sobol(d=num_states, scramble=False)
        raw_uniform = engine.random_base2(m=m)
        interior_mask = np.all((raw_uniform > 0.0) & (raw_uniform < 1.0), axis=1)
        interior_uniform = raw_uniform[interior_mask]

        gaussian = norm.ppf(np.clip(interior_uniform, eps, 1.0 - eps))
        norms = np.linalg.norm(gaussian, axis=1)
        valid_mask = norms > eps
        directions = gaussian[valid_mask] / norms[valid_mask, None]

        if directions.shape[0] >= num_directions:
            return directions[:num_directions]
        m += 1


def _find_level_ray_intersections(
    lyapunov_fn: Callable[[th.Tensor], th.Tensor],
    c_level: float,
    directions: NDArray,
    *,
    device: th.device | None = None,
    initial_radius: float = 1.0,
    growth_factor: float = 2.0,
    max_radius: float = 1e6,
    max_bisection_steps: int = 60,
) -> tuple[NDArray, NDArray]:
    """Approximate the first radial exit of ``V(x) <= c_level`` along multiple rays.
    
    Parameters
    ----------
    lyapunov_fn: Callable[[th.Tensor], th.Tensor]
        The Lyapunov function to evaluate, accepting Torch tensors.
    c_level: float
        The sublevel set value to intersect.
    directions: NDArray
        The direction vectors along which to search for the level set intersection.
    device: th.device | None, optional
        The device to use for Torch tensors.
    initial_radius: float, optional
        The initial radius for the search.
    growth_factor: float, optional
        The factor by which to grow the search radius.
    max_radius: float, optional
        The maximum radius to search.
    max_bisection_steps: int, optional
        The maximum number of bisection steps to perform.

    Returns
    -------
    tuple[NDArray, NDArray]
        A tuple containing the estimated radii at which the level set is intersected for each direction 
        and a boolean array indicating which directions were truncated at max_radius.
    """
    num_directions = directions.shape[0]
    directions_ts = th.as_tensor(directions, dtype=th.float32, device=device)

    if initial_radius <= 0.0:
        raise ValueError(f"initial_radius must be positive, got {initial_radius}.")
    if growth_factor <= 1.0:
        raise ValueError(f"growth_factor must exceed 1, got {growth_factor}.")
    if max_radius <= 0.0:
        raise ValueError(f"max_radius must be positive, got {max_radius}.")
    if initial_radius > max_radius:
        raise ValueError(
            f"initial_radius ({initial_radius}) must not exceed max_radius ({max_radius})."
        )
    if max_bisection_steps < 0:
        raise ValueError(
            f"max_bisection_steps must be non-negative, got {max_bisection_steps}."
        )

    active_mask = th.ones(num_directions, dtype=th.bool, device=device)
    low = th.zeros(num_directions, 1, dtype=th.float32, device=device)
    high = th.full((num_directions, 1), initial_radius, dtype=th.float32, device=device)

    max_growth_steps = int(np.ceil(np.log(max_radius / initial_radius) / np.log(growth_factor))) + 2
    for _ in range(max_growth_steps):
        if not active_mask.any():
            break
        points = high[active_mask] * directions_ts[active_mask]

        with th.no_grad():
            values = lyapunov_fn(points)

        finite_values = th.isfinite(values)
        exceeded = ((~finite_values) | (values > c_level)).squeeze(-1)

        active_indices = active_mask.nonzero(as_tuple=True)[0]
        exceeded_indices = active_indices[exceeded]
        active_mask[exceeded_indices] = False

        not_exceeded_indices = active_indices[~exceeded]
        if len(not_exceeded_indices) > 0:
            monotonic_points = (
                high[not_exceeded_indices].detach() * directions_ts[not_exceeded_indices]
            ).requires_grad_(True)
            monotonic_values = lyapunov_fn(monotonic_points)
            monotonic_gradients = th.autograd.grad(
                outputs=monotonic_values,
                inputs=monotonic_points,
                grad_outputs=th.ones_like(monotonic_values),
                create_graph=False,
                retain_graph=False,
            )[0]
            dV_dr_growth = (
                monotonic_gradients * directions_ts[not_exceeded_indices]
            ).sum(dim=1, keepdim=True)
            if (dV_dr_growth < 0).any():
                raise ValueError(
                    "Monotonicity violated along at least one ray; the star-shaped level-set assumption does not hold for this rho."
                )

        low[not_exceeded_indices] = high[not_exceeded_indices]
        high[not_exceeded_indices] = th.clamp(high[not_exceeded_indices] * growth_factor, max=max_radius)

        reached_max = (high[not_exceeded_indices] >= max_radius).squeeze(-1)
        truncated_indices = not_exceeded_indices[reached_max]
        active_mask[truncated_indices] = False

    truncated_mask = (high >= max_radius).squeeze(-1)

    active_roots = ~truncated_mask
    curr_r = 0.5 * (low + high)
    curr_r[~active_roots] = max_radius

    for _ in range(max_bisection_steps):
        if not active_roots.any():
            break

        points = curr_r[active_roots] * directions_ts[active_roots]
        points.requires_grad_(True)
        values = lyapunov_fn(points)

        finite_mask = th.isfinite(values).squeeze(-1)
        active_indices = active_roots.nonzero(as_tuple=True)[0]

        if (~finite_mask).any():
            nonfinite_indices = active_indices[~finite_mask]
            high[nonfinite_indices] = curr_r[nonfinite_indices]
            curr_r[nonfinite_indices] = 0.5 * (low[nonfinite_indices] + high[nonfinite_indices])

        if not finite_mask.any():
            continue

        finite_values = values[finite_mask]
        finite_active_indices = active_indices[finite_mask]

        gradients_all = th.autograd.grad(
            outputs=finite_values,
            inputs=points,
            grad_outputs=th.ones_like(finite_values),
            create_graph=False,
            retain_graph=False,
        )[0]
        gradients = gradients_all[finite_mask]

        dV_dr = (gradients * directions_ts[finite_active_indices]).sum(dim=1, keepdim=True)
        non_monotonic = (dV_dr < 0).squeeze(-1)
        if non_monotonic.any():
            raise ValueError(
                "Monotonicity violated along at least one ray; the star-shaped level-set assumption does not hold for this rho."
            )

        err_detached = finite_values.detach() - c_level

        is_pos = err_detached > 0
        high_next = th.where(is_pos, curr_r[finite_active_indices], high[finite_active_indices])
        low_next = th.where(~is_pos, curr_r[finite_active_indices], low[finite_active_indices])

        low[finite_active_indices] = low_next
        high[finite_active_indices] = high_next
        curr_r[finite_active_indices] = 0.5 * (low_next + high_next)

        converged = (
            (th.abs(err_detached).squeeze(-1) < 1e-5)
            | ((high_next - low_next).squeeze(-1) < 1e-5)
        )
        converged_indices = finite_active_indices[converged]
        active_roots[converged_indices] = False

    return curr_r.squeeze(-1).cpu().numpy(), truncated_mask.cpu().numpy()


def estimate_level_set_area(
    lyapunov_fn: Callable[[th.Tensor], th.Tensor],
    rho: float,
    *,
    num_states: int,
    num_directions: int = 512,
    device: th.device | None = None,
    initial_radius: float = 1.0,
    growth_factor: float = 2.0,
    max_radius: float = 1e6,
    max_bisection_steps: int = 60,
) -> LevelSetEstimate:
    """Estimate the star-shaped nD measure of ``V(x) <= rho`` from sphere rays.

    Parameters
    ----------
    lyapunov_fn: Callable[[th.Tensor], th.Tensor]
        The Lyapunov function to evaluate, accepting Torch tensors.
    rho: float
        The sublevel set value to estimate.
    num_states: int
        The number of dimensions of the state space.
    num_directions: int, optional
        The number of rays to sample on the unit sphere for the estimation.
    device: th.device | None, optional
        The device to use for Torch tensors.
    initial_radius: float, optional
        The initial radius for the ray intersection search.
    growth_factor: float, optional
        The factor by which to grow the search radius during the ray intersection search.
    max_radius: float, optional
        The maximum radius to search for ray intersections.
    max_bisection_steps: int, optional
        The maximum number of bisection steps to perform for refining ray intersections.

    Returns
    -------
    LevelSetEstimate
        A dataclass containing the estimated measure and related information about the level set.
    """
    if rho < 0.0:
        raise ValueError(f"rho must be non-negative, got {rho}.")
    if num_states < 1:
        raise ValueError(f"num_states must be positive, got {num_states}.")
    if num_directions < 1:
        raise ValueError(f"num_directions must be positive, got {num_directions}.")

    directions = _sample_unit_sphere_directions(num_states, num_directions)
    base_point = th.zeros((1, num_states), dtype=th.float32, device=device)
    base_value = float(lyapunov_fn(base_point)[0])

    if base_value > rho:
        raise ValueError(
            f"Cannot infer level-set limits from rho={rho:.6g} because V(0)={base_value:.6g} > rho."
        )

    radii, truncated_mask = _find_level_ray_intersections(
        lyapunov_fn,
        rho,
        directions,
        device=device,
        initial_radius=initial_radius,
        growth_factor=growth_factor,
        max_radius=max_radius,
        max_bisection_steps=max_bisection_steps,
    )

    unit_sphere_surface_area = _unit_sphere_surface_area(num_states)
    measure = _sublevel_surface_area(radii, unit_sphere_surface_area, num_states)

    estimate = LevelSetEstimate(
        rho=float(rho),
        num_states=int(num_states),
        num_directions=int(num_directions),
        directions=directions,
        radii=radii,
        measure=measure,
        unit_sphere_surface_area=unit_sphere_surface_area,
        max_radius=float(max_radius),
        truncated_mask=truncated_mask,
    )
    __logger__.info(
        "Estimated star-shaped level-set measure at rho=%.6f in %dD from %d sphere rays: measure=%.6f, truncated_fraction=%.4f.",
        estimate.rho,
        estimate.num_states,
        estimate.num_directions,
        estimate.measure,
        estimate.truncated_fraction,
    )
    return estimate