from __future__ import annotations

import logging

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import gamma, pi
from typing import Any, Literal, Self

import numpy as np
import torch as th
from numpy.typing import NDArray
from scipy.stats import norm, qmc

from ..utils.base_config import JsonDataclass
from ..utils.constants import *

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class RayShootingDetails(JsonDataclass):
    """Estimation details specific to deterministic sphere ray shooting."""

    NP_ARRAY_FIELDS = ("directions", "radii", "truncated_mask")

    num_directions: int
    directions: NDArray
    radii: NDArray
    unit_sphere_surface_area: float
    max_radius: float
    truncated_mask: NDArray

    @property
    def truncated(self) -> bool:
        """Whether any direction failed to leave the sublevel set before ``max_radius``."""
        return bool(np.any(self.truncated_mask))

    @property
    def truncated_fraction(self) -> float:
        """Fraction of directions truncated at ``max_radius``."""
        return float(np.mean(self.truncated_mask))


@dataclass(frozen=True)
class MonteCarloDetails(JsonDataclass):
    """Estimation details specific to uniform Monte-Carlo sampling."""

    NP_ARRAY_FIELDS = ("bounds",)

    num_samples: int
    bounds: NDArray
    box_volume: float
    inside_fraction: float


@dataclass(frozen=True)
class LevelSetEstimate(JsonDataclass):
    """Size estimate for ``V(x) <= rho`` containing common metrics and method-specific details."""

    DEFAULT_FILE_NAME = LEVEL_SET_FILENAME

    rho: float
    num_states: int
    measure: float
    method: str
    details: RayShootingDetails | MonteCarloDetails

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        values = dict(data)
        if "details" not in values:
            method = values.get("method", "ray_shooting")
            if method == "ray_shooting":
                values["details"] = RayShootingDetails.from_dict(values)
            else:
                values["details"] = MonteCarloDetails.from_dict(values)
        return super().from_dict(values)


def _unit_sphere_surface_area(num_states: int) -> float:
    r"""Return the surface area of the unit sphere ``S^(n-1)`` in ``R^n``.
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


def _sublevel_measure(
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
        The nD volume of the star-shaped set defined by the given ray lengths.
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
    bisection_tol: float = 1e-5,
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
    bisection_tol: float, optional
        The tolerance for convergence in the bisection method when refining ray intersections.

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
    if bisection_tol <= 0.0:
        raise ValueError(f"bisection_tol must be positive, got {bisection_tol}.")

    active_mask = th.ones(num_directions, dtype=th.bool, device=device)
    low = th.zeros(num_directions, 1, dtype=th.float32, device=device)
    high = th.full((num_directions, 1), initial_radius, dtype=th.float32, device=device)

    # Scaling step
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

        # Monotonicity check
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
                __logger__.warning(
                    "Monotonicity violated along at least one ray; the star-shaped level-set assumption does not hold for rho=%.6f.",
                    c_level,
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

    # Bisection step
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

        # Monotonicity check
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
            __logger__.warning(
                "Monotonicity violated along at least one ray; the star-shaped level-set assumption does not hold for rho=%.6f.",
                c_level,
            )

        err_detached = finite_values.detach() - c_level

        is_pos = err_detached > 0
        high_next = th.where(is_pos, curr_r[finite_active_indices], high[finite_active_indices])
        low_next = th.where(~is_pos, curr_r[finite_active_indices], low[finite_active_indices])

        low[finite_active_indices] = low_next
        high[finite_active_indices] = high_next
        curr_r[finite_active_indices] = 0.5 * (low_next + high_next)

        converged = (
            (th.abs(err_detached).squeeze(-1) < bisection_tol)
            | ((high_next - low_next).squeeze(-1) < bisection_tol)
        )
        converged_indices = finite_active_indices[converged]
        active_roots[converged_indices] = False

    return curr_r.squeeze(-1).cpu().numpy(), truncated_mask.cpu().numpy()


def estimate_level_set_measure_ray_shooting(
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
    bisection_tol: float = 1e-5,
) -> LevelSetEstimate:
    """Estimate the star-shaped nD measure of ``V(x) <= rho`` from sphere rays."""
    if rho < 0.0:
        raise ValueError(f"rho must be non-negative, got {rho}.")
    if num_states < 1:
        raise ValueError(f"num_states must be positive, got {num_states}.")
    if num_directions < 1:
        raise ValueError(f"num_directions must be positive, got {num_directions}.")

    directions = _sample_unit_sphere_directions(num_states, num_directions)
    base_point = th.zeros((1, num_states), dtype=th.float32, device=device)
    with th.no_grad():
        base_value = float(lyapunov_fn(base_point)[0].detach().item())

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
        bisection_tol=bisection_tol,
    )

    unit_sphere_surface_area = _unit_sphere_surface_area(num_states)
    measure = _sublevel_measure(radii, unit_sphere_surface_area, num_states)

    details = RayShootingDetails(
        num_directions=int(num_directions),
        directions=directions,
        radii=radii,
        unit_sphere_surface_area=unit_sphere_surface_area,
        max_radius=float(max_radius),
        truncated_mask=truncated_mask,
    )
    estimate = LevelSetEstimate(
        rho=float(rho),
        num_states=int(num_states),
        measure=float(measure),
        method="ray_shooting",
        details=details,
    )
    __logger__.info(
        "Estimated star-shaped level-set measure at rho=%.6f in %dD from %d sphere rays: measure=%.6f, truncated_fraction=%.4f.",
        estimate.rho,
        estimate.num_states,
        details.num_directions,
        estimate.measure,
        details.truncated_fraction,
    )
    return estimate


def estimate_level_set_measure_monte_carlo(
    lyapunov_fn: Callable[[th.Tensor], th.Tensor],
    rho: float,
    *,
    num_states: int,
    bounds: float | Sequence[tuple[float, float]] | NDArray | None = None,
    num_samples: int = 100_000,
    batch_size: int = 10_000,
    device: th.device | None = None,
    max_radius: float = 10.0,
) -> LevelSetEstimate:
    """Estimate the nD measure of ``V(x) <= rho`` via uniform Monte-Carlo sampling."""
    if rho < 0.0:
        raise ValueError(f"rho must be non-negative, got {rho}.")
    if num_states < 1:
        raise ValueError(f"num_states must be positive, got {num_states}.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")

    if bounds is None:
        low = np.full(num_states, -float(max_radius), dtype=np.float32)
        high = np.full(num_states, float(max_radius), dtype=np.float32)
    elif isinstance(bounds, (int, float)):
        r = float(bounds)
        if r <= 0.0:
            raise ValueError(f"bounds radius must be positive, got {bounds}.")
        low = np.full(num_states, -r, dtype=np.float32)
        high = np.full(num_states, r, dtype=np.float32)
    else:
        bounds_arr = np.asarray(bounds, dtype=np.float32)
        if bounds_arr.shape == (num_states, 2):
            low = bounds_arr[:, 0]
            high = bounds_arr[:, 1]
        elif bounds_arr.shape == (2, num_states):
            low = bounds_arr[0, :]
            high = bounds_arr[1, :]
        else:
            raise ValueError(
                f"bounds array must have shape ({num_states}, 2) or (2, {num_states}), got {bounds_arr.shape}."
            )

    if (low >= high).any():
        raise ValueError(f"Lower bounds must be strictly less than upper bounds, got low={low}, high={high}.")

    bounds_matrix = np.column_stack([low, high])
    box_volume = float(np.prod(high - low))
    low_ts = th.as_tensor(low, dtype=th.float32, device=device)
    high_ts = th.as_tensor(high, dtype=th.float32, device=device)
    range_ts = high_ts - low_ts

    total_inside = 0
    num_processed = 0

    while num_processed < num_samples:
        cur_batch = min(batch_size, num_samples - num_processed)
        rand_uniform = th.rand((cur_batch, num_states), dtype=th.float32, device=device)
        points = low_ts + rand_uniform * range_ts

        with th.no_grad():
            values = lyapunov_fn(points).squeeze(-1)

        inside = (values <= rho) & th.isfinite(values)
        total_inside += int(inside.sum().item())
        num_processed += cur_batch

    inside_fraction = total_inside / num_samples
    measure = box_volume * inside_fraction

    details = MonteCarloDetails(
        num_samples=int(num_samples),
        bounds=bounds_matrix,
        box_volume=box_volume,
        inside_fraction=float(inside_fraction),
    )
    estimate = LevelSetEstimate(
        rho=float(rho),
        num_states=int(num_states),
        measure=float(measure),
        method="monte_carlo",
        details=details,
    )
    __logger__.info(
        "Estimated Monte-Carlo level-set measure at rho=%.6f in %dD from %d samples: measure=%.6f (box_volume=%.4f, inside_fraction=%.4f).",
        estimate.rho,
        estimate.num_states,
        details.num_samples,
        estimate.measure,
        box_volume,
        inside_fraction,
    )
    return estimate


def estimate_level_set_measure(
    lyapunov_fn: Callable[[th.Tensor], th.Tensor],
    rho: float,
    *,
    num_states: int,
    method: Literal["ray_shooting", "monte_carlo"] = "ray_shooting",
    bounds: float | Sequence[tuple[float, float]] | NDArray | None = None,
    num_samples: int | None = None,
    num_directions: int = 512,
    device: th.device | None = None,
    initial_radius: float = 1.0,
    growth_factor: float = 2.0,
    max_radius: float = 1e6,
    max_bisection_steps: int = 60,
    bisection_tol: float = 1e-5,
) -> LevelSetEstimate:
    """Estimate the nD measure of ``V(x) <= rho``.

    Parameters
    ----------
    lyapunov_fn: Callable[[th.Tensor], th.Tensor]
        The Lyapunov function to evaluate, accepting Torch tensors.
    rho: float
        The sublevel set value to estimate.
    num_states: int
        The number of dimensions of the state space.
    method: {"ray_shooting", "monte_carlo"}, optional
        The estimation method to use:
        - "ray_shooting": deterministic star-shaped sphere ray bisection (default).
        - "monte_carlo": uniform Monte-Carlo sampling inside a bounding box.
    bounds: float | Sequence[tuple[float, float]] | NDArray | None, optional
        Bounding box for Monte-Carlo sampling.
    num_samples: int | None, optional
        Number of Monte-Carlo samples (used when method="monte_carlo"). Default is 100,000.
    num_directions: int, optional
        Number of rays (used when method="ray_shooting"). Default is 512.
    device: th.device | None, optional
        The device to use for Torch tensors.
    initial_radius: float, optional
        The initial radius for ray shooting.
    growth_factor: float, optional
        The growth factor for ray shooting.
    max_radius: float, optional
        The maximum radius / bounding radius.
    max_bisection_steps: int, optional
        Max bisection steps for ray shooting.
    bisection_tol: float, optional
        Bisection tolerance for ray shooting.

    Returns
    -------
    LevelSetEstimate
        A dataclass containing the estimated measure and related information.
    """
    if method == "ray_shooting":
        return estimate_level_set_measure_ray_shooting(
            lyapunov_fn=lyapunov_fn,
            rho=rho,
            num_states=num_states,
            num_directions=num_directions,
            device=device,
            initial_radius=initial_radius,
            growth_factor=growth_factor,
            max_radius=max_radius,
            max_bisection_steps=max_bisection_steps,
            bisection_tol=bisection_tol,
        )
    elif method == "monte_carlo":
        actual_samples = num_samples if num_samples is not None else 100_000
        mc_max_radius = max_radius if max_radius < 1e5 else 10.0
        return estimate_level_set_measure_monte_carlo(
            lyapunov_fn=lyapunov_fn,
            rho=rho,
            num_states=num_states,
            bounds=bounds,
            num_samples=actual_samples,
            device=device,
            max_radius=mc_max_radius,
        )
    else:
        raise ValueError(
            f"Unknown estimation method '{method}'. Allowed methods are 'ray_shooting' and 'monte_carlo'."
        )