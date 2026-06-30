from __future__ import annotations

import torch as th
import torch.nn as nn
import logging

from contextlib import nullcontext
from pkg_logger import suppress_native_output
from dataclasses import dataclass
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

__logger__ = logging.getLogger(__name__)


def _extract_affine_l1_term(lyap_model: nn.Module) -> tuple[th.Tensor, th.Tensor] | None:
    """Extract ``(M, x_star)`` of the affine L1 Lyapunov term when available.

    The neural Lyapunov candidate has the structure
    ``V(x) = |phi(x) - phi(x*)| + ||M (x - x*)||_1`` with ``M = eps I + R^T R``.
    Because the feature term is non-negative, ``||M (x - x*)||_1`` is itself a
    sound lower bound on ``V(x)``. This helper duck-types on the candidate so it
    works for any model that exposes ``_pd_matrix()`` (or ``r_factor``/``eps``)
    together with an ``x_star`` buffer, and returns ``None`` otherwise.

    Parameters
    ----------
    lyap_model : nn.Module
        Candidate Lyapunov model to introspect.

    Returns
    -------
    tuple[th.Tensor, th.Tensor] | None
        ``(M, x_star)`` with ``M`` of shape ``(nx, nx)`` and ``x_star`` of shape
        ``(nx,)``, or ``None`` when the model does not expose the affine term.
    """
    pd_matrix_fn = getattr(lyap_model, "_pd_matrix", None)
    x_star = getattr(lyap_model, "x_star", None)
    if not callable(pd_matrix_fn) or x_star is None:
        return None

    with th.no_grad():
        pd_matrix = pd_matrix_fn()
        if pd_matrix.ndim != 2 or pd_matrix.shape[0] != pd_matrix.shape[1]:
            return None
        x_star_vec = x_star.detach().reshape(-1)

    return pd_matrix.detach(), x_star_vec


def affine_l1_lower_bound(
    regions: th.Tensor,
    pd_matrix: th.Tensor,
    x_star: th.Tensor,
) -> th.Tensor:
    """Exact per-row lower bound of ``||M (x - x*)||_1`` over axis-aligned boxes.

    For a fixed matrix ``M`` the L1 term decouples into a sum of absolute affine
    forms ``sum_i |m_i^T (x - x*)|``. Each ``|m_i^T delta|`` is minimised
    independently over the box ``delta in [lb - x*, ub - x*]``: the affine form
    ``m_i^T delta`` ranges over an interval ``[a_i, b_i]`` whose endpoints follow
    from the sign of each coefficient, and the minimum of ``|.|`` over that
    interval is ``0`` if it straddles zero and ``min(|a_i|, |b_i|)`` otherwise.
    Summing the per-row minima gives a sound (slightly conservative) lower bound
    on the L1 term and therefore on ``V(x)`` when composed with a non-negative
    feature term.

    Parameters
    ----------
    regions : th.Tensor
        Packed region bounds of shape ``(N, 2, nx)`` as ``[lb, ub]``.
    pd_matrix : th.Tensor
        The matrix ``M = eps I + R^T R`` of shape ``(nx, nx)``.
    x_star : th.Tensor
        Equilibrium point of shape ``(nx,)``.

    Returns
    -------
    th.Tensor
        Lower bound of shape ``(N,)``.
    """
    if regions.ndim != 3 or regions.shape[1] != 2:
        raise ValueError("regions must have shape (N, 2, nx).")

    device = pd_matrix.device
    dtype = pd_matrix.dtype
    lbs = regions[:, 0].to(device=device, dtype=dtype)
    ubs = regions[:, 1].to(device=device, dtype=dtype)

    # Shift the box by the equilibrium so the affine form becomes M @ delta.
    delta_lo = lbs - x_star.reshape(1, -1)
    delta_hi = ubs - x_star.reshape(1, -1)

    # For each output row i and box, the affine form m_i^T delta attains its
    # interval endpoints by choosing delta_lo / delta_hi per coordinate based on
    # the coefficient sign. Vectorise over (N, rows, coords).
    m = pd_matrix.unsqueeze(0)                 # (1, rows, coords)
    pos = m.clamp_min(0.0)                      # contributes via upper/lower directly
    neg = m.clamp_max(0.0)                      # sign flips lo/hi roles

    lo = delta_lo.unsqueeze(1)                  # (N, 1, coords)
    hi = delta_hi.unsqueeze(1)                  # (N, 1, coords)

    form_lo = (pos * lo + neg * hi).sum(dim=2)  # (N, rows) min of m_i^T delta
    form_hi = (pos * hi + neg * lo).sum(dim=2)  # (N, rows) max of m_i^T delta

    # Min of |.| over [form_lo, form_hi]: 0 if interval straddles 0 else nearest.
    straddles = (form_lo <= 0.0) & (form_hi >= 0.0)
    nearest = th.minimum(form_lo.abs(), form_hi.abs())
    row_min_abs = th.where(straddles, th.zeros_like(nearest), nearest)

    return row_min_abs.sum(dim=1)

@dataclass(frozen=True)
class LyapunovRegionBounds:
    """Lower and upper LiRPA bounds of ``V(x)`` over packed regions."""

    lower: th.Tensor
    upper: th.Tensor

    def inside_mask(self, threshold: float) -> th.Tensor:
        """Return boolean mask of regions with certifiably ``V(x) <= threshold``."""
        return self.upper <= threshold

    def outside_mask(self, threshold: float) -> th.Tensor:
        """Return boolean mask of regions with certifiably ``V(x) > threshold``."""
        return self.lower > threshold

    def sublevel_masks(self, threshold: float) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return ``(inside, boundary, outside)`` masks for ``V(x) <= threshold``."""
        inside = self.inside_mask(threshold)
        outside = self.outside_mask(threshold)
        boundary = ~(inside | outside)
        return inside, boundary, outside
    
    def get_sublevel_masked_regions(self, regions: th.Tensor, threshold: float) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return ``(inside, boundary, outside)`` region tensors for ``V(x) <= threshold``."""
        inside_mask, boundary_mask, outside_mask = self.sublevel_masks(threshold)
        inside_regions = regions[inside_mask]
        boundary_regions = regions[boundary_mask]
        outside_regions = regions[outside_mask]
        __logger__.debug(
            "Classified %d / %d regions as outside sublevel: V(x) > %.4f.",
            len(outside_regions),
            len(regions),
            threshold,
        )
        return inside_regions, boundary_regions, outside_regions


class LiRPALyapunovRegionBounds:
    """Compute LiRPA bounds of a scalar Lyapunov model over axis-aligned regions."""

    def __init__(
        self,
        lyap_model: nn.Module,
        state_dim: int,
        *,
        batch_size: int = 512,
        default_bound_method: str = "crown",
        suppress_native_output: bool = True,
        use_affine_l1_lower_bound: bool = True,
        device: th.device = th.device("cpu"),
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.device = device
        self.state_dim = int(state_dim)
        self.batch_size = int(batch_size)
        self.default_bound_method = default_bound_method.strip().lower()
        self.suppress_native_output = bool(suppress_native_output)
        self.lyap_model = lyap_model.to(self.device).eval()
        self.bounded_model = self._build_bounded_model()

        self.use_affine_l1_lower_bound = bool(use_affine_l1_lower_bound)
        self._affine_l1_term: tuple[th.Tensor, th.Tensor] | None = None
        if self.use_affine_l1_lower_bound:
            self._affine_l1_term = self._resolve_affine_l1_term()

    def _resolve_affine_l1_term(self) -> tuple[th.Tensor, th.Tensor] | None:
        """Cache the ``(M, x_star)`` affine term on the managed device, if present."""
        affine_term = _extract_affine_l1_term(self.lyap_model)
        if affine_term is None:
            __logger__.debug(
                "Lyapunov model %s does not expose an affine L1 term; "
                "falling back to LiRPA-only lower bounds.",
                type(self.lyap_model).__name__,
            )
            return None
        pd_matrix, x_star = affine_term
        return (
            pd_matrix.to(device=self.device, dtype=th.float32),
            x_star.to(device=self.device, dtype=th.float32),
        )

    def _get_suppress_ctx(self):
        """Return context manager controlling native backend output."""
        if self.suppress_native_output:
            return suppress_native_output(suppress_stderr=True)
        return nullcontext()

    def _build_bounded_model(self) -> BoundedModule:
        dummy_x = th.zeros(1, self.state_dim, device=self.device)
        with th.no_grad():
            output = self.lyap_model(dummy_x)

        if output.ndim == 1:
            output = output.unsqueeze(1)
        if output.ndim != 2 or output.shape[1] != 1:
            raise ValueError("lyap_model must return a scalar output of shape (batch, 1).")

        return BoundedModule(
            self.lyap_model,
            (dummy_x,),
            device=self.device,
            verbose=False,
            bound_opts={"perturb_bound": True},
        )

    def compute_bounds_for_regions(
        self,
        regions: th.Tensor,
        *,
        method: str | None = None,
    ) -> LyapunovRegionBounds:
        """Compute LiRPA lower and upper bounds for ``V`` on packed regions."""
        if regions.ndim != 3 or regions.shape[1] != 2 or regions.shape[2] != self.state_dim:
            raise ValueError("regions must have shape (N, 2, state_dim).")

        if len(regions) == 0:
            empty = th.empty((0,), dtype=th.float32, device=self.device)
            return LyapunovRegionBounds(lower=empty, upper=empty)

        bound_method = self.default_bound_method if method is None else method.strip().lower()
        lower_bounds: list[th.Tensor] = []
        upper_bounds: list[th.Tensor] = []

        lbs = regions[:, 0].to(device=self.device, dtype=th.float32)
        ubs = regions[:, 1].to(device=self.device, dtype=th.float32)

        for start_idx in range(0, len(regions), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(regions))
            batch_lbs = lbs[start_idx:end_idx]
            batch_ubs = ubs[start_idx:end_idx]
            batch_centers = 0.5 * (batch_lbs + batch_ubs)

            ptb = PerturbationLpNorm(norm=float("inf"), x_L=batch_lbs, x_U=batch_ubs)
            bounded_input = BoundedTensor(batch_centers, ptb)
            
            with self._get_suppress_ctx():
                bounds_ctx = nullcontext() if bound_method == "alpha-crown" else th.no_grad()
                with bounds_ctx:
                    lb_out, ub_out = self.bounded_model.compute_bounds(
                        x=(bounded_input,),
                        method=bound_method,
                    )

            lower_bounds.append(lb_out.reshape(-1)[: end_idx - start_idx])
            upper_bounds.append(ub_out.reshape(-1)[: end_idx - start_idx])

        lower = th.cat(lower_bounds, dim=0)
        upper = th.cat(upper_bounds, dim=0)

        if self._affine_l1_term is not None:
            pd_matrix, x_star = self._affine_l1_term
            analytic_lower = affine_l1_lower_bound(
                regions.to(device=self.device, dtype=th.float32),
                pd_matrix,
                x_star,
            ).to(dtype=lower.dtype)
            # Both are valid lower bounds on V; keep the tighter (larger) one.
            # The analytic term ignores the non-negative feature contribution and
            # is therefore always sound, while being immune to the abs-relaxation
            # collapse that loosens the LiRPA bound near the origin.
            tightened_lower = th.maximum(lower, analytic_lower)
            improved = int((tightened_lower > lower + 1e-9).sum().item())
            if improved > 0:
                __logger__.debug(
                    "Affine L1 lower bound tightened %d / %d region V lower bounds.",
                    improved,
                    len(lower),
                )
            lower = tightened_lower

        return LyapunovRegionBounds(
            lower=lower,
            upper=upper,
        )