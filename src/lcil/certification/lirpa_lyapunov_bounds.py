from __future__ import annotations

from contextlib import nullcontext
from pkg_logger import suppress_native_output
from dataclasses import dataclass

import torch as th
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


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

        return LyapunovRegionBounds(
            lower=th.cat(lower_bounds, dim=0),
            upper=th.cat(upper_bounds, dim=0),
        )