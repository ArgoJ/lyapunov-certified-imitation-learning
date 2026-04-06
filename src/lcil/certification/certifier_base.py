from __future__ import annotations

import torch as th
import torch.nn as nn
import numpy as np
import logging

from contextlib import nullcontext
from typing import Sequence
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from auto_LiRPA import BoundedTensor, PerturbationLpNorm, BoundedModule
from pkg_logger import suppress_native_output
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

from .config import LyapunovCertificationConfig
from .models import ClosedLoopLyapunovConditionVerifier
from ..utils.helpers import none_to_float

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionCertificationResult:
    """Result container for a full-region certification pass."""
    success: bool
    counter_examples: NDArray
    failed_regions: NDArray
    certified_regions: NDArray


@dataclass(frozen=True)
class RecursiveCertificationResult:
    """Internal tensor result for recursive region certification."""

    certified: th.Tensor
    failed: th.Tensor
    counterexamples: th.Tensor

    @classmethod
    def empty(cls, state_dim: int, device: th.device) -> RecursiveCertificationResult:
        empty = th.empty((0, 2, state_dim), device=device)
        return cls(
            certified=empty,
            failed=empty,
            counterexamples=empty,
        )

    def with_certified(self, regions: th.Tensor) -> RecursiveCertificationResult:
        return replace(self, certified=regions)

    def with_failed(self, regions: th.Tensor) -> RecursiveCertificationResult:
        return replace(self, failed=regions)
    
    def with_cert_and_failed(self, bs: th.Tensor, is_safe: th.Tensor) -> RecursiveCertificationResult:
        failed_mask = ~is_safe

        certified_bs = bs[is_safe]
        failed_bs = bs[failed_mask]
        return replace(self, certified=certified_bs, failed=failed_bs)

    def with_counterexamples(self, regions: th.Tensor) -> RecursiveCertificationResult:
        return replace(self, counterexamples=regions)

    def __add__(self, other: RecursiveCertificationResult) -> RecursiveCertificationResult:
        return RecursiveCertificationResult(
            certified=th.cat([self.certified, other.certified], dim=0),
            failed=th.cat([self.failed, other.failed], dim=0),
            counterexamples=th.cat([self.counterexamples, other.counterexamples], dim=0),
        )


class _NegatedLyapunovModelWrapper(nn.Module):
    """Wrap ``V(x)`` as ``-V(x)`` for outside-sublevel proofs."""

    def __init__(self, lyap_model: nn.Module):
        super().__init__()
        self.lyap_model = lyap_model

    def forward(self, x: th.Tensor) -> th.Tensor:
        return -self.lyap_model(x)


class BaseCertifier(ABC):
    """
    Abstract base class for Lyapunov certifiers. 
    Definie the core logic for the Certification, while delegating backend-specific details to subclasses via abstract methods.
    """

    def __init__(
            self, 
            policy_model: nn.Module,
            lyap_model: nn.Module,
            dyn_model: nn.Module,
            config: LyapunovCertificationConfig,
            device: th.device = th.device("cpu"),
    ):
        """Initialize a backend-agnostic Lyapunov certifier.

        Parameters
        ----------
        policy_model : nn.Module
            Control policy network ``u = pi(x)`` used in the closed-loop verifier.
        lyap_model : nn.Module
            Candidate Lyapunov network ``V(x)``.
        dyn_model : nn.Module
            Dynamics model for one-step state propagation in closed loop.
        config : LyapunovCertificationConfig
            Certification configuration (bounds, grid step, rho search settings,
            backend options).
        device : th.device, optional
            Device used for all model parameters and certification tensors,
            by default ``th.device("cpu")``.
        """
        self.config = config
        self.device = device

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()

        self.bounds = self._resolve_bounds(config.cert_bounds, device)
        self.ibp_filter_model = self._get_ibp_filter(
            self.lyap_model, config.state_dim, device
        ) if config.use_ibp_filter else None
        self.regions = None
        self.verifier = None

    # ==========================================
    # ABSTRACT METHODS
    # ==========================================

    @abstractmethod
    def setup_backend(self, *args, **kwargs) -> None:
        """Initialize backend-specific verifier state.

        Parameters
        ----------
        *args
            Backend-specific positional arguments.
        **kwargs
            Backend-specific keyword arguments.
        """
        pass

    @abstractmethod
    def _certify_batched_regions(
            self,
            bs: th.Tensor,
            rho: float,
            early_exit: bool = True,
            *args, **kwargs
    ) -> th.Tensor:
        """Certify a batch of axis-aligned regions.

        Parameters
        ----------
        bs : th.Tensor
            Packed region bounds with shape ``(n, 2, state_dim)``.
        rho : float
            Lyapunov level-set value to certify.
        early_exit : bool, optional
            If ``True``, backend may stop once a violation is found.
        *args
            Backend-specific positional arguments.
        **kwargs
            Backend-specific keyword arguments.

        Returns
        -------
        th.Tensor
            Boolean tensor indicating safe regions.
        """
        pass

    # ==========================================
    # CORE LOGIC
    # ==========================================
    @staticmethod
    def _build_center_refined_axis_edges(
        lb: th.Tensor,
        ub: th.Tensor,
        num_bins: int,
        refinement_factor: float,
    ) -> th.Tensor:
        """Build 1D grid edges with optional geometric refinement toward zero."""
        if refinement_factor == 1.0 or lb >= 0.0 or ub <= 0.0:
            return th.linspace(lb, ub, steps=num_bins + 1, device=lb.device)

        neg_span = float((-lb).item())
        pos_span = float(ub.item())

        neg_bins = int(round(num_bins * neg_span / (neg_span + pos_span))) if neg_span + pos_span > 0 else 0
        neg_bins = max(1, min(num_bins - 1, neg_bins))
        pos_bins = num_bins - neg_bins

        def _side_edges(span: float, bins: int, sign: float, device: th.device) -> th.Tensor:
            if bins <= 0 or span <= 0.0:
                return th.zeros(1, device=device)
            if bins == 1:
                distances = th.tensor([span], dtype=th.float32, device=device)
            else:
                powers = th.arange(bins - 1, -1, -1, dtype=th.float32, device=device)
                weights = refinement_factor ** powers
                widths = span * weights / weights.sum()
                distances = th.cumsum(widths, dim=0)
            return sign * distances

        neg_edges = _side_edges(neg_span, neg_bins, -1.0, lb.device)
        pos_edges = _side_edges(pos_span, pos_bins, 1.0, ub.device)

        return th.cat([
            th.tensor([lb.item()], dtype=th.float32, device=lb.device),
            neg_edges.flip(0)[1:],
            th.zeros(1, dtype=th.float32, device=lb.device),
            pos_edges[:-1],
            th.tensor([ub.item()], dtype=th.float32, device=lb.device),
        ])

    @staticmethod
    def _resolve_bounds(bounds: Sequence[float], device: th.device) -> th.Tensor:
        """Convert state bounds to a tensor on the target device.

        Parameters
        ----------
        bounds : Sequence[float]
            Lower and upper bounds as ``(2, state_dim)``.
        device : th.device
            Device where the tensor should be allocated.

        Returns
        -------
        th.Tensor
            Bounds tensor with shape ``(2, state_dim)`` and dtype ``float32``.

        Raises
        ------
        ValueError
            If the bounds do not have shape ``(2, state_dim)``.
        """
        bounds = th.as_tensor(bounds, dtype=th.float32, device=device)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("bounds must be a sequence of shape (2, nx) [lb, ub].")
        return bounds

    @staticmethod
    def _iterative_rho_search(
        total:int, 
        desc: str, 
        initial_values: tuple[float, float], 
        step_fn: callable
    ) -> tuple[float | None, float]:
        """Run iterative rho updates until stopping criterion is met.

        Parameters
        ----------
        total : int
            Maximum number of iterations.
        desc : str
            Progress-bar description.
        initial_values : tuple[float, float]
            Initial ``(rho_lo, rho_up)`` values.
        step_fn : callable
            Function mapping ``(rho_lo, rho_up)`` to ``(stop, rho_lo, rho_up)``.

        Returns
        -------
        tuple[float | None, float]
            Final ``(rho_lo, rho_up)`` pair.
        """
        rho_lo, rho_up = initial_values
        with Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("rho_lo: {task.fields[rho_lo]:.4f}"),
            TextColumn("rho_up: {task.fields[rho_up]:.4f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                desc,
                total=total,
                rho_lo=none_to_float(rho_lo),
                rho_up=none_to_float(rho_up),
            )
            for _ in range(total):
                try:
                    stop, rho_lo, rho_up = step_fn(rho_lo, rho_up)
                    if stop: 
                        break
                finally:
                    progress.update(
                        task, 
                        rho_lo=none_to_float(rho_lo), 
                        rho_up=none_to_float(rho_up)
                    )
        return rho_lo, rho_up

    def _get_suppress_ctx(self):
        """Return context manager controlling native backend output.

        Returns
        -------
        contextlib.AbstractContextManager
            Suppression context if enabled, otherwise ``nullcontext``.
        """
        if self.config.suppress_native_output:
            lirpa_ctx = suppress_native_output(suppress_stderr=True)
        else:
            lirpa_ctx = nullcontext()
        return lirpa_ctx

    @staticmethod
    def _get_ibp_filter(lyap_model: nn.Module, input_size: int, device: th.device) -> BoundedModule:
        if not isinstance(lyap_model, nn.Module):
            raise ValueError("Lyapunov model must be provided to construct IBP filter.")
    
        negated_lyap_model = _NegatedLyapunovModelWrapper(lyap_model).to(device)
        negated_lyap_model.eval()
        dummy_x = th.zeros(1, input_size, device=device)
        return BoundedModule(
            negated_lyap_model,
            (dummy_x,),
            device=device,
            verbose=False,
            bound_opts={"perturb_bound": True},
        )

    def _filter_sublevel_regions(
        self,
        bs: th.Tensor,
        rho: float,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Keep only boxes not proven to satisfy ``V(x) > rho`` everywhere."""
        if len(bs) == 0:
            return bs, bs

        lbs, ubs = self._unpack_regions(bs)

        keep_mask = th.ones(len(lbs), dtype=th.bool, device=self.device)
        negated_threshold = -float(rho)

        for start_idx in range(0, len(lbs), self.config.batch_size):
            end_idx = min(start_idx + self.config.batch_size, len(lbs))
            batch_lbs = lbs[start_idx:end_idx]
            batch_ubs = ubs[start_idx:end_idx]
            batch_centers = 0.5 * (batch_lbs + batch_ubs)

            ptb = PerturbationLpNorm(norm=float("inf"), x_L=batch_lbs, x_U=batch_ubs)
            bounded_input = BoundedTensor(batch_centers, ptb)

            with th.no_grad():
                _, ub_out = self.ibp_filter_model.compute_bounds(
                    x=(bounded_input,),
                    method="ibp",
                )

            batch_is_outside = ub_out.flatten() <= negated_threshold
            keep_mask[start_idx:end_idx] = ~batch_is_outside

        if keep_mask.all():
            return bs, bs[:0]

        __logger__.info(
            "Pruned %d / %d regions proven outside V(x) <= %.6f.",
            int((~keep_mask).sum().item()),
            len(lbs),
            float(rho),
        )
        return bs[keep_mask], bs[~keep_mask]

    def _regions_tensor_to_np(self, regions: th.Tensor) -> NDArray:
        """Convert a region tensor ``(N, 2, state_dim)`` to NumPy."""
        if regions.numel() == 0:
            return np.empty((0, 2, self.config.state_dim), dtype=np.float32)
        return regions.cpu().numpy()

    def _split_failed_regions(self, failed_bs: th.Tensor) -> th.Tensor:
        """Split each failed box along its widest dimension into two packed sub-boxes."""
        mids = 0.5 * (failed_bs[:, 0] + failed_bs[:, 1])
        widths = failed_bs[:, 1] - failed_bs[:, 0]
        split_dims = th.argmax(widths, dim=1)
        row_idx = th.arange(failed_bs.shape[0], device=self.device)

        low_bs = failed_bs.clone()
        high_bs = failed_bs.clone()

        low_bs[row_idx, 1, split_dims] = mids[row_idx, split_dims]
        high_bs[row_idx, 0, split_dims] = mids[row_idx, split_dims]

        return th.cat([low_bs, high_bs], dim=0)

    @staticmethod
    def _pack_regions(lbs: th.Tensor, ubs: th.Tensor) -> th.Tensor:
        """Pack lower/upper bounds into ``(N, 2, state_dim)`` format."""
        return th.stack([lbs, ubs], dim=1)

    @staticmethod
    def _unpack_regions(bs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Unpack ``(N, 2, state_dim)`` regions into lower/upper bounds."""
        return bs[:, 0], bs[:, 1]

    def _setup_verifier(self) -> ClosedLoopLyapunovConditionVerifier:
        """Construct and initialize the closed-loop Lyapunov verifier.

        Returns
        -------
        ClosedLoopLyapunovConditionVerifier
            Verifier module moved to ``self.device`` and set to eval mode.
        """
        lbx_batched = self.bounds[0].unsqueeze(0)
        ubx_batched = self.bounds[1].unsqueeze(0)

        verifier = ClosedLoopLyapunovConditionVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=lbx_batched,
            ubx=ubx_batched,
            invariance_weight=self.config.invariance_weight,
        )

        return verifier

    def _resolve_origin_exclusion(self) -> th.Tensor:
        """Resolve origin exclusion widths per state dimension.

        Returns
        -------
        th.Tensor
            Non-negative exclusion widths with shape ``(state_dim,)`` clipped to
            remain within available bounds around the origin.
        """
        # Default: 1% of per-dimension bound radius, capped at 0.1.
        default_exclusion = th.minimum(
            self.bounds.abs().max(dim=0).values * 0.01,
            th.full((self.config.state_dim,), 0.1, dtype=th.float32, device=self.device),
        )

        raw_exclusion = self.config.origin_exclusion
        if raw_exclusion is None:
            exclusion = default_exclusion
        elif isinstance(raw_exclusion, (int, float)):
            scalar = float(raw_exclusion)
            if scalar < 0:
                raise ValueError("cert_origin_exclusion must be non-negative.")
            exclusion = th.full(
                (self.config.state_dim,),
                scalar,
                dtype=th.float32,
                device=self.device,
            )
        else:
            exclusion = th.as_tensor(raw_exclusion, dtype=th.float32, device=self.device).reshape(-1)
            if exclusion.numel() != self.config.state_dim:
                raise ValueError(
                    "cert_origin_exclusion must be scalar or match state_dim."
                )
            if (exclusion < 0).any():
                raise ValueError("cert_origin_exclusion must be non-negative.")

        # Ensure the exclusion does not extend outside available bounds around zero.
        max_centered = th.clamp(th.minimum(-self.bounds[0], self.bounds[1]), min=0.0)
        return th.minimum(exclusion, max_centered)

    def _process_regions(
            self, 
            bs: th.Tensor,
            rho: float,
            collect_details: bool,
        ) -> RecursiveCertificationResult:
        """Process one region batch and return step-level certification data.

        Parameters
        ----------
        bs : th.Tensor
            Packed lower and upper bounds of regions with shape ``(n, 2, state_dim)``.
        rho : float
            Lyapunov level-set value to certify.
        collect_details : bool
            If ``False``, returns immediately once any failing region is found.
            If ``True``, subdivides failed regions to collect detailed
            failed boxes and counterexamples.

        Returns
        -------
        RecursiveCertificationResult
            Step-level certified, failed and filtered (counterexample) regions.
        """
        empty_result = RecursiveCertificationResult.empty(
            state_dim=self.config.state_dim, device=self.device
        )
        if len(bs) == 0:
            return empty_result

        filtered_bs = bs[:0]

        if self.config.use_ibp_filter and self.ibp_filter_model is not None:
            bs, filtered_bs = self._filter_sublevel_regions(bs, rho)

        result = empty_result.with_counterexamples(filtered_bs)
        if len(bs) == 0:
            return result

        is_safe = self._certify_batched_regions(
            bs, rho, early_exit=not collect_details
        )

        return result.with_cert_and_failed(bs, is_safe)

    def _certify_regions(
            self, 
            rho: float,
            collect_details: bool = True
    ) -> RegionCertificationResult:
        """Certify all pre-built regions for a fixed ``rho``.

        Parameters
        ----------
        rho : float
            Lyapunov level-set value to test.
        collect_details : bool, optional
            Whether to recursively split failed regions to gather detailed failure
            information, by default ``True``.

        Returns
        -------
        RegionCertificationResult
            Aggregated success flag plus certified regions, failed regions, and
            counterexamples as NumPy arrays.
        """
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}.")

        recursive_result = RecursiveCertificationResult.empty(
            state_dim=self.config.state_dim, device=self.device
        )

        pending_bs = self.regions
        max_depth = self.config.max_recursion_depth if collect_details else 0

        if not collect_details:
            step_result = self._process_regions(
                pending_bs,
                rho,
                collect_details=False,
            )
            recursive_result = recursive_result + step_result
        else:
            with Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                TextColumn("failed: {task.fields[failed]:.0f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    "Certify and split",
                    total=float(max_depth + 1),
                    failed=float(len(pending_bs)),
                )

                for depth in range(max_depth + 1):
                    progress.update(task, failed=float(len(pending_bs)))

                    if len(pending_bs) == 0:
                        break

                    step_result = self._process_regions(
                        pending_bs,
                        rho,
                        collect_details=True,
                    )
                    if depth >= max_depth:
                        recursive_result = recursive_result + step_result
                        progress.advance(task)
                        break

                    recursive_result = recursive_result + step_result.with_failed(step_result.failed[:0])

                    if len(step_result.failed) == 0:
                        progress.advance(task)
                        break

                    pending_bs = self._split_failed_regions(step_result.failed)
                    progress.advance(task)

        certified_regions_np = self._regions_tensor_to_np(recursive_result.certified)
        failed_regions_np = self._regions_tensor_to_np(recursive_result.failed)
        counter_examples_np = self._regions_tensor_to_np(recursive_result.counterexamples)

        return RegionCertificationResult(
            success=failed_regions_np.shape[0] == 0,
            counter_examples=counter_examples_np,
            failed_regions=failed_regions_np,
            certified_regions=certified_regions_np,
        )

    def _scale_rho_up(self, rho_lo: float, rho_up: float) -> tuple[bool, float, float]:
        """Scales up ``rho_lo`` until the new trial is not certified.

        Parameters
        ----------
        rho_lo : float
            Lower bound of rho, updated if trial is certified.
        rho_up : float
            Upper bound of rho, updated if trial is not certified.

        Returns
        -------
        tuple[bool, float, float]
            A tuple of (stop, rho_lo, rho_up) where stop is a boolean indicating if the search should stop,
            and rho_lo and rho_up are the updated bounds.
        """
        trial = rho_up * self.config.rho_scaling
        if self.is_rho_certified(rho=trial):
            return False, trial, trial
        else:
            return True, rho_lo, trial

    def _scale_rho_down(self, rho_lo: float, rho_up: float) -> tuple[bool, float, float]:
        """Scales down ``rho_up`` until the new trial is either certified 
        or it is below rho_min, in which case we stop and return the last certified rho as lower bound.

        Parameters
        ----------
        rho_lo : float
            Lower bound of rho, updated if trial is certified or the trial is lower than ``rho_lo``.
        rho_up : float
            Upper bound of rho, updated if trial is not certified.

        Returns
        -------
        tuple[bool, float, float]
            A tuple of (stop, rho_lo, rho_up) where stop is a boolean indicating if the search should stop,
            and rho_lo and rho_up are the updated bounds.
        """
        trial = max(self.config.rho_min, rho_up / self.config.rho_scaling)
        if self.is_rho_certified(rho=trial):
            return True, trial, rho_up
         
        if trial <= self.config.rho_min:
            return True, None, trial
        return False, rho_lo, trial

    def _bisect_rho(self, rho_lo: float, rho_up: float) -> tuple[bool, float, float]:
        """Uses the midpoint of rho upper and lower for bisection.

        Parameters
        ----------
        rho_lo : float
            Lower bound of rho (known to be certified).
        rho_up : float
            Upper bound of rho (known to be **not** certified).

        Returns
        -------
        tuple[bool, float, float]
            A tuple of (stop, rho_lo, rho_up) where stop is a boolean indicating if the search should stop,
            and rho_lo and rho_up are the updated bounds.
        """
        rho_mid = 0.5 * (rho_lo + rho_up)

        if self.is_rho_certified(rho=rho_mid):
            rho_lo = rho_mid
        else:
            rho_up = rho_mid

        if rho_up - rho_lo <= self.config.bisection_tol:
            return True, rho_lo, rho_up
        else:
            return False, rho_lo, rho_up
    
    def _build_regions(self) -> th.Tensor:
        """Build packed axis-aligned grid cells excluding the origin window.

        Returns
        -------
        th.Tensor
            Region bounds with shape ``(n, 2, state_dim)``.
        """
        origin_exclusion = self._resolve_origin_exclusion()
        lbx, ubx = self.bounds[0], self.bounds[1]
        bin_counts = self.config.bins_per_dim
        refinement_factors = self.config.center_refinement_factor

        lb_axes = []
        ub_axes = []
        for idx in range(self.config.state_dim):
            edges = self._build_center_refined_axis_edges(
                lb=lbx[idx],
                ub=ubx[idx],
                num_bins=int(bin_counts[idx]),
                refinement_factor=float(refinement_factors[idx]),
            )
            lb_axes.append(edges[:-1])
            ub_axes.append(edges[1:])

        lb_mesh = th.meshgrid(*lb_axes, indexing="ij")
        ub_mesh = th.meshgrid(*ub_axes, indexing="ij")
        lbs = th.stack([axis.flatten() for axis in lb_mesh], dim=1)
        ubs = th.stack([axis.flatten() for axis in ub_mesh], dim=1)

        overlaps_origin_per_dim = (lbs < origin_exclusion.unsqueeze(0)) & (ubs > -origin_exclusion.unsqueeze(0))
        overlaps_origin = overlaps_origin_per_dim.all(dim=1)
        valid_mask = ~overlaps_origin

        lbs = lbs[valid_mask]
        ubs = ubs[valid_mask]
        return self._pack_regions(lbs, ubs)

    def is_rho_certified(self, rho: float) -> bool:
        """Check whether all regions satisfy Lyapunov conditions at ``rho``.

        Parameters
        ----------
        rho : float
            Lyapunov level-set value to test.

        Returns
        -------
        bool
            ``True`` if certification succeeds for all regions.
        """
        result = self._certify_regions(rho=rho, collect_details=False) 
        return result.success

    def find_max_rho(self, rho_estimate: float) -> tuple[float, RegionCertificationResult]:
        """Search for the largest certifiable rho and return details.

        Parameters
        ----------
        rho_estimate : float
            Positive initial guess for rho search.

        Returns
        -------
        tuple[float, RegionCertificationResult]
            Best certified ``rho`` and detailed region-level certification result.

        Raises
        ------
        ValueError
            If ``rho_estimate`` is not positive.
        """
        if rho_estimate <= 0:
            raise ValueError("rho_estimate must be positive.")

        __logger__.info(f"Starting Lyapunov certification with {self.config.cert_method.upper()} method.", )

        self.verifier = self._setup_verifier()
        self.regions = self._build_regions()
        __logger__.info(f"Built {len(self.regions)} certification regions (state_dim={self.config.state_dim}).")
        self.setup_backend()

        if self.config.rho_min > rho_estimate:
            __logger__.warning(
                f"Provided rho_estimate ({rho_estimate:.4f}) is below rho_min ({self.config.rho_min:.4f}). Starting search from rho_min."
            )
            initial_rho = self.config.rho_min
        else:
            initial_rho = float(rho_estimate)

        initial_ok = self.is_rho_certified(rho=initial_rho)

        # Scale up
        if initial_ok:
            rho_lo, rho_up = self._iterative_rho_search(
                total=self.config.max_scale_steps,
                desc="Scale up",
                initial_values=(initial_rho, initial_rho),
                step_fn=self._scale_rho_up
            )
            
            if rho_lo == rho_up:
                __logger__.warning(f"Maximum scaling steps ({self.config.max_scale_steps}) reached without finding an upper bound.")

        # Scale down
        else:
            rho_lo, rho_up = self._iterative_rho_search(
                total=self.config.max_scale_steps,
                desc="Scale down",
                initial_values=(None, initial_rho),
                step_fn=self._scale_rho_down
            )

            # Fallback
            if rho_lo is None:
                rho_lo = self.config.rho_min
                if rho_up <= self.config.rho_min or not self.is_rho_certified(rho=self.config.rho_min):
                    __logger__.error(f"Could not even certify rho_min ({self.config.rho_min:.0e}).")
                    rho_up = self.config.rho_min # no bisection

        # Bisect
        if rho_up - rho_lo >= self.config.bisection_tol:
            rho_lo, rho_up = self._iterative_rho_search(
                total=self.config.max_bisection_steps,
                desc="Bisect rho",
                initial_values=(rho_lo, rho_up),
                step_fn=self._bisect_rho
            )

        details = self._certify_regions(rho=rho_lo, collect_details=True)
        return rho_lo, details
