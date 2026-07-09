from __future__ import annotations

import logging
import numpy as np
import torch as th
import torch.nn as nn

from dataclasses import dataclass, replace
from typing import Sequence
from copy import deepcopy
from numpy.typing import NDArray

from .config import LyapunovCertificationConfig
from .region_manager import RegionManager
from .progress import CertificationProgress, ProgressLevel
from .abcrown_region_certifier import CompleteABCrownCertifier, CoreABCrownCertifier, EarlyExitLevel
from .lirpa_lyapunov_bounds import LiRPALyapunovRegionBounds, LyapunovRegionBounds
from ..utils.region_builder import RegionBuilder
from ..utils.constants import *

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecursiveCertificationResult:
    """Internal tensor result for recursive region certification."""

    resolved: th.Tensor
    unresolved: th.Tensor
    irrelevant: th.Tensor
    counterexample_found: bool = False

    @property
    def vacuous(self) -> bool:
        """Whether no regions remain relevant."""
        return self.resolved.numel() == 0 and self.unresolved.numel() == 0

    @property
    def global_success(self) -> bool:
        """Whether all regions inside ``V(x) <= rho`` were certified."""
        return self.unresolved.numel() == 0 and self.resolved.numel() > 0

    @property
    def partial_success(self) -> bool:
        """Whether at least one region is inside the tested sublevel set and certified."""
        return self.resolved.numel() > 0

    @classmethod
    def empty(cls, state_dim: int, device: th.device) -> RecursiveCertificationResult:
        empty = th.empty((0, 2, state_dim), device=device)
        return cls(
            resolved=empty,
            unresolved=empty,
            irrelevant=empty,
            counterexample_found=False,
        )

    def with_unresolved(self, regions: th.Tensor) -> RecursiveCertificationResult:
        return replace(self, unresolved=regions)

    def __add__(self, other: RecursiveCertificationResult) -> RecursiveCertificationResult:
        return RecursiveCertificationResult(
            resolved=th.cat([self.resolved, other.resolved], dim=0),
            unresolved=th.cat([self.unresolved, other.unresolved], dim=0),
            irrelevant=th.cat([self.irrelevant, other.irrelevant], dim=0),
            counterexample_found=self.counterexample_found or other.counterexample_found,
        )




class RecursiveCertifier:
    """Lyapunov certifier with recursive region splitting."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovCertificationConfig,
        device: th.device = th.device("cpu"),
        progress_level: int | ProgressLevel = ProgressLevel.ALL,
    ):
        """Initialize a Lyapunov certifier.

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
        progress_level : int | ProgressLevel, optional
            Progress verbosity level: ``0`` disables progress bars, ``1`` shows
            only the outermost task, and ``2`` enables nested tasks.
        """
        self.config = config
        self.device = device

        self.progress = CertificationProgress(progress_level)
        self.progress.log_initialization(self.__class__.__name__)

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()

        self.bounds = self._resolve_bounds(config.cert_bounds, device)

        self.bounder: LiRPALyapunovRegionBounds | None = None
        self.certifier: CompleteABCrownCertifier | None = None
        self.core_certifier: CoreABCrownCertifier | None = None
        self.region_manager = RegionManager(
            region_builder=self._build_region_builder(),
        )
        self.details = None


    # ==========================================
    # BUILDER AND GETTER
    # ==========================================
    def _build_region_bounder(self) -> LiRPALyapunovRegionBounds:
        """Construct a LiRPA helper for classifying regions against ``V(x) <= rho``."""
        return LiRPALyapunovRegionBounds(
            lyap_model=self.lyap_model,
            state_dim=self.config.state_dim,
            batch_size=self.config.batch_size,
            default_bound_method=self.config.lirpa_method,
            use_affine_l1_lower_bound=self.config.use_affine_l1_sublevel_bounds,
            device=self.device,
        )

    def _get_region_bounder(self) -> LiRPALyapunovRegionBounds:
        """Return the cached LiRPA region bounder."""
        if self.bounder is None:
            self.bounder = self._build_region_bounder()
        return self.bounder

    def _build_region_certifier(self) -> CompleteABCrownCertifier:
        """Construct the shared per-region ABCrown certifier."""
        return CompleteABCrownCertifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            config=self.config,
            device=self.device,
        )

    def _get_region_certifier(self) -> CompleteABCrownCertifier:
        """Return the cached ABCrown region certifier."""
        if self.certifier is None:
            self.certifier = self._build_region_certifier()
        return self.certifier

    def _build_core_region_certifier(self) -> CoreABCrownCertifier:
        """Construct the shared per-region core ABCrown certifier."""
        return CoreABCrownCertifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            config=self.config,
            device=self.device,
        )

    def _get_core_region_certifier(self) -> CoreABCrownCertifier:
        """Return the cached core ABCrown certifier."""
        if self.core_certifier is None:
            self.core_certifier = self._build_core_region_certifier()
        return self.core_certifier

    def _build_region_builder(self) -> RegionBuilder:
        """Construct a region builder for the current certification bounds."""
        return RegionBuilder(
            bounds=self.bounds,
            bins_per_dim=self.config.bins_per_dim,
            center_refinement_factor=self.config.center_refinement_factor,
            origin_exclusion=self.config.origin_exclusion,
            split_dim_weights=self.config.split_dim_weights,
            device=self.device,
        )

    
    # ==========================================
    # REGION UTILS
    # ==========================================
    @property
    def regions(self) -> th.Tensor | None:
        return self.region_manager.regions

    @regions.setter
    def regions(self, regions: th.Tensor | None) -> None:
        if regions is None:
            self.region_manager.clear_regions()
            return
        self.region_manager.set_regions(regions)

    def _ensure_region_bounds(
        self,
        regions: th.Tensor | None = None,
        *,
        make_current: bool,
    ) -> tuple[th.Tensor, LyapunovRegionBounds]:
        """Return region bounds, computing and caching them on demand."""
        target_regions, cached_region_bounds = self.region_manager.ensure_cached(
            regions,
            make_current=make_current,
        )
        if cached_region_bounds is not None:
            return target_regions, cached_region_bounds

        computed_region_bounds = self._get_region_bounder().compute_bounds_for_regions(
            target_regions,
            method=self.config.lirpa_method,
        )

        cached_region_bounds = self.region_manager.cache_region_bounds(
            computed_region_bounds,
            regions=target_regions,
            make_current=make_current,
        )
        return target_regions, cached_region_bounds

    def cache_region_bounds(self, regions: th.Tensor | None = None) -> LyapunovRegionBounds:
        """Compute and cache V bounds for the current root certification regions."""
        _, region_bounds = self._ensure_region_bounds(
            regions,
            make_current=True,
        )
        return region_bounds


    # =========================================
    # HELPERS
    # =========================================
    @staticmethod
    def _resolve_bounds(bounds: Sequence[float], device: th.device) -> th.Tensor:
        """Convert state bounds to a tensor on the target device."""
        bounds = th.as_tensor(bounds, dtype=th.float32, device=device)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("bounds must be a sequence of shape (2, nx) [lb, ub].")
        return bounds

    def _regions_tensor_to_np(self, regions: th.Tensor) -> NDArray:
        """Convert a region tensor ``(N, 2, state_dim)`` to NumPy."""
        if regions.numel() == 0:
            return np.empty((0, 2, self.config.state_dim), dtype=np.float32)
        return regions.cpu().numpy()



    # ========================================
    # CERTIFICATION
    # ========================================
    def _run_core_certification(
        self,
        regions: th.Tensor,
        rho: float,
        *,
        early_exit: EarlyExitLevel,
    ):
        if len(regions) == 0:
            return None
        result = self._get_core_region_certifier().certify_regions(
            regions=regions,
            rho=rho,
            early_exit=early_exit,
            progress=self.progress,
        )
        return self.region_manager.apply_core_certification_result(
            regions,
            verified_mask=result.verified_mask,
            counterexample_mask=result.counterexample_mask,
            unknown_mask=result.unknown_mask,
        )

    def _run_complete_certification(
        self,
        regions: th.Tensor,
        rho: float,
        *,
        early_exit: EarlyExitLevel,
    ):
        if len(regions) == 0:
            return None
        result = self._get_region_certifier().certify_regions(
            regions=regions,
            rho=rho,
            early_exit=early_exit,
            progress=self.progress,
        )
        update = self.region_manager.apply_complete_certification_result(
            regions,
            verified_mask=result.verified_mask,
            failed_mask=result.failed_mask,
            rho=rho,
        )
        return result, update

    def _process_regions(
            self, 
            bs: th.Tensor,
            rho: float,
            *,
            early_exit: EarlyExitLevel,
        ) -> RecursiveCertificationResult:
        """Process one region batch and return step-level certification data.

        Parameters
        ----------
        bs : th.Tensor
            Packed lower and upper bounds of regions with shape ``(n, 2, state_dim)``.
        rho : float
            Lyapunov level-set value to certify.
        early_exit : bool
            If ``True``, returns immediately once any failing region is found.

        Returns
        -------
        RecursiveCertificationResult
            Step-level resolved, unresolved and irrelevant regions.
        """
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}.")

        resolved_parts: list[th.Tensor] = []
        unresolved_parts: list[th.Tensor] = []
        irrelevant_regions = self.region_manager.empty_regions()
        counterexample_found = False
        
        def append_resolved(regions: th.Tensor, update_progress: bool = True) -> None:
            if len(regions) > 0:
                resolved_parts.append(regions)
                if update_progress:
                    self.progress.add_recursive_counts(resolved=len(regions), pending=-len(regions))

        def append_unresolved(regions: th.Tensor, update_progress: bool = True) -> None:
            if len(regions) > 0:
                unresolved_parts.append(regions)
                if update_progress:
                    self.progress.add_recursive_counts(unresolved=len(regions), pending=-len(regions))

        def counterexample_found_or(condition: bool):
            nonlocal counterexample_found
            if early_exit != EarlyExitLevel.NONE:
                counterexample_found = counterexample_found or condition

        def finish() -> RecursiveCertificationResult:
            return RecursiveCertificationResult(
                resolved=self.region_manager.pack_regions(resolved_parts),
                unresolved=self.region_manager.pack_regions(unresolved_parts),
                irrelevant=irrelevant_regions,
                counterexample_found=counterexample_found,
            )

        if len(bs) == 0:
            return finish()

        bs, region_bounds = self._ensure_region_bounds(bs, make_current=bs is self.regions)
        partition = self.region_manager.partition_certification_regions(
            bs,
            region_bounds=region_bounds,
            rho=rho,
            sublevel_tolerance=self.config.sublevel_tolerance,
        )

        irrelevant_regions = partition.irrelevant_regions
        if len(irrelevant_regions) > 0:
            self.progress.add_recursive_counts(irrelevant=len(irrelevant_regions), pending=-len(irrelevant_regions))
            
        if not partition.has_relevant_regions:
            return finish()

        append_resolved(partition.cached_complete_safe_regions)
        append_resolved(partition.cached_core_safe_regions)

        cached_inside_core_cex_bs = partition.cached_inside_counterexample_regions
        if len(cached_inside_core_cex_bs) > 0:
            append_unresolved(cached_inside_core_cex_bs)
            counterexample_found = True
            if early_exit != EarlyExitLevel.NONE:
                return finish()

        append_unresolved(partition.cached_inside_unknown_regions)
        if early_exit == EarlyExitLevel.ON_UNKNOWN and len(partition.cached_inside_unknown_regions) > 0:
            counterexample_found = True
            return finish()

        inside_core_unchecked_bs = partition.inside_core_unchecked_regions
        boundary_core_unchecked_bs = partition.boundary_core_unchecked_regions
        boundary_complete_candidates = partition.boundary_complete_candidate_regions

        if self.config.skip_boundary_core_cert and len(boundary_core_unchecked_bs) > 0:
            boundary_complete_candidates = self.region_manager.pack_regions(
                (boundary_complete_candidates, boundary_core_unchecked_bs))
            boundary_core_unchecked_bs = self.region_manager.empty_regions()

        inside_core_update = self._run_core_certification(
            inside_core_unchecked_bs,
            rho,
            early_exit=early_exit,
        )
        if inside_core_update is not None:
            append_resolved(inside_core_update.verified_regions, update_progress=False)
            append_unresolved(inside_core_update.failed_regions, update_progress=False)
            if early_exit == EarlyExitLevel.ON_UNKNOWN and len(inside_core_update.failed_regions) > 0:
                counterexample_found = True
                return finish()
            counterexample_found_or(inside_core_update.counterexample_found)
            if early_exit != EarlyExitLevel.NONE and inside_core_update.counterexample_found:
                return finish()

        boundary_core_update = self._run_core_certification(
            boundary_core_unchecked_bs,
            rho,
            early_exit=EarlyExitLevel.NONE,
        )
        if boundary_core_update is not None:
            append_resolved(boundary_core_update.verified_regions, update_progress=False)
            boundary_complete_candidates = self.region_manager.pack_regions(
                (boundary_complete_candidates, boundary_core_update.failed_regions))

        if len(boundary_complete_candidates) == 0:
            return finish()

        complete_result, complete_update = self._run_complete_certification(
            boundary_complete_candidates,
            rho,
            early_exit=early_exit,
        )
        if complete_update is not None and complete_result is not None:
            append_resolved(complete_update.verified_regions, update_progress=False)
            append_unresolved(complete_update.failed_regions, update_progress=False)
            if early_exit == EarlyExitLevel.ON_UNKNOWN and len(complete_update.failed_regions) > 0:
                counterexample_found = True
                return finish()
            counterexample_found_or(complete_result.any_counterexample)
        return finish()

    def _certify_recursive_regions(
        self,
        rho: float,
        *,
        early_exit: EarlyExitLevel = EarlyExitLevel.NONE,
        force_display: bool = False,
    ) -> RecursiveCertificationResult:
        """Run recursive certification for a fixed ``rho`` over all regions."""
        empty_result = RecursiveCertificationResult.empty(
            state_dim=self.config.state_dim,
            device=self.device,
        )
        recursive_result = deepcopy(empty_result)
        
        def finish_with_counterexample() -> RecursiveCertificationResult:
            return recursive_result

        max_depth = self.config.max_recursion_depth
        recursive_result = replace(recursive_result, unresolved=self.region_manager.ensure_regions())

        with self.progress:
            self.progress.start_recursive("Region Splits", max_depth, force_display=force_display)

            try:
                for depth in range(max_depth + 1):
                    current_early_exit = early_exit
                    if early_exit == EarlyExitLevel.ON_COUNTEREXAMPLE and depth == max_depth:
                        current_early_exit = EarlyExitLevel.ON_UNKNOWN
                        
                    pending_bs = recursive_result.unresolved
                    
                    self.progress.update_recursive(
                        advance=0,
                        is_completed=False,
                        max_depth=max_depth,
                        n_pending=len(pending_bs),
                    )

                    if len(pending_bs) == 0:
                        break

                    recursive_result = replace(
                        recursive_result, unresolved=self.region_manager.empty_regions()
                    )

                    for partition in th.split(pending_bs, self.config.batch_size):
                        step_result = self._process_regions(
                            partition,
                            rho,
                            early_exit=current_early_exit,
                        )

                        if current_early_exit != EarlyExitLevel.NONE and step_result.counterexample_found:
                            return finish_with_counterexample()
                        
                        recursive_result = recursive_result + step_result
                    recursive_result = recursive_result.with_unresolved(recursive_result.unresolved[:0])

                    if len(recursive_result.unresolved) == 0:
                        self.progress.update_recursive(is_completed=True, max_depth=max_depth, n_pending=0)
                        break

                    pending_bs, terminal_failed_bs = self.region_manager.split_failed_regions_on_certification_frontier(
                        step_result.unresolved,
                        recursive_result.resolved,
                    )
                    if len(terminal_failed_bs) > 0:
                        recursive_result = recursive_result + empty_result.with_unresolved(terminal_failed_bs)

                    if len(pending_bs) == 0:
                        self.progress.update_recursive(is_completed=True, max_depth=max_depth, n_pending=0)
                        break

                    self.progress.update_recursive(advance=1)
            finally:
                self.progress.stop_recursive()

        return recursive_result