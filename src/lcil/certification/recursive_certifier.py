from __future__ import annotations

import logging
import numpy as np
import torch as th
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, TypeVar
from numpy.typing import NDArray

from .config import LyapunovCertificationConfig
from .region_manager import CertificationRegionPartition, CoreStatus, RegionManager
from .progress import CertificationProgress, ProgressLevel
from .abcrown_region_certifier import (
    CompleteABCrownCertifier,
    CoreABCrownCertifier,
    EarlyExitLevel,
)
from .lirpa_lyapunov_bounds import LiRPALyapunovRegionBounds, LyapunovRegionBounds
from ..utils.region_builder import RegionBuilder

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecursiveCertificationResult:
    """Internal tensor result for recursive region certification."""

    resolved: th.Tensor
    unresolved: th.Tensor
    irrelevant: th.Tensor
    counterexample_found: bool = False

    @property
    def global_success(self) -> bool:
        """Whether all regions inside ``V(x) <= rho`` were certified."""
        return self.unresolved.numel() == 0 and self.resolved.numel() > 0

    @property
    def partial_success(self) -> bool:
        """Whether at least one region is inside the tested sublevel set and certified."""
        return self.resolved.numel() > 0


_CertifierT = TypeVar("_CertifierT", CompleteABCrownCertifier, CoreABCrownCertifier)


@dataclass
class _RecursiveLoopResult:
    """Internal result container for recursive certification loops."""

    all_resolved: list[th.Tensor] = field(default_factory=list)
    unresolved_leaves: list[th.Tensor] = field(default_factory=list)
    unresolved_non_leaves: list[th.Tensor] = field(default_factory=list)
    all_irrelevant: list[th.Tensor] = field(default_factory=list)
    counterexample_found: bool = False

    @property
    def all_unresolved(self) -> list[th.Tensor]:
        return self.unresolved_non_leaves + self.unresolved_leaves


class _StepCollector:
    """Accumulates resolved, unresolved, and irrelevant regions for a step while updating progress."""

    def __init__(self, region_manager: RegionManager, progress: CertificationProgress) -> None:
        self.rm = region_manager
        self.progress = progress
        self.resolved_parts: list[th.Tensor] = []
        self.unresolved_parts: list[th.Tensor] = []
        self.irrelevant: th.Tensor = region_manager.empty_regions()
        self.counterexample_found: bool = False

    def add_resolved(self, regions: th.Tensor, update_progress: bool = True) -> None:
        if len(regions) > 0:
            self.resolved_parts.append(regions)
            if update_progress:
                self.progress.add_recursive_counts(resolved=len(regions), pending=-len(regions))

    def add_unresolved(self, regions: th.Tensor, update_progress: bool = True) -> None:
        if len(regions) > 0:
            self.unresolved_parts.append(regions)
            if update_progress:
                self.progress.add_recursive_counts(unresolved=len(regions), pending=-len(regions))

    def set_irrelevant(self, regions: th.Tensor) -> None:
        self.irrelevant = regions
        if len(regions) > 0:
            self.progress.add_recursive_counts(irrelevant=len(regions), pending=-len(regions))

    def record_complete_update(
        self,
        result: Any,
        update: Any,
        early_exit: EarlyExitLevel,
    ) -> None:
        """Record results from complete specification verification."""
        if update is not None and result is not None:
            self.add_resolved(update.verified_regions)
            self.add_unresolved(update.failed_regions)
            if early_exit == EarlyExitLevel.ON_UNKNOWN and len(update.failed_regions) > 0:
                self.counterexample_found = True
            elif early_exit != EarlyExitLevel.NONE and result.any_counterexample:
                self.counterexample_found = True

    def build_result(self) -> RecursiveCertificationResult:
        return RecursiveCertificationResult(
            resolved=self.rm.pack_regions(self.resolved_parts),
            unresolved=self.rm.pack_regions(self.unresolved_parts),
            irrelevant=self.irrelevant,
            counterexample_found=self.counterexample_found,
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
    def _get_region_bounder(self) -> LiRPALyapunovRegionBounds:
        """Return the cached LiRPA region bounder."""
        if self.bounder is None:
            self.bounder = LiRPALyapunovRegionBounds(
                lyap_model=self.lyap_model,
                state_dim=self.config.state_dim,
                batch_size=self.config.batch_size,
                default_bound_method=self.config.lirpa_method,
                use_affine_l1_lower_bound=self.config.use_affine_l1_sublevel_bounds,
                device=self.device,
            )
        return self.bounder

    def _build_certifier(self, certifier_cls: type[_CertifierT]) -> _CertifierT:
        """Construct a certifier instance configured with models and device."""
        return certifier_cls(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            config=self.config,
            device=self.device,
        )

    def _get_region_certifier(self) -> CompleteABCrownCertifier:
        """Return the cached ABCrown region certifier."""
        if self.certifier is None:
            self.certifier = self._build_certifier(CompleteABCrownCertifier)
        return self.certifier

    def _get_core_region_certifier(self) -> CoreABCrownCertifier:
        """Return the cached core ABCrown certifier."""
        if self.core_certifier is None:
            self.core_certifier = self._build_certifier(CoreABCrownCertifier)
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
    def _resolve_bounds(bounds: Any, device: th.device) -> th.Tensor:
        """Convert state bounds to a tensor of shape (2, nx) on target device."""
        b = th.as_tensor(bounds, dtype=th.float32, device=device)
        if b.ndim != 2 or b.shape[0] != 2:
            raise ValueError("bounds must be a sequence of shape (2, nx) [lb, ub].")
        return b

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
        rho: float = 0.0,
        *,
        early_exit: EarlyExitLevel = EarlyExitLevel.NONE,
    ):
        del early_exit, rho
        if len(regions) == 0:
            return None
        result = self._get_core_region_certifier().certify_regions(
            regions=regions,
            rho=0.0,
            early_exit=EarlyExitLevel.NONE,
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
        is_leaf: bool = False,
    ):
        if len(regions) == 0:
            return None, None
        result = self._get_region_certifier().certify_regions(
            regions=regions,
            rho=rho,
            early_exit=early_exit,
            progress=self.progress,
            is_leaf=is_leaf,
        )
        update = self.region_manager.apply_complete_certification_result(
            regions,
            verified_mask=result.verified_mask,
            failed_mask=result.failed_mask,
            rho=rho,
        )
        return result, update

    def _prepare_step_partition(
        self,
        bs: th.Tensor,
        rho: float,
    ) -> tuple[CertificationRegionPartition | None, _StepCollector]:
        """Validate inputs, partition regions against rho, and record cached safe/irrelevant regions.

        Returns (partition, collector). If partition is None or has no relevant regions,
        the collector is ready to build the final step result immediately.
        """
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}.")

        collector = _StepCollector(self.region_manager, self.progress)
        if len(bs) == 0:
            return None, collector

        bs, region_bounds = self._ensure_region_bounds(bs, make_current=bs is self.regions)
        partition = self.region_manager.partition_certification_regions(
            bs,
            region_bounds=region_bounds,
            rho=rho,
            sublevel_tolerance=self.config.sublevel_tolerance,
        )

        collector.set_irrelevant(partition.irrelevant_regions)

        if not partition.has_relevant_regions:
            return None, collector

        collector.add_resolved(partition.cached_complete_safe_regions)
        collector.add_resolved(partition.cached_core_safe_regions)
        return partition, collector

    def _process_core_regions(
        self,
        bs: th.Tensor,
        rho: float | None = None,
    ) -> RecursiveCertificationResult:
        """Process one region batch with the rho-independent Core Check.

        Evaluates dV + kappa*V <= 0 and next_step in cert_bounds.
        Cached safe regions are resolved immediately.
        Unchecked regions undergo Core Check.
        Failing regions (counterexample or unknown) are marked unresolved.
        """
        del rho
        collector = _StepCollector(self.region_manager, self.progress)
        if len(bs) == 0:
            return collector.build_result()

        bs, _ = self._ensure_region_bounds(bs, make_current=bs is self.regions)

        core_status = self.region_manager.get_core_status(bs)
        collector.add_resolved(bs[core_status == CoreStatus.SAFE])

        unchecked_bs = bs[core_status == CoreStatus.UNCHECKED]
        if len(unchecked_bs) > 0:
            core_update = self._run_core_certification(unchecked_bs)
            if core_update is not None:
                collector.add_resolved(core_update.verified_regions)
                collector.add_unresolved(core_update.failed_regions)
                __logger__.info(
                    "Core Check completed on %d regions: %d safe, %d failed.",
                    len(unchecked_bs),
                    len(core_update.verified_regions),
                    len(core_update.failed_regions),
                )

        collector.add_unresolved(
            bs[(core_status == CoreStatus.COUNTEREXAMPLE) | (core_status == CoreStatus.UNKNOWN)]
        )
        return collector.build_result()

    def _process_complete_regions(
        self,
        bs: th.Tensor,
        rho: float,
        *,
        early_exit: EarlyExitLevel | bool = EarlyExitLevel.NONE,
        is_leaf: bool = False,
    ) -> RecursiveCertificationResult:
        """Process one region batch with the complete specification verifier."""
        if isinstance(early_exit, bool):
            early_exit = EarlyExitLevel.ON_COUNTEREXAMPLE if early_exit else EarlyExitLevel.NONE

        partition, collector = self._prepare_step_partition(bs, rho)
        if partition is None:
            return collector.build_result()

        to_verify_parts = [
            partition.inside_core_unchecked_regions,
            partition.cached_inside_counterexample_regions,
            partition.cached_inside_unknown_regions,
            partition.boundary_core_unchecked_regions,
            partition.boundary_complete_candidate_regions,
        ]
        regions_to_verify = self.region_manager.pack_regions(to_verify_parts)

        if len(regions_to_verify) == 0:
            return collector.build_result()

        complete_result, complete_update = self._run_complete_certification(
            regions_to_verify,
            rho,
            early_exit=early_exit,
            is_leaf=is_leaf,
        )
        collector.record_complete_update(complete_result, complete_update, early_exit)
        return collector.build_result()

    def _run_recursive_loop(
        self,
        title: str,
        pending_bs: th.Tensor,
        step_fn: Callable[[th.Tensor, int, bool], RecursiveCertificationResult],
        *,
        start_depth: int = 0,
        early_exit: EarlyExitLevel = EarlyExitLevel.NONE,
        on_resolved: Callable[[], Any] | None = None,
        force_display: bool = False,
    ) -> _RecursiveLoopResult:
        """Execute a depth-bounded recursive certification loop over regions."""
        max_depth = self.config.max_recursion_depth
        all_resolved: list[th.Tensor] = []
        unresolved_leaves: list[th.Tensor] = []
        unresolved_non_leaves: list[th.Tensor] = []
        all_irrelevant: list[th.Tensor] = []
        counterexample_found = False

        if len(pending_bs) == 0:
            return _RecursiveLoopResult()

        def _update_progress(advance: int = 0, n_pending: int = 0, is_completed: bool = False) -> None:
            n_unresolved = sum(len(u) for u in unresolved_leaves + unresolved_non_leaves)
            self.progress.update_recursive(
                advance=advance,
                n_pending=n_pending,
                n_unresolved=n_unresolved,
                is_completed=is_completed,
                max_depth=max_depth if is_completed else 0,
            )

        with self.progress:
            self.progress.start_recursive(title, max_depth, force_display=force_display)
            self.progress.update_recursive(n_pending=len(pending_bs))

            try:
                for depth in range(start_depth, max_depth + 1):
                    if len(pending_bs) == 0:
                        _update_progress(is_completed=True)
                        break

                    is_leaf = (depth == max_depth)
                    step_result = step_fn(pending_bs, depth, is_leaf)

                    if len(step_result.resolved) > 0:
                        all_resolved.append(step_result.resolved)
                        if on_resolved is not None:
                            on_resolved()
                    if len(step_result.irrelevant) > 0:
                        all_irrelevant.append(step_result.irrelevant)
                    if step_result.counterexample_found:
                        counterexample_found = True

                    if early_exit != EarlyExitLevel.NONE and step_result.counterexample_found:
                        if len(step_result.unresolved) > 0:
                            unresolved_leaves.append(step_result.unresolved)
                        break

                    if len(step_result.unresolved) == 0:
                        _update_progress(is_completed=True)
                        break

                    if is_leaf:
                        unresolved_leaves.append(step_result.unresolved)
                        _update_progress(advance=1)
                        break

                    split_bs = self.region_manager.split_regions(step_result.unresolved)
                    if len(split_bs) == 0:
                        unresolved_non_leaves.append(step_result.unresolved)
                        _update_progress(advance=1)
                        break

                    pending_bs = split_bs
                    _update_progress(advance=1, n_pending=len(pending_bs))
            finally:
                self.progress.stop_recursive()

        return _RecursiveLoopResult(
            all_resolved=all_resolved,
            unresolved_leaves=unresolved_leaves,
            unresolved_non_leaves=unresolved_non_leaves,
            all_irrelevant=all_irrelevant,
            counterexample_found=counterexample_found,
        )

    def _core_recursive_certify(
        self,
        pending_bs: th.Tensor | None = None,
        rho: float | None = None,
        *,
        force_display: bool = False,
    ) -> tuple[list[th.Tensor], list[th.Tensor], list[th.Tensor]]:
        """Run recursive Core Check certification up to max_recursion_depth.

        Returns
        -------
        tuple[list[th.Tensor], list[th.Tensor], list[th.Tensor]]
            (all_resolved, unresolved_leaves, unresolved_non_leaves)
        """
        del rho
        if pending_bs is None:
            pending_bs = self.region_manager.ensure_regions()

        loop_res = self._run_recursive_loop(
            title="Core Split",
            pending_bs=pending_bs,
            step_fn=lambda bs, depth, is_leaf: self._process_core_regions(bs),
            start_depth=0,
            early_exit=EarlyExitLevel.NONE,
            on_resolved=self.region_manager.propagate_core_safe_to_parents,
            force_display=force_display,
        )
        return (
            loop_res.all_resolved,
            loop_res.unresolved_leaves,
            loop_res.unresolved_non_leaves,
        )

    def _complete_recursive_certify(
        self,
        pending_bs: th.Tensor,
        rho: float,
        *,
        start_depth: int = 0,
        early_exit: EarlyExitLevel = EarlyExitLevel.NONE,
        force_display: bool = False,
    ) -> tuple[list[th.Tensor], list[th.Tensor], list[th.Tensor], bool]:
        """Run complete specification certification with recursive splitting if needed.

        Returns
        -------
        tuple[list[th.Tensor], list[th.Tensor], list[th.Tensor], bool]
            (all_resolved, all_unresolved, all_irrelevant, counterexample_found)
        """
        loop_res = self._run_recursive_loop(
            title="Complete Split",
            pending_bs=pending_bs,
            step_fn=lambda bs, depth, is_leaf: self._process_complete_regions(
                bs,
                rho,
                early_exit=(
                    EarlyExitLevel.ON_UNKNOWN
                    if early_exit == EarlyExitLevel.ON_COUNTEREXAMPLE and is_leaf
                    else early_exit
                ),
                is_leaf=is_leaf,
            ),
            start_depth=start_depth,
            early_exit=early_exit,
            force_display=force_display,
        )
        return (
            loop_res.all_resolved,
            loop_res.all_unresolved,
            loop_res.all_irrelevant,
            loop_res.counterexample_found,
        )

    def _pack_result(
        self,
        resolved: Sequence[th.Tensor],
        unresolved: Sequence[th.Tensor],
        irrelevant: Sequence[th.Tensor],
        counterexample_found: bool = False,
    ) -> RecursiveCertificationResult:
        """Pack lists of region tensors into a RecursiveCertificationResult."""
        return RecursiveCertificationResult(
            resolved=self.region_manager.pack_regions(resolved),
            unresolved=self.region_manager.pack_regions(unresolved),
            irrelevant=self.region_manager.pack_regions(irrelevant),
            counterexample_found=counterexample_found,
        )

    def _certify_recursive_regions(
        self,
        rho: float,
        *,
        early_exit: EarlyExitLevel | bool = EarlyExitLevel.NONE,
        force_display: bool = False,
        show_progress: bool | None = None,
    ) -> RecursiveCertificationResult:
        """Run recursive certification for a fixed ``rho`` over all regions."""
        del show_progress
        if isinstance(early_exit, bool):
            early_exit = EarlyExitLevel.ON_COUNTEREXAMPLE if early_exit else EarlyExitLevel.NONE

        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}.")

        if self.config.skip_core_cert:
            comp_resolved, comp_unresolved, comp_irrelevant, cex_found = (
                self._complete_recursive_certify(
                    self.region_manager.ensure_regions(),
                    rho,
                    start_depth=0,
                    early_exit=early_exit,
                    force_display=force_display,
                )
            )
            return self._pack_result(comp_resolved, comp_unresolved, comp_irrelevant, cex_found)

        # Phase 1: Core recursive certification up to max_recursion_depth (all core regions + splits)
        core_resolved, unresolved_leaves, unresolved_non_leaves = (
            self._core_recursive_certify(
                force_display=force_display,
            )
        )

        all_core_unresolved = unresolved_non_leaves + unresolved_leaves
        pending_leaves = self.region_manager.pack_regions(unresolved_leaves)
        if len(pending_leaves) == 0:
            return self._pack_result(core_resolved, all_core_unresolved, [], False)

        # Phase 2: Complete certification on remaining unresolved leaves
        comp_resolved, comp_unresolved, comp_irrelevant, cex_found = (
            self._complete_recursive_certify(
                pending_leaves,
                rho,
                start_depth=self.config.max_recursion_depth,
                early_exit=early_exit,
                force_display=force_display,
            )
        )

        return self._pack_result(
            core_resolved + comp_resolved,
            unresolved_non_leaves + comp_unresolved,
            comp_irrelevant,
            cex_found,
        )