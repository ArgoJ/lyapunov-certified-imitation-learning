from __future__ import annotations

import logging
import os
import numpy as np
import torch as th
import torch.nn as nn

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence
from numpy.typing import NDArray
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .abcrown_region_certifier import ABCrownRegionCertifier
from .config import LyapunovCertificationConfig
from .lirpa_lyapunov_bounds import LiRPALyapunovRegionBounds
from .models import LyapunovMultiOutputVerifier
from .region_builder import RegionBuilder
from ..utils.helpers import none_to_float

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionCertificationResult:
    """Result container for a full-region certification pass.

    ``success`` denotes global certification over the inspected region set at
    ``rho``. It is therefore ``False`` whenever any failed regions remain,
    even if some subregions were certified successfully.
    """

    global_success: bool
    partial_success: bool
    rho: float
    outside_sublevel_regions: NDArray
    failed_regions: NDArray
    certified_regions: NDArray

    def save(self, path: str | Path) -> None:
        """Persist certification details to a NumPy ``.npz`` archive."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target_path,
            global_success=np.asarray(self.global_success, dtype=np.bool_),
            partial_success=np.asarray(self.partial_success, dtype=np.bool_),
            rho=np.asarray(self.rho, dtype=np.float64),
            outside_sublevel_regions=self.outside_sublevel_regions,
            failed_regions=self.failed_regions,
            certified_regions=self.certified_regions,
        )

    @classmethod
    def load(cls, path: str | Path) -> RegionCertificationResult:
        """Load certification details from a NumPy ``.npz`` archive."""
        data = np.load(Path(path), allow_pickle=False)
        required_keys = {
            "global_success",
            "partial_success",
            "rho",
            "outside_sublevel_regions",
            "failed_regions",
            "certified_regions",
        }
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Missing keys in certification result file: {missing}")

        return cls(
            global_success=bool(np.asarray(data["global_success"]).item()),
            partial_success=bool(np.asarray(data["partial_success"]).item()),
            rho=float(np.asarray(data["rho"]).item()),
            outside_sublevel_regions=np.asarray(data["outside_sublevel_regions"]),
            failed_regions=np.asarray(data["failed_regions"]),
            certified_regions=np.asarray(data["certified_regions"]),
        )


@dataclass(frozen=True)
class RecursiveCertificationResult:
    """Internal tensor result for recursive region certification."""

    resolved: th.Tensor
    unresolved: th.Tensor
    irrelevant: th.Tensor

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
        )

    def with_failed(self, regions: th.Tensor) -> RecursiveCertificationResult:
        return replace(self, unresolved=regions)

    def with_cert_and_failed(
        self,
        bs: th.Tensor,
        is_safe: th.Tensor,
    ) -> RecursiveCertificationResult:
        failed_mask = ~is_safe
        certified_bs = bs[is_safe]
        failed_bs = bs[failed_mask]
        return replace(self, resolved=certified_bs, unresolved=failed_bs)

    def with_irrelevant(self, regions: th.Tensor) -> RecursiveCertificationResult:
        return replace(self, irrelevant=regions)

    def __add__(self, other: RecursiveCertificationResult) -> RecursiveCertificationResult:
        return RecursiveCertificationResult(
            resolved=th.cat([self.resolved, other.resolved], dim=0),
            unresolved=th.cat([self.unresolved, other.unresolved], dim=0),
            irrelevant=th.cat([self.irrelevant, other.irrelevant], dim=0),
        )


class BisectCertifier:
    """Lyapunov certifier using a bisection-based region refinement strategy."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovCertificationConfig,
        device: th.device = th.device("cpu"),
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
        """
        self.config = config
        self.device = device

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()

        self.bounds = self._resolve_bounds(config.cert_bounds, device)

        self.regions = None
        self.region_builder: RegionBuilder | None = None
        self.bounder: LiRPALyapunovRegionBounds | None = None
        self.certifier: ABCrownRegionCertifier | None = None
        self.details = None


    # ==========================================
    # COMMON UTILITIES
    # ==========================================
    def _build_region_builder(self) -> RegionBuilder:
        """Construct a region builder for the current certification bounds."""
        return RegionBuilder(
            bounds=self.bounds,
            bins_per_dim=self.config.bins_per_dim,
            center_refinement_factor=self.config.center_refinement_factor,
            origin_exclusion=self.config.origin_exclusion,
            device=self.device,
        )

    def _get_region_builder(self) -> RegionBuilder:
        """Return the cached region builder used for region refinement."""
        if self.region_builder is None:
            self.region_builder = self._build_region_builder()
        return self.region_builder

    def _build_region_certifier(self) -> ABCrownRegionCertifier:
        """Construct the shared per-region ABCrown certifier."""
        return ABCrownRegionCertifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            config=self.config,
            device=self.device,
        )

    def _get_region_certifier(self) -> ABCrownRegionCertifier:
        """Return the cached ABCrown region certifier."""
        if self.certifier is None:
            self.certifier = self._build_region_certifier()
        return self.certifier

    def _build_safe_output_constraint(self, y, rho: float):
        """Build the shared safe predicate for the multi-output verifier."""
        return self._get_region_certifier()._build_safe_output_constraint(y=y, rho=rho)

    def _build_region_bounder(self) -> LiRPALyapunovRegionBounds:
        """Construct a LiRPA helper for classifying regions against ``V(x) <= rho``."""
        return LiRPALyapunovRegionBounds(
            lyap_model=self.lyap_model,
            state_dim=self.config.state_dim,
            batch_size=self.config.batch_size,
            default_bound_method=self.config.cert_method,
            device=self.device,
        )

    def _get_region_bounder(self) -> LiRPALyapunovRegionBounds:
        """Return the cached LiRPA region bounder."""
        if self.bounder is None:
            self.bounder = self._build_region_bounder()
        return self.bounder

    def _partition_regions_by_sublevel(
        self,
        bs: th.Tensor,
        rho: float,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Split regions into ``(relevant, irrelevant)`` using LiRPA bounds on ``V``.

        A region is irrelevant only when LiRPA proves ``V(x) > rho + tol`` over
        the entire box. Boundary-touching boxes remain relevant candidates and
        must still be discharged by the verifier.
        """
        if len(bs) == 0:
            return bs, bs[:0]

        threshold = float(rho + self.config.sublevel_tolerance)
        region_bounds = self._get_region_bounder().compute_bounds_for_regions(
            bs,
            method=self.config.cert_method,
        )
        outside_mask = region_bounds.outside_mask(threshold)
        relevant_bs = bs[~outside_mask]
        irrelevant_bs = bs[outside_mask]

        if len(irrelevant_bs) == 0:
            return bs, bs[:0]

        __logger__.info(
            "Classified %d / %d regions as irrelevant with V(x) > %.6f everywhere.",
            len(irrelevant_bs),
            len(bs),
            threshold,
        )
        return relevant_bs, irrelevant_bs


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

    @staticmethod
    def _pack_regions(lbs: th.Tensor, ubs: th.Tensor) -> th.Tensor:
        """Pack lower/upper bounds into ``(N, 2, state_dim)`` format."""
        return th.stack([lbs, ubs], dim=1)

    @staticmethod
    def _unpack_regions(bs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Unpack ``(N, 2, state_dim)`` regions into lower/upper bounds."""
        return bs[:, 0], bs[:, 1]

    def _split_failed_regions_on_certification_frontier(
        self,
        failed_bs: th.Tensor,
        resolved_bs: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Split only unresolved boxes adjacent to already resolved boxes.

        Returns ``(retry_regions, terminal_regions)`` where only
        ``retry_regions`` remain on the certification frontier and should be
        certified again.
        """
        return self._get_region_builder().split_regions_adjacent_to_reference(
            failed_bs,
            resolved_bs,
        )


    # ========================================
    # CERTIFICATION
    # ========================================
    def _process_regions(
            self, 
            bs: th.Tensor,
            rho: float,
            early_exit: bool,
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

        empty_result = RecursiveCertificationResult.empty(
            state_dim=self.config.state_dim,
            device=self.device,
        )
        if len(bs) == 0:
            return empty_result

        bs, irrelevant_bs = self._partition_regions_by_sublevel(bs, rho)

        result = empty_result.with_irrelevant(irrelevant_bs)
        if len(bs) == 0:
            return result

        is_safe = self._get_region_certifier().certify_regions(
            regions=bs,
            rho=rho,
            early_exit=early_exit,
            show_progress=False,
        )
        return result.with_cert_and_failed(bs, is_safe)

    def _certify_recursive_regions(
        self,
        rho: float,
        *,
        show_progress: bool,
    ) -> RecursiveCertificationResult:
        """Run recursive certification for a fixed ``rho`` over all regions."""
        if self.regions is None:
            raise RuntimeError("Certification regions are not initialized.")

        recursive_result = RecursiveCertificationResult.empty(
            state_dim=self.config.state_dim,
            device=self.device,
        )
        empty_result = RecursiveCertificationResult.empty(
            state_dim=self.config.state_dim,
            device=self.device,
        )

        pending_bs = self.regions
        max_depth = self.config.max_recursion_depth

        if show_progress:
            progress = Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("failed: {task.fields[failed]:.0f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            progress.__enter__()
            task = progress.add_task(
                "Certify and split",
                total=float(max_depth + 1),
                failed=float(len(pending_bs)),
            )
        else:
            progress = None
            task = None

        try:
            for depth in range(max_depth + 1):
                if progress is not None and task is not None:
                    progress.update(task, failed=float(len(pending_bs)))

                if len(pending_bs) == 0:
                    if progress is not None and task is not None:
                        progress.update(task, completed=max_depth + 1)
                    break

                step_result = self._process_regions(
                    pending_bs,
                    rho,
                    early_exit=False,
                )

                if depth >= max_depth:
                    recursive_result = recursive_result + step_result
                    if progress is not None and task is not None:
                        progress.update(task, completed=max_depth + 1)
                    break

                recursive_result = recursive_result + step_result.with_failed(step_result.unresolved[:0])

                if len(step_result.unresolved) == 0:
                    if progress is not None and task is not None:
                        progress.update(task, completed=max_depth + 1)
                    break

                pending_bs, terminal_failed_bs = self._split_failed_regions_on_certification_frontier(
                    step_result.unresolved,
                    recursive_result.resolved,
                )
                if len(terminal_failed_bs) > 0:
                    recursive_result = recursive_result + empty_result.with_failed(terminal_failed_bs)

                if len(pending_bs) == 0:
                    if progress is not None and task is not None:
                        progress.update(task, completed=max_depth + 1)
                    break

                if progress is not None and task is not None:
                    progress.update(task, advance=1)
                    progress.refresh()
        finally:
            if progress is not None:
                progress.__exit__(None, None, None)

        return recursive_result


    # =========================================
    # RHO SEARCH AND BISECTION
    # ========================================
    @staticmethod
    def _iterative_rho_search(
        total: int,
        desc: str,
        initial_values: tuple[float, float],
        step_fn: callable,
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
            MofNCompleteColumn(),
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
                        advance=1,
                        rho_lo=none_to_float(rho_lo),
                        rho_up=none_to_float(rho_up),
                    )
                    progress.refresh()
        return rho_lo, rho_up

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
        return True, rho_lo, trial

    def _scale_rho_down(self, rho_lo: float | None, rho_up: float) -> tuple[bool, float | None, float]:
        """Scales down ``rho_up`` until the new trial is either certified 
        or it is below rho_min, in which case we stop and return the last certified rho as lower bound.

        Parameters
        ----------
        rho_lo : float | None
            Lower bound of rho, updated if trial is certified or the trial is lower than ``rho_lo``.
        rho_up : float
            Upper bound of rho, updated if trial is not certified.

        Returns
        -------
        tuple[bool, float | None, float]
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
        return False, rho_lo, rho_up

    def is_rho_certified(self, rho: float) -> bool:
        """Check whether all regions satisfy Lyapunov conditions at ``rho``."""
        result = self._certify_recursive_regions(
            rho=rho,
            show_progress=False,
        )
        return result.global_success

    def find_max_rho(self, rho_estimate: float) -> float:
        """Search for the largest certifiable rho and return it."""
        if rho_estimate <= 0:
            raise ValueError("rho_estimate must be positive.")

        __logger__.info("Starting Lyapunov certification.")

        self.regions = self._get_region_builder().build_regions()

        if self.config.rho_min > rho_estimate:
            __logger__.warning(
                "Provided rho_estimate (%.4f) is below rho_min (%.4f). Starting search from rho_min.",
                rho_estimate,
                self.config.rho_min,
            )
            initial_rho = self.config.rho_min
        else:
            initial_rho = float(rho_estimate)

        initial_ok = self.is_rho_certified(rho=initial_rho)
        has_certified_lower_bound = True

        # Scale up
        if initial_ok:
            rho_lo, rho_up = self._iterative_rho_search(
                total=self.config.max_scale_steps,
                desc="Scale up",
                initial_values=(initial_rho, initial_rho),
                step_fn=self._scale_rho_up,
            )
            if rho_lo == rho_up:
                __logger__.warning(
                    "Maximum scaling steps (%d) reached without finding an upper bound.",
                    self.config.max_scale_steps,
                )

        # Scale down
        else:
            rho_lo, rho_up = self._iterative_rho_search(
                total=self.config.max_scale_steps,
                desc="Scale down",
                initial_values=(None, initial_rho),
                step_fn=self._scale_rho_down,
            )

            # Fallback
            if rho_lo is None:
                has_certified_lower_bound = False
                rho_lo = self.config.rho_min
                rho_up = self.config.rho_min
                __logger__.error(
                    "Could not find any certified rho >= rho_min (%.0e).",
                    self.config.rho_min,
                )

        # Bisect
        if has_certified_lower_bound and rho_up - rho_lo >= self.config.bisection_tol:
            rho_lo, rho_up = self._iterative_rho_search(
                total=self.config.max_bisection_steps,
                desc="Bisect rho",
                initial_values=(rho_lo, rho_up),
                step_fn=self._bisect_rho,
            )

        if not has_certified_lower_bound:
            __logger__.warning("No certified rho found.")
            return 0.0

        __logger__.info("Found best certified rho: %.6f", rho_lo)
        return rho_lo


    # =========================================
    # COLLECT CERTIFICATION DETAILS
    # =========================================
    def _collect_certification_details(self, rho: float) -> RegionCertificationResult:
        """Collect region-wise certification details for a fixed ``rho``.

        Parameters
        ----------
        rho : float
            Lyapunov level-set value to test.

        Returns
        -------
        RegionCertificationResult
            Aggregated certification result with region partitions.
        """
        recursive_result = self._certify_recursive_regions(rho, show_progress=True)

        certified_regions_np = self._regions_tensor_to_np(recursive_result.resolved)
        failed_regions_np = self._regions_tensor_to_np(recursive_result.unresolved)
        outside_sublevel_regions_np = self._regions_tensor_to_np(recursive_result.irrelevant)

        if recursive_result.vacuous:
            __logger__.warning(
                "Certification at rho=%.6f is completely filtered: all regions are outside V(x) <= rho.",
                float(rho),
            )

        self.details = RegionCertificationResult(
            global_success=recursive_result.global_success,
            partial_success=recursive_result.partial_success,
            rho=rho,
            outside_sublevel_regions=outside_sublevel_regions_np,
            failed_regions=failed_regions_np,
            certified_regions=certified_regions_np,
        )
        __logger__.info(
            "Certification detail pass at rho=%.6f: success=%s, certified=%d, failed=%d, outside_sublevel=%d.",
            float(self.details.rho),
            self.details.global_success,
            len(self.details.certified_regions),
            len(self.details.failed_regions),
            len(self.details.outside_sublevel_regions),
        )
        return self.details

    def certify(self, rho_estimate: float) -> RegionCertificationResult:
        """Convenience method to run the full certification and return details."""
        best_rho = self.find_max_rho(rho_estimate)

        if best_rho >= self.config.rho_min:
            inspection_rho = best_rho
        else:
            inspection_rho = max(float(rho_estimate), self.config.rho_min)
            __logger__.warning(
                "No globally certified rho found above rho_min (%.0e). Collecting fallback inspection details at rho=%.6f.",
                self.config.rho_min,
                inspection_rho,
            )

        self.details = self._collect_certification_details(rho=inspection_rho)
        return self.details

    def save(
        self,
        save_folder: str | os.PathLike,
    ) -> Path:
        """Save certification details and config to disk."""
        save_path = Path(save_folder).resolve()
        save_path.mkdir(parents=True, exist_ok=True)

        details_path = save_path / "certification_details.npz"
        if self.details is not None:
            self.details.save(details_path)

        config_path = save_path / "certification_config.json"
        self.config.save(config_path)

        __logger__.info("Saved certification details to %s", save_path)
        return details_path