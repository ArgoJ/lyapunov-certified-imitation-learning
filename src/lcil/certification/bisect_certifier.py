from __future__ import annotations

import logging
import os
import numpy as np
import torch as th

from torch import nn
from dataclasses import dataclass
from pathlib import Path
from numpy.typing import NDArray

from .recursive_certifier import RecursiveCertifier, RecursiveCertificationResult
from .abcrown_region_certifier import EarlyExitLevel
from .progress import ProgressLevel
from .config import LyapunovCertificationConfig
from ..utils.search_utils import search_and_bisect_value
from ..utils.constants import *

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionCertificationResult:
    """Result container for a full-region certification pass.

    ``global_success`` denotes global certification over the inspected region
    set at ``rho``. It is therefore ``False`` whenever any uncertified
    sublevel-candidate regions remain, even if some subregions were certified
    successfully.
    """

    global_success: bool
    partial_success: bool
    rho: float
    outside_sublevel_regions: NDArray
    uncertified_regions: NDArray
    certified_sublevel_regions: NDArray
    certified_boundary_regions: NDArray

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
            uncertified_regions=self.uncertified_regions,
            certified_sublevel_regions=self.certified_sublevel_regions,
            certified_boundary_regions=self.certified_boundary_regions,
            failed_regions=self.uncertified_regions,
            certified_regions=self.certified_sublevel_regions,
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
            "uncertified_regions",
            "certified_sublevel_regions",
            "certified_boundary_regions",
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
            uncertified_regions=np.asarray(data["uncertified_regions"]),
            certified_sublevel_regions=np.asarray(data["certified_sublevel_regions"]),
            certified_boundary_regions=np.asarray(data["certified_boundary_regions"]),
        )


class BisectCertifier(RecursiveCertifier):
    """Lyapunov certifier using a bisection-based region refinement strategy."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovCertificationConfig,
        device: th.device = th.device("cpu"),
        progress_level: int | ProgressLevel = ProgressLevel.ALL,
        *,
        save_dir: str | os.PathLike | None = None,
    ):
        super().__init__(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            config=config,
            device=device,
            progress_level=progress_level,
        )
        self.save_dir: Path | None = Path(save_dir).resolve() if save_dir is not None else None

    # =========================================
    # CERTIFICATION SEARCH
    # ========================================
    def _create_result(self, rho: float, rec_result: RecursiveCertificationResult) -> RegionCertificationResult:
        bs_bounds = self.region_manager.get_cached_region_bounds(rec_result.resolved)
        inside_mask, boundary_mask, _ = (
            bs_bounds.sublevel_masks(rho + self.config.sublevel_tolerance)
            if bs_bounds is not None and len(rec_result.resolved) > 0
            else (slice(None), slice(0), None)
        )
        return RegionCertificationResult(
            global_success=rec_result.global_success,
            partial_success=rec_result.partial_success,
            rho=float(rho),
            outside_sublevel_regions=self._regions_tensor_to_np(rec_result.irrelevant),
            uncertified_regions=self._regions_tensor_to_np(rec_result.unresolved),
            certified_sublevel_regions=self._regions_tensor_to_np(rec_result.resolved[inside_mask]),
            certified_boundary_regions=self._regions_tensor_to_np(rec_result.resolved[boundary_mask]),
        )

    def is_rho_certified(self, rho: float) -> bool:
        """Check whether all regions satisfy Lyapunov conditions at ``rho``."""
        result = self._certify_recursive_regions(
            rho=rho,
            early_exit=EarlyExitLevel.ON_COUNTEREXAMPLE,
        )

        if self.save_dir is not None:
            cert_result = self._create_result(rho, result)
            subfolder = self.save_dir / CERTIFICATION_EVALUATED_RHOS_DIRNAME
            subfolder.mkdir(parents=True, exist_ok=True)
            cert_result.save(subfolder / f"rho_{rho:.6f}.npz")

        return result.global_success

    def find_max_rho(
        self,
        rho_estimate: float,
    ) -> float:
        """Search for the largest certifiable rho and return it."""
        __logger__.info("Starting Lyapunov certification.")
        
        self.cache_region_bounds()

        with self.progress:
            best_rho = search_and_bisect_value(
                initial_estimate=rho_estimate,
                eval_fn=lambda rho: self.is_rho_certified(rho),
                min_val=self.config.rho_min,
                scaling_factor=self.config.rho_scaling,
                bisection_tol=self.config.bisection_tol,
                max_scale_steps=self.config.max_scale_steps,
                max_bisection_steps=self.config.max_bisection_steps,
                value_name="rho",
                progress=self.progress,
            )

        if best_rho > 0.0:
            __logger__.info("Found best certified rho: %.6f", best_rho)
            
        return best_rho

    # =========================================
    # COLLECT CERTIFICATION DETAILS
    # =========================================
    def _collect_certification_details(self, rho: float) -> RegionCertificationResult:
        """Collect region-wise certification details for a fixed ``rho``."""
        recursive_result = self._certify_recursive_regions(
            rho=rho, force_display=True
        )
        self.details = self._create_result(rho, recursive_result)
        return self.details

    def certify(
        self,
        rho_estimate: float,
        collect_details_on_failed: bool = False,
    ) -> RegionCertificationResult | None:
        """Convenience method to run the full certification and return details."""
        best_rho = self.find_max_rho(rho_estimate)

        # Successful certification
        if best_rho >= self.config.rho_min:
            return self._collect_certification_details(rho=best_rho)

        # Failed certification
        __logger__.warning(
            "No globally certified rho found above rho_min (%.0e).", self.config.rho_min
        )

        if not collect_details_on_failed:
            return None

        # Fallback detail collection
        fallback_rho = self.region_manager.get_best_fallback_rho(
            rho_min=self.config.rho_min,
            sublevel_tolerance=self.config.sublevel_tolerance,
        )
        __logger__.info("Collecting diagnostic details at fallback rho=%.6f.", fallback_rho)
        
        return self._collect_certification_details(rho=fallback_rho)

    def save(
        self,
        save_folder: str | os.PathLike,
    ) -> Path | None:
        """Save certification details and config to disk."""
        save_path = Path(save_folder).resolve()
        save_path.mkdir(parents=True, exist_ok=True)

        config_path = save_path / CERTIFICATION_CONFIG_FILENAME
        self.config.save(config_path)

        details_path = None
        if self.details is None:
            failed_flag_path = save_path / CERTIFICATION_FAILED_FLAG_FILENAME
            failed_flag_path.touch(exist_ok=True)
        else:
            details_path = save_path / CERTIFICATION_DETAILS_FILENAME
            self.details.save(details_path)

        __logger__.info("Saved certification details to %s", save_path)
        return details_path