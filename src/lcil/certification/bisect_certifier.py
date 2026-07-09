from __future__ import annotations

import logging
import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from numpy.typing import NDArray

from .recursive_certifier import RecursiveCertifier
from .abcrown_region_certifier import EarlyExitLevel
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

    # =========================================
    # CERTIFICATION SEARCH
    # ========================================
    def is_rho_certified(self, rho: float) -> bool:
        """Check whether all regions satisfy Lyapunov conditions at ``rho``."""
        result = self._certify_recursive_regions(
            rho=rho,
            early_exit=EarlyExitLevel.ON_COUNTEREXAMPLE,
        )
        return result.global_success

    def find_max_rho(self, rho_estimate: float) -> float:
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
        recursive_result = self._certify_recursive_regions(
            rho=rho, force_display=True)

        bs_bounds = self.region_manager.get_cached_region_bounds(recursive_result.resolved)
        inside_mask, boundary_mask, _ = bs_bounds.sublevel_masks(rho + self.config.sublevel_tolerance)
        
        certified_sublevel_regions_np = self._regions_tensor_to_np(recursive_result.resolved[inside_mask])
        certified_boundary_regions_np = self._regions_tensor_to_np(recursive_result.resolved[boundary_mask])
        uncertified_regions_np = self._regions_tensor_to_np(recursive_result.unresolved)
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
            uncertified_regions=uncertified_regions_np,
            certified_sublevel_regions=certified_sublevel_regions_np,
            certified_boundary_regions=certified_boundary_regions_np,
        )
        __logger__.debug(
            "Certification detail pass at rho=%.6f: success=%s, certified_sublevel=%d, uncertified=%d, outside_sublevel=%d.",
            float(self.details.rho),
            self.details.global_success,
            len(self.details.certified_sublevel_regions),
            len(self.details.uncertified_regions),
            len(self.details.outside_sublevel_regions),
        )
        return self.details

    def certify(
        self, rho_estimate: float, collect_details_on_failed: bool = False
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
            sublevel_tolerance=self.config.sublevel_tolerance
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