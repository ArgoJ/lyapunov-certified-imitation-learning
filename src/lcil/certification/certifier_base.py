from __future__ import annotations

import torch as th
import torch.nn as nn
import numpy as np

from contextlib import nullcontext
from typing import Sequence
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod

from .config import LyapunovCertificationConfig
from .models import ClosedLoopLyapunovConditionVerifier
from ..utils.package_logger import get_package_logger, PackageLogger

__logger__ = get_package_logger(__name__)


@dataclass(frozen=True)
class RegionCertificationResult:
    """Result container for a full-region certification pass."""
    success: bool
    counter_examples: np.ndarray
    failed_regions: np.ndarray
    certified_regions: np.ndarray


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
        """_summary_

        Parameters
        ----------
        policy_model : nn.Module
            _description_
        lyap_model : nn.Module
            _description_
        dyn_model : nn.Module
            _description_
        config : LyapunovCertificationConfig
            _description_
        device : th.device, optional
            _description_, by default th.device("cpu")
        """
        self.config = self._resolve_config(config)
        self.device = device

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()

        self.bounds = self._resolve_bounds(config.state_bounds, device)
        
        self.regions = None
        self.verifier = None

    # ==========================================
    # ABSTRACT METHODS
    # ==========================================

    @abstractmethod
    def setup_backend(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def _certify_batched_regions(
            self,
            lbs: th.Tensor,
            ubs: th.Tensor,
            rho: float,
            early_exit: bool = True,
            *args, **kwargs
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        pass

    # ==========================================
    # CORE LOGIC
    # ==========================================

    @staticmethod
    def _resolve_config(config: LyapunovCertificationConfig) -> LyapunovCertificationConfig:
        resolved_config = replace(
            config,
            cert_method=config.cert_method.strip().lower(),
            cert_rho_scaling=max(config.cert_rho_scaling, 1.01),
        )
        if resolved_config.cert_step <= 0:
            raise ValueError("cert_step must be positive.")
        return resolved_config

    @staticmethod
    def _resolve_bounds(state_bounds: Sequence[float], device: th.device) -> th.Tensor:
        bounds = th.as_tensor(state_bounds, dtype=th.float32, device=device)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("state_bounds must be a sequence of shape (2, nx) [lb, ub].")
        return bounds

    @staticmethod
    def _iterative_rho_search(total:int, desc: str, initial_values: tuple[float, float], step_fn: callable) -> tuple[float | None, float]:
        rho_lo, rho_up = initial_values
        with __logger__.tqdm(range(total), desc=desc) as pbar:
            for _ in pbar:
                try:
                    stop, rho_lo, rho_up = step_fn(rho_lo, rho_up)
                    if stop: 
                        break
                finally:
                    lo_display = f"{rho_lo:.4f}" if rho_lo is not None else "None"
                    pbar.set_postfix({"rho_lo": lo_display, "rho_up": f"{rho_up:.4f}"})
        return rho_lo, rho_up

    def _get_suppress_ctx(self):
        if self.config.cert_suppress_native_output:
            lirpa_ctx = PackageLogger.suppress_native_output(suppress_stderr=True)
        else:
            lirpa_ctx = nullcontext()
        return lirpa_ctx

    def _setup_verifier(self) -> ClosedLoopLyapunovConditionVerifier:
        verifier = ClosedLoopLyapunovConditionVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=self.bounds[0],
            ubx=self.bounds[1],
            invariance_weight=self.config.invariance_weight,
        ).to(self.device)
        verifier.eval()
        return verifier

    def _resolve_origin_exclusion(self) -> th.Tensor:
        """Resolve origin exclusion widths per state dimension."""
        # Default: 1% of per-dimension bound radius, capped at 0.1.
        default_exclusion = th.minimum(
            self.bounds.abs().max(dim=0).values * 0.01,
            th.full((self.config.state_dim,), 0.1, dtype=th.float32, device=self.device),
        )

        raw_exclusion = self.config.cert_origin_exclusion
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

    def _certify_regions_recursive(
            self, 
            lbs: th.Tensor, 
            ubs: th.Tensor,
            rho: float,
            collect_details: bool, 
            depth: int, 
            max_depth: int
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """_summary_

        Parameters
        ----------
        lbs : th.Tensor
            _description_
        ubs : th.Tensor
            _description_
        rho : float
            _description_
        collect_details : bool
            _description_
        depth : int
            _description_
        max_depth : int
            _description_

        Returns
        -------
        tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]
            _description_
        """
        if len(lbs) == 0:
            empty = th.empty((0, 2), device=self.device)
            return empty, empty, empty, empty, empty

        is_safe, centers, _ = self._certify_batched_regions(
            lbs, ubs, rho,
            early_exit=not collect_details
        )

        certified_lbs = lbs[is_safe]
        certified_ubs = ubs[is_safe]

        failed_mask = ~is_safe
        failed_lbs = lbs[failed_mask]
        failed_ubs = ubs[failed_mask]
        counter_examples = centers[failed_mask]

        final_failed_lbs = th.empty((0, 2), device=self.device)
        final_failed_ubs = th.empty((0, 2), device=self.device)

        if len(failed_lbs) > 0:
            if not collect_details:
                # Early Exit
                return certified_lbs, certified_ubs, failed_lbs, failed_ubs, counter_examples

            if depth >= max_depth:
                final_failed_lbs = failed_lbs
                final_failed_ubs = failed_ubs
            else:
                # 4 Sub-Regions with midpoint splitting
                mids = (failed_lbs + failed_ubs) / 2.0
                
                # Q1: Lower-Left
                q1_lb, q1_ub = failed_lbs, mids
                # Q2: Lower-Right
                q2_lb = th.stack([mids[:, 0], failed_lbs[:, 1]], dim=1)
                q2_ub = th.stack([failed_ubs[:, 0], mids[:, 1]], dim=1)
                # Q3: Upper-Left
                q3_lb = th.stack([failed_lbs[:, 0], mids[:, 1]], dim=1)
                q3_ub = th.stack([mids[:, 0], failed_ubs[:, 1]], dim=1)
                # Q4: Upper-Right
                q4_lb, q4_ub = mids, failed_ubs

                sub_lbs = th.cat([q1_lb, q2_lb, q3_lb, q4_lb], dim=0)
                sub_ubs = th.cat([q1_ub, q2_ub, q3_ub, q4_ub], dim=0)

                # Rekursiv certification of all sub-boxes
                sub_c_lbs, sub_c_ubs, sub_f_lbs, sub_f_ubs, sub_cex = self._certify_regions_recursive(
                    sub_lbs, sub_ubs, rho, collect_details, depth + 1, max_depth
                )

                certified_lbs = th.cat([certified_lbs, sub_c_lbs], dim=0)
                certified_ubs = th.cat([certified_ubs, sub_c_ubs], dim=0)
                final_failed_lbs = sub_f_lbs
                final_failed_ubs = sub_f_ubs
                counter_examples = th.cat([counter_examples, sub_cex], dim=0)

        return certified_lbs, certified_ubs, final_failed_lbs, final_failed_ubs, counter_examples

    def _certify_regions(
            self, 
            rho: float,
            collect_details: bool = True
    ) -> RegionCertificationResult:
        """_summary_

        Parameters
        ----------
        rho : float
            _description_
        collect_details : bool, optional
            _description_, by default True

        Returns
        -------
        RegionCertificationResult
            _description_
        """
        lbs, ubs = self.regions
        
        c_lbs, c_ubs, f_lbs, f_ubs, cex = self._certify_regions_recursive(
            lbs, ubs, rho, collect_details, depth=0, max_depth=3
        )

        # convert for output -> NumPy arrays for easier downstream use
        c_lbs_np = c_lbs.cpu().numpy() if c_lbs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)
        c_ubs_np = c_ubs.cpu().numpy() if c_ubs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)
        f_lbs_np = f_lbs.cpu().numpy() if f_lbs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)
        f_ubs_np = f_ubs.cpu().numpy() if f_ubs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)

        # Stack into shape (N, 2, state_dim): [:,0,:] = lb, [:,1,:] = ub
        certified_regions_np = np.stack([c_lbs_np, c_ubs_np], axis=1) if c_lbs_np.shape[0] > 0 else np.empty((0, 2, self.config.state_dim), dtype=np.float32)
        failed_regions_np = np.stack([f_lbs_np, f_ubs_np], axis=1) if f_lbs_np.shape[0] > 0 else np.empty((0, 2, self.config.state_dim), dtype=np.float32)
        counter_examples_np = cex.cpu().numpy() if cex.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)

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
        trial = rho_up * self.config.cert_rho_scaling
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
        trial = max(self.config.rho_min, rho_up / self.config.cert_rho_scaling)
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

        if rho_up - rho_lo <= self.config.cert_bisection_tol:
            return True, rho_lo, rho_up
        else:
            return False, rho_lo, rho_up
    
    def build_regions(self) -> tuple[th.Tensor, th.Tensor]:
        """Build 2D grid regions and exclude cells overlapping the origin window."""
        if self.config.state_dim != 2:
            raise ValueError("certification currently supports state_dim == 2.")

        origin_exclusion = self._resolve_origin_exclusion()
        excl_x = origin_exclusion[0].item()
        excl_y = origin_exclusion[1].item()

        lbx, ubx = self.bounds[0], self.bounds[1]
        step = self.config.cert_step

        x_vals = th.arange(lbx[0].item(), ubx[0].item(), step, device=self.device)
        y_vals = th.arange(lbx[1].item(), ubx[1].item(), step, device=self.device)
        grid_y, grid_x = th.meshgrid(y_vals, x_vals, indexing="ij")

        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        overlaps_origin = (grid_x < excl_x) & ((grid_x + step) > -excl_x) & \
                          (grid_y < excl_y) & ((grid_y + step) > -excl_y)

        valid_mask = ~overlaps_origin
        valid_x = grid_x[valid_mask]
        valid_y = grid_y[valid_mask]

        lbs = th.stack([valid_x, valid_y], dim=1)
        ubs = lbs + step
        return lbs, ubs

    def is_rho_certified(self, rho: float) -> bool:
        result = self._certify_regions(rho=rho, collect_details=False) 
        return result.success

    def certify(self, rho_estimate: float) -> tuple[float, RegionCertificationResult]:
        if rho_estimate <= 0:
            raise ValueError("rho_estimate must be positive.")

        __logger__.info("Starting Lyapunov certification with %s method.", self.config.cert_method.upper())

        self.verifier = self._setup_verifier()
        self.regions = self.build_regions()
        self.setup_backend()

        if self.config.rho_min > rho_estimate:
            __logger__.warning(
                "Provided rho_estimate (%.4f) is below rho_min (%.4f). Starting search from rho_min.", 
                rho_estimate, self.config.rho_min
            )
            initial_rho = self.config.rho_min
        else:
            initial_rho = float(rho_estimate)

        initial_ok = self.is_rho_certified(rho=initial_rho)

        # Scale up
        if initial_ok:
            rho_lo, rho_up = self._iterative_rho_search(
                total=self.config.cert_max_scale_steps,
                desc="Scale up",
                initial_values=(initial_rho, initial_rho),
                step_fn=self._scale_rho_up
            )
            
            if rho_lo == rho_up:
                __logger__.warning("Maximum scaling steps reached without finding an upper bound.")
                return rho_lo, self._certify_regions(rho=rho_lo, collect_details=True)

        # Scale down
        else:
            rho_lo, rho_up = self._iterative_rho_search(
                total=self.config.cert_max_scale_steps,
                desc="Scale down",
                initial_values=(None, initial_rho),
                step_fn=self._scale_rho_down
            )

            # Fallback
            if rho_lo is None:
                if not self.is_rho_certified(rho=self.config.rho_min):
                    __logger__.error("Could not even certify rho_min (%.4f).", self.config.rho_min)
                    return self.config.rho_min, self._certify_regions(rho=self.config.rho_min, collect_details=True)
                
                rho_lo = self.config.rho_min
                if rho_up <= rho_lo:
                    rho_up = rho_lo * self.config.cert_rho_scaling

        # Bisect
        rho_lo, rho_up = self._iterative_rho_search(
            total=self.config.cert_max_bisection_steps,
            desc="Bisect rho",
            initial_values=(rho_lo, rho_up),
            step_fn=self._bisect_rho
        )

        details = self._certify_regions(rho=rho_lo, collect_details=True)
        return rho_lo, details
