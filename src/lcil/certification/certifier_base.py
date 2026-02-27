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
    # ABSTRAKTE METHODS
    # ==========================================

    @abstractmethod
    def build_regions(self) -> tuple[th.Tensor, th.Tensor]:
        pass

    @abstractmethod
    def setup_backend(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def certify_regions(self, rho: float, collect_details: bool = True) -> RegionCertificationResult:
        pass

    # ==========================================
    # CORE LOGIK
    # ==========================================

    @staticmethod
    def _get_suppress_ctx(suppress_native_output: bool = False):
        if suppress_native_output:
            lirpa_ctx = PackageLogger.suppress_native_output(suppress_stderr=True)
        else:
            lirpa_ctx = nullcontext()
        return lirpa_ctx

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

    def is_rho_certified(self, rho: float) -> bool:
        result = self.certify_regions(rho=rho, collect_details=False) 
        return result.success

    def certify(self, rho_estimate: float) -> tuple[float, RegionCertificationResult]:
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

        if initial_ok:
            rho_lo = initial_rho
            rho_up = initial_rho
            found_upper_failure = False
            
            with __logger__.tqdm(range(self.config.cert_max_scale_steps), desc="Scale up: upper rho") as pbar:
                for _ in pbar:
                    trial = rho_up * self.config.cert_rho_scaling
                    if self.is_rho_certified(rho=trial):
                        rho_lo = trial
                        rho_up = trial
                    else:
                        rho_up = trial
                        found_upper_failure = True
                        break
                    pbar.set_postfix({"rho_lo": rho_lo, "rho_up": rho_up})

            if not found_upper_failure:
                return rho_lo, self.certify_regions(rho=rho_lo, collect_details=True)

        else:
            rho_up = initial_rho
            rho_lo: float | None = None
            trial = initial_rho
            
            with __logger__.tqdm(range(self.config.cert_max_scale_steps), desc="Scale down: lower rho") as pbar:
                for _ in pbar:
                    trial = max(self.config.rho_min, trial / self.config.cert_rho_scaling)
                    if self.is_rho_certified(rho=trial):
                        rho_lo = trial
                        break
                    rho_up = trial
                    if trial <= self.config.rho_min:
                        break
                    pbar.set_postfix({"rho_lo": rho_lo, "rho_up": rho_up})

            if rho_lo is None:
                if not self.is_rho_certified(rho=self.config.rho_min):
                    return self.config.rho_min, self.certify_regions(rho=self.config.rho_min, collect_details=True)
                rho_lo = self.config.rho_min
                if rho_up <= rho_lo:
                    rho_up = rho_lo * self.config.cert_rho_scaling

        with __logger__.tqdm(range(self.config.cert_max_bisection_steps), desc="Bisection: max rho") as pbar:
            for _ in pbar:
                if rho_up - rho_lo <= self.config.cert_bisection_tol:
                    break

                rho_mid = 0.5 * (rho_lo + rho_up)
                if self.is_rho_certified(rho=rho_mid):
                    rho_lo = rho_mid
                else:
                    rho_up = rho_mid

                pbar.set_postfix({"rho_lo": rho_lo, "rho_up": rho_up})

        details = self.certify_regions(rho=rho_lo, collect_details=True)
        return rho_lo, details