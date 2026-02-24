from __future__ import annotations

import torch as th
import torch.nn as nn

from typing import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .config import LyapunovCertificationConfig
from .models import ClosedLoopLyapunovConditionVerifier
from ..utils.package_logger import get_package_logger, PackageLogger

__logger__ = get_package_logger(__name__)


@dataclass(frozen=True)
class RegionCertificationResult:
    """Result container for a full-region certification pass."""
    success: bool
    counter_examples: list[th.Tensor]
    failed_regions: list[tuple[list[float], list[float]]]
    certified_regions: list[tuple[list[float], list[float]]]


class LiRPACertifier:
    """_summary_
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
        self.fallback_methods = self._get_fallback_methods(self.config.cert_method)
        self.device = device

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()

        self.bounds = self._resolve_bounds(config.state_bounds, device)
        self.regions = self._build_regions()

        self._lirpa_model = None
        self._verifier_module = None

    @staticmethod
    def _get_lirpa_ctx(suppress_native_output: bool = False):
        """_summary_

        Parameters
        ----------
        suppress_native_output : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        if suppress_native_output:
            lirpa_ctx = PackageLogger.suppress_native_output()
        else:
            lirpa_ctx = nullcontext()
        return lirpa_ctx

    @staticmethod
    def _get_fallback_methods(method : str) -> list[str]:
        """_summary_

        Parameters
        ----------
        method : str
            _description_

        Returns
        -------
        list[str]
            _description_
        """
        fallback_methods = [method.strip().lower()]
        if fallback_methods[0] == "alpha-crown":
            fallback_methods.extend(["crown", "crown-ibp", "ibp"])
        return fallback_methods

    @staticmethod
    def _resolve_config(config: LyapunovCertificationConfig) -> LyapunovCertificationConfig:
        """_summary_

        Parameters
        ----------
        config : LyapunovCertificationConfig
            _description_

        Returns
        -------
        LyapunovCertificationConfig
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        resolved_config = replace(
            config,
            cert_method=config.cert_method.strip().lower(),
            cert_rho_scaling=max(config.cert_rho_scaling, 1.01),
        )
        if resolved_config.cert_step <= 0:
            raise ValueError("cert_step must be positive.")
        if resolved_config.cert_method not in {"alpha-crown", "crown", "crown-ibp", "ibp"}:
            raise ValueError("cert_method must be one of 'alpha-crown', 'crown', 'crown-ibp', or 'ibp'.")
        return resolved_config

    @staticmethod
    def _resolve_bounds(state_bounds: Sequence[float], device: th.device) -> th.Tensor:
        """_summary_

        Parameters
        ----------
        state_bounds : Sequence[float]
            _description_
        device : th.device
            _description_

        Returns
        -------
        th.Tensor
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        bounds = th.as_tensor(state_bounds, dtype=th.float32, device=device)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("state_bounds must be a sequence of shape (2, nx) [lb, ub].")
        return bounds

    def _setup_lirpa_model(self, verifier: nn.Module) -> BoundedModule:
        """_summary_

        Parameters
        ----------
        verifier : nn.Module
            _description_

        Returns
        -------
        BoundedModule
            _description_
        """
        dummy_input = th.zeros(1, self.config.state_dim, device=self.device)
        lirpa_model = BoundedModule(
            verifier,
            dummy_input,
            device=self.device,
            verbose=False,
            bound_opts={'perturb_bound': True},
        )
        return lirpa_model

    def _setup_verifier(self, rho: float) -> ClosedLoopLyapunovConditionVerifier:
        """_summary_

        Parameters
        ----------
        rho : float
            _description_

        Returns
        -------
        ClosedLoopLyapunovConditionVerifier
            _description_
        """
        lbx, ubx = self.bounds[0], self.bounds[1]
        verifier = ClosedLoopLyapunovConditionVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=lbx,
            ubx=ubx,
            kappa=self.config.kappa,
            invariance_weight=self.config.invariance_weight,
            rho=max(self.config.rho_min, rho),
        ).to(self.device)
        verifier.eval()
        return verifier

    def _build_regions(self) -> tuple[th.Tensor, th.Tensor]:
        """_summary_

        Returns
        -------
        tuple[th.Tensor, th.Tensor]
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if self.config.state_dim != 2:
            raise ValueError("certification currently supports state_dim == 2.")

        train_diameter = th.max(th.abs(self.bounds)).item()
        origin_exclusion = self.config.cert_origin_exclusion or min(train_diameter * 0.01, 0.1)

        lbx, ubx = self.bounds[0], self.bounds[1]
        step = self.config.cert_step

        x_vals = th.arange(lbx[0].item(), ubx[0].item(), step, device=self.device)
        y_vals = th.arange(lbx[1].item(), ubx[1].item(), step, device=self.device)
        grid_y, grid_x = th.meshgrid(y_vals, x_vals, indexing="ij")
        
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        mask = ~((th.abs(grid_x) < origin_exclusion) & (th.abs(grid_y) < origin_exclusion))
        
        valid_x = grid_x[mask]
        valid_y = grid_y[mask]

        # Regionen als 2D-Tensoren (N, 2) speichern
        lbs = th.stack([valid_x, valid_y], dim=1)
        ubs = lbs + step
        
        return lbs, ubs


    def _certify_batched_regions(
            self,
            lbs: th.Tensor,
            ubs: th.Tensor,
            max_batch_size: int = 512,
            early_exit: bool = True,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """_summary_

        Parameters
        ----------
        lbs : th.Tensor
            _description_
        ubs : th.Tensor
            _description_
        max_batch_size : int, optional
            _description_, by default 512
        early_exit : bool, optional
            _description_, by default True

        Returns
        -------
        tuple[th.Tensor, th.Tensor, th.Tensor]
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        num_regions = len(lbs)
        if num_regions != len(ubs):
            raise ValueError("lbs and ubs must have the same length.")
        
        # Preallocate output tensors
        is_certified = th.empty(num_regions, dtype=th.bool, device=self.device)
        max_uppers = th.empty(num_regions, dtype=th.float32, device=self.device)
        centers_out = th.empty_like(lbs)

        for idx in range(0, num_regions, max_batch_size):
            end_idx = min(idx + max_batch_size, num_regions)
            
            b_lbs = lbs[idx : end_idx]
            b_ubs = ubs[idx : end_idx]
            b_centers = (b_lbs + b_ubs) / 2.0

            # write centers to preallocated output tensor
            centers_out[idx : end_idx] = b_centers

            ptb = PerturbationLpNorm(norm=float("inf"), x_L=b_lbs, x_U=b_ubs)
            bounded_input = BoundedTensor(b_centers, ptb)
            ub_out = None

            ctx = self._get_lirpa_ctx(self.config.cert_suppress_native_output)
            for candidate_method in self.fallback_methods:
                try:
                    with ctx:
                        if candidate_method == "alpha-crown":
                            _, ub_out = self.lirpa_model.compute_bounds(x=(bounded_input,), method=candidate_method)
                        else:
                            with th.no_grad():
                                _, ub_out = self.lirpa_model.compute_bounds(x=(bounded_input,), method=candidate_method)
                    break
                except Exception as exc:
                    __logger__.warning(f"Method '{candidate_method}' failed on batch: {exc}")

            if ub_out is None:
                # Fallback on failure
                is_certified[idx : end_idx] = False
                max_uppers[idx : end_idx] = float("inf")
            else:
                max_u = ub_out.flatten()
                max_uppers[idx : end_idx] = max_u
                is_certified[idx : end_idx] = max_u <= self.config.condition_tolerance

            # cleanup
            del b_lbs, b_ubs, b_centers, bounded_input, ptb, ub_out
            if self.device.type == "cuda":
                th.cuda.empty_cache()

            if early_exit and not is_certified[idx : end_idx].all():
                return is_certified, centers_out, max_uppers

        return is_certified, centers_out, max_uppers


    def _certify_regions_recursive(
            self, 
            lbs: th.Tensor, 
            ubs: th.Tensor, 
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

        is_safe, centers, _ = self._certify_batched_regions(lbs, ubs, early_exit=not collect_details)

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
                    sub_lbs, sub_ubs, collect_details, depth + 1, max_depth
                )

                certified_lbs = th.cat([certified_lbs, sub_c_lbs], dim=0)
                certified_ubs = th.cat([certified_ubs, sub_c_ubs], dim=0)
                final_failed_lbs = sub_f_lbs
                final_failed_ubs = sub_f_ubs
                counter_examples = th.cat([counter_examples, sub_cex], dim=0)

        return certified_lbs, certified_ubs, final_failed_lbs, final_failed_ubs, counter_examples


    def certify_regions(self, collect_details: bool = True) -> RegionCertificationResult:
        """_summary_

        Parameters
        ----------
        collect_details : bool, optional
            _description_, by default True

        Returns
        -------
        RegionCertificationResult
            _description_
        """
        lbs, ubs = self.regions
        
        c_lbs, c_ubs, f_lbs, f_ubs, cex = self._certify_regions_recursive(
            lbs, ubs, collect_details, depth=0, max_depth=3
        )

        # convert for output
        certified_regions = [
            (lb.tolist(), ub.tolist()) for lb, ub in zip(c_lbs.cpu(), c_ubs.cpu())
        ]
        failed_regions = [
            (lb.tolist(), ub.tolist()) for lb, ub in zip(f_lbs.cpu(), f_ubs.cpu())
        ]
        counter_examples_list = [t for t in cex.cpu()]

        return RegionCertificationResult(
            success=len(failed_regions) == 0,
            counter_examples=counter_examples_list,
            failed_regions=failed_regions,
            certified_regions=certified_regions,
        )


    def is_rho_certified(self, rho: float) -> bool:
        """_summary_

        Parameters
        ----------
        rho : float
            _description_

        Returns
        -------
        bool
            _description_
        """
        self.verifier.set_rho(rho)
        result = self.certify_regions(collect_details=False) 
        return result.success


    def certify(
            self,
            rho_estimate: float,
    ) -> tuple[float, RegionCertificationResult]:
        """_summary_

        Parameters
        ----------
        rho_estimate : float
            _description_

        Returns
        -------
        tuple[float, RegionCertificationResult]
            _description_
        """
        __logger__.info("Starting Lyapunov certification with method: %s", self.config.cert_method)

        self.verifier = self._setup_verifier(rho_estimate)
        self.lirpa_model = self._setup_lirpa_model(self.verifier)

        if self.config.rho_min > rho_estimate:
            __logger__.warning(
                "Provided rho_estimate (%.4f) is below rho_min (%.4f). Starting search from rho_min.", 
                rho_estimate, self.config.rho_min
            )
            initial_rho = self.config.rho_min
        else:
            initial_rho = float(rho_estimate)

        initial_ok = self.is_rho_certified(rho=initial_rho)

        # Initial rho passed, scale up to find an upper bound.
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

            if not found_upper_failure:
                self.verifier.set_rho(rho_lo)
                details = self.certify_regions(collect_details=True)
                return rho_lo, details

        # Initial rho failed, scale down to find a certified rho.
        if not initial_ok:
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

            if rho_lo is None:
                rho_min_ok = self.is_rho_certified(rho=self.config.rho_min)
                if not rho_min_ok:
                    self.verifier.set_rho(self.config.rho_min)
                    details = self.certify_regions(collect_details=True)
                    __logger__.warning(
                        "rho min not certified, try lower than %.2f. Returning rho_min with details.", 
                        self.config.rho_min
                    )
                    return self.config.rho_min, details

                rho_lo = self.config.rho_min
                rho_up = max(rho_up, initial_rho)

        # Bisection between rho_lo and rho_up to find the largest certified rho within tolerance.
        with __logger__.tqdm(range(self.config.cert_max_bisection_steps), desc="Bisection: max rho") as pbar:
            for _ in pbar:
                if rho_up - rho_lo <= self.config.cert_bisection_tol:
                    break

                rho_mid = 0.5 * (rho_lo + rho_up)
                if self.is_rho_certified(rho=rho_mid):
                    rho_lo = rho_mid
                else:
                    rho_up = rho_mid

        self.verifier.set_rho(rho_lo)
        details = self.certify_regions(collect_details=True)
        return rho_lo, details

