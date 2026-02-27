from __future__ import annotations

import torch as th
import torch.nn as nn
import numpy as np

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .certifier_base import BaseCertifier, RegionCertificationResult
from ..utils.package_logger import get_package_logger

__logger__ = get_package_logger(__name__)


class LiRPACertifier(BaseCertifier):
    """_summary_
    """

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

    def build_regions(self) -> tuple[th.Tensor, th.Tensor]:
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
        
        mask = ~overlaps_origin
        
        valid_x = grid_x[mask]
        valid_y = grid_y[mask]

        lbs = th.stack([valid_x, valid_y], dim=1)
        ubs = lbs + step
        
        return lbs, ubs

    def setup_backend(self) -> None:
        """_summary_
        """
        dummy_x = th.zeros(1, self.config.state_dim, device=self.device)
        dummy_rho = th.zeros(1, 1, device=self.device)
        dummy_kappa = th.zeros(1, 1, device=self.device)

        self.lirpa_model = BoundedModule(
            self.verifier,
            (dummy_x, dummy_rho, dummy_kappa),
            device=self.device,
            verbose=False,
            bound_opts={'perturb_bound': True},
        )
        self.fallback_methods = self._get_fallback_methods(self.config.cert_method)

    def _certify_batched_regions(
            self,
            lbs: th.Tensor,
            ubs: th.Tensor,
            rho: float,
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
        rho : float
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

            batch_len = b_centers.shape[0]
            b_rho = th.full((batch_len, 1), rho, dtype=th.float32, device=self.device)
            b_kappa = th.full((batch_len, 1), self.config.kappa, dtype=th.float32, device=self.device)

            ub_out = None
            ctx = self._get_suppress_ctx(self.config.cert_suppress_native_output)
            for candidate_method in self.fallback_methods:
                try:
                    with ctx:
                        if candidate_method == "alpha-crown":
                            _, ub_out = self.lirpa_model.compute_bounds(
                                x=(bounded_input, b_rho, b_kappa), 
                                method=candidate_method
                            )
                        else:
                            with th.no_grad():
                                _, ub_out = self.lirpa_model.compute_bounds(
                                    x=(bounded_input, b_rho, b_kappa),
                                    method=candidate_method
                                )
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
            # if self.device.type == "cuda":
            #     th.cuda.empty_cache()

            if early_exit and not is_certified[idx : end_idx].all():
                return is_certified, centers_out, max_uppers

        return is_certified, centers_out, max_uppers

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

    def certify_regions(
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
