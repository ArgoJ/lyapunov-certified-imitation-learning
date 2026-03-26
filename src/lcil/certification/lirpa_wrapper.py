from __future__ import annotations

import torch as th

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from pkg_logger import get_package_logger

from .certifier_base import BaseCertifier

__logger__ = get_package_logger(__name__)


class LiRPACertifier(BaseCertifier):
    """_summary_
    """
    def __init__(self, policy_model, lyap_model, dyn_model, config, device = ...):
        super().__init__(policy_model, lyap_model, dyn_model, config, device)
        self.lirpa_model = None
        self.fallback_methods = None

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
            early_exit: bool = True,
    ) -> tuple[th.Tensor, th.Tensor]:
        """_summary_

        Parameters
        ----------
        lbs : th.Tensor
            _description_
        ubs : th.Tensor
            _description_
        rho : float
            _description_
        early_exit : bool, optional
            _description_, by default True

        Returns
        -------
        tuple[th.Tensor, th.Tensor]
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if self.lirpa_model is None or self.fallback_methods is None:
            raise RuntimeError("LiRPACertifier backend is not properly initialized.")

        num_regions = len(lbs)
        if num_regions != len(ubs):
            raise ValueError("lbs and ubs must have the same length.")
        
        # Preallocate output tensors
        is_certified = th.zeros(num_regions, dtype=th.bool, device=self.device)
        centers_out = (lbs + ubs) / 2.0

        for idx in range(0, num_regions, self.config.batch_size):
            end_idx = min(idx + self.config.batch_size, num_regions)

            b_lbs = lbs[idx : end_idx]
            b_ubs = ubs[idx : end_idx]
            b_centers = centers_out[idx : end_idx]

            ptb = PerturbationLpNorm(norm=float("inf"), x_L=b_lbs, x_U=b_ubs)
            bounded_input = BoundedTensor(b_centers, ptb)

            batch_len = b_centers.shape[0]
            b_rho = th.full((batch_len, 1), rho, dtype=th.float32, device=self.device)
            b_kappa = th.full((batch_len, 1), self.config.kappa, dtype=th.float32, device=self.device)

            ub_out = None
            for candidate_method in self.fallback_methods:
                try:
                    with self._get_suppress_ctx():
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
                is_certified[idx : end_idx] = False
            else:
                max_u = ub_out.flatten()
                is_certified[idx : end_idx] = max_u <= self.config.condition_tolerance

            # cleanup
            del b_lbs, b_ubs, b_centers, bounded_input, ptb, ub_out
            # if self.device.type == "cuda":
            #     th.cuda.empty_cache()

            # Early exit check
            if early_exit and not is_certified[idx : end_idx].all():
                return is_certified, centers_out

        return is_certified, centers_out