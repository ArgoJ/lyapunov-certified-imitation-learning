from __future__ import annotations
import logging
import torch as th
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .certifier_base import BaseCertifier

__logger__ = logging.getLogger(__name__)


class LiRPACertifier(BaseCertifier):
    """Lyapunov certifier using auto_LiRPA bound propagation."""

    def __init__(
        self,
        policy_model,
        lyap_model,
        dyn_model,
        config,
        device: th.device = th.device("cpu"),
    ):
        super().__init__(policy_model, lyap_model, dyn_model, config, device)
        self.lirpa_model = None
        self.fallback_methods = None
        self._alpha_crown_disabled = False

    @staticmethod
    def _get_fallback_methods(method : str) -> list[str]:
        """Return backend fallback methods ordered from strongest to weakest."""
        fallback_methods = [method.strip().lower()]
        if fallback_methods[0] == "alpha-crown":
            fallback_methods.extend(["crown", "crown-ibp", "ibp"])
        return fallback_methods

    def setup_backend(self) -> None:
        """Initialize the bounded LiRPA model and fallback method list."""
        self.lirpa_model = self._get_bounded_module()
        self.fallback_methods = self._get_fallback_methods(self.config.cert_method)
        self._alpha_crown_disabled = False

    @staticmethod
    def _is_alpha_shape_mismatch_error(exc: Exception) -> bool:
        """Return True for known alpha-crown internal shape mismatch failures."""
        msg = str(exc).lower()
        return (
            "size of tensor a" in msg
            or "must match the size of tensor b" in msg
            or "einsum(): subscript" in msg
        )

    def _disable_alpha_crown_for_run(self, reason: Exception) -> None:
        """Disable alpha-crown for this certify run and keep safe fallbacks."""
        if self._alpha_crown_disabled:
            return
        self._alpha_crown_disabled = True
        if self.fallback_methods is not None:
            self.fallback_methods = [m for m in self.fallback_methods if m != "alpha-crown"]
        __logger__.warning(
            "Disabling alpha-crown for this run due to backend shape mismatch: %s",
            reason,
        )

    def _get_bounded_module(self) -> BoundedModule:
        """No additional setup needed for auto_LiRPA."""
        dummy_x = th.zeros(1, self.config.state_dim, device=self.device)
        dummy_rho = th.zeros(1, 1, device=self.device)
        dummy_kappa = th.zeros(1, 1, device=self.device)

        lirpa_bound_opts = {
            'perturb_bound': True,
            'optimize_bound_args': {
                # Keep alpha initialization local per activation and avoid stale sharing.
                # The old keys 'apply_alpha_for_intermediate_bounds' and
                # 'shared_alphas' are not used by auto_LiRPA 0.7.x.
                'use_shared_alpha': False,
                'fix_interm_bounds': True,
            }
        }

        return BoundedModule(
            self.verifier,
            (dummy_x, dummy_rho, dummy_kappa),
            device=self.device,
            verbose=False,
            bound_opts=lirpa_bound_opts,
        )

    def _certify_batched_regions(
            self,
            bs: th.Tensor,
            rho: float,
            early_exit: bool = True,
    ) -> th.Tensor:
        """Certify a packed batch of regions with LiRPA bounds.

        Parameters
        ----------
        bs : th.Tensor
            Packed region bounds with shape ``(n, 2, state_dim)``.
        rho : float
            Lyapunov level-set value to certify.
        early_exit : bool, optional
            Stop after first failed region if ``True``.

        Returns
        -------
        th.Tensor
            Boolean tensor indicating safe regions.
        """
        if self.lirpa_model is None or self.fallback_methods is None:
            raise RuntimeError("LiRPACertifier backend is not properly initialized.")

        num_regions = len(bs)
        if num_regions == 0:
            return th.empty((0,), dtype=th.bool, device=self.device)
        effective_batch_size = min(self.config.batch_size, num_regions)
        
        # Preallocate output tensors
        is_certified = th.zeros(num_regions, dtype=th.bool, device=self.device)
        lbs, ubs = self._unpack_regions(bs)
        centers_out = (lbs + ubs) / 2.0

        for idx in range(0, num_regions, effective_batch_size):
            end_idx = min(idx + effective_batch_size, num_regions)

            b_lbs = lbs[idx : end_idx]
            b_ubs = ubs[idx : end_idx]
            b_centers = centers_out[idx : end_idx]

            batch_len = b_centers.shape[0]
            pad_len = self.config.batch_size - batch_len

            if pad_len > 0:
                # Fülle die Tensoren bis zur geforderten batch_size auf
                b_lbs = th.cat([b_lbs, b_lbs[-1:].expand(pad_len, -1)], dim=0)
                b_ubs = th.cat([b_ubs, b_ubs[-1:].expand(pad_len, -1)], dim=0)
                b_centers = th.cat([b_centers, b_centers[-1:].expand(pad_len, -1)], dim=0)

            ptb = PerturbationLpNorm(norm=float("inf"), x_L=b_lbs, x_U=b_ubs)
            bounded_input = BoundedTensor(b_centers, ptb)

            b_rho = th.full((self.config.batch_size, 1), rho, dtype=th.float32, device=self.device)
            b_kappa = th.full((self.config.batch_size, 1), self.config.kappa, dtype=th.float32, device=self.device)

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
                    if (
                        candidate_method == "alpha-crown"
                        and self._is_alpha_shape_mismatch_error(exc)
                    ):
                        self._disable_alpha_crown_for_run(exc)
                    self.lirpa_model = self._get_bounded_module()
                    __logger__.warning(f"Method '{candidate_method}' failed on batch: {exc}")
                    # raise exc

            if ub_out is None:
                is_certified[idx : end_idx] = False
            else:
                max_u = ub_out.flatten()[:batch_len]
                is_certified[idx : end_idx] = max_u <= self.config.condition_tolerance

            # cleanup
            del b_lbs, b_ubs, b_centers, bounded_input, ptb, ub_out

            # Early exit check
            if early_exit and not is_certified[idx : end_idx].all():
                return is_certified

        return is_certified