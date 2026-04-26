from __future__ import annotations

from contextlib import nullcontext
import logging

import torch as th
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from pkg_logger import suppress_native_output

from .config import LyapunovCertificationConfig
from .models import LyapunovMultiOutputVerifier

__logger__ = logging.getLogger(__name__)


class ConservativeLiRPAVerifier:
    """Cheap conservative LiRPA pre-verifier for region-level safe checks.

    This helper intentionally exposes only a narrow box-wise API for the
    ABCrown precheck path. It does not implement rho search, recursive
    splitting, or result aggregation.
    """

    _DEFAULT_METHOD = "crown-ibp"

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        bounds: th.Tensor,
        config: LyapunovCertificationConfig,
        device: th.device = th.device("cpu"),
        method: str = _DEFAULT_METHOD,
    ) -> None:
        self.device = device
        self.config = config
        self.policy_model = policy_model
        self.lyap_model = lyap_model
        self.dyn_model = dyn_model
        self.bounds = th.as_tensor(bounds, dtype=th.float32, device=device)
        self.fallback_methods = self._get_fallback_methods(method)
        self._alpha_crown_disabled = False

        self.verifier = self._build_verifier()
        self.lirpa_model = self._build_bounded_module()

    @staticmethod
    def _get_fallback_methods(method: str) -> list[str]:
        """Return candidate LiRPA methods for conservative pre-verification."""
        fallback_methods = [method.strip().lower()]
        if fallback_methods[0] == "alpha-crown":
            fallback_methods.extend(["crown", "crown-ibp", "ibp"])
        elif fallback_methods[0] == "crown-ibp":
            fallback_methods.append("ibp")
        return fallback_methods

    def _evaluate_certified_mask(
        self,
        lb_out: th.Tensor,
        ub_out: th.Tensor,
        batch_len: int,
    ) -> th.Tensor:
        """Evaluate componentwise safe conditions from LiRPA lower/upper bounds."""
        lb_slice = lb_out[:batch_len]
        ub_slice = ub_out[:batch_len]

        positive_ok = lb_slice[:, 1] >= (-self.config.condition_tolerance)
        decrease_ok = lb_slice[:, 0] >= (-self.config.condition_tolerance)

        next_lower_ok = lb_slice[:, 2:] >= (
            self.bounds[0].unsqueeze(0) - self.config.condition_tolerance
        )
        next_upper_ok = ub_slice[:, 2:] <= (
            self.bounds[1].unsqueeze(0) + self.config.condition_tolerance
        )
        safe_x_next = next_lower_ok.all(dim=1) & next_upper_ok.all(dim=1)

        return positive_ok & decrease_ok & safe_x_next

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
        """Disable alpha-crown for this verifier instance and keep safe fallbacks."""
        if self._alpha_crown_disabled:
            return
        self._alpha_crown_disabled = True
        self.fallback_methods = [m for m in self.fallback_methods if m != "alpha-crown"]
        __logger__.warning(
            "Disabling alpha-crown for conservative precheck due to backend shape mismatch: %s",
            reason,
        )

    def _build_verifier(self) -> LyapunovMultiOutputVerifier:
        """Construct a multi-output verifier for componentwise safe bounds.

        The outside-sublevel disjunct is already handled upstream via
        ``BaseCertifier._filter_sublevel_regions``. The pre-verifier therefore
        checks the stronger condition that positivity, decrease, and
        invariance hold across the full box. This stays conservative while
        avoiding the looser min/max fused single-output graph.
        """
        return LyapunovMultiOutputVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=self.bounds[0].unsqueeze(0),
            ubx=self.bounds[1].unsqueeze(0),
            kappa=self.config.kappa,
            sublevel_tolerance=self.config.sublevel_tolerance,
            condition_margin=self.config.condition_margin,
        )

    def _build_bounded_module(self) -> BoundedModule:
        """Build the bounded LiRPA module used for conservative region checks."""
        dummy_x = th.zeros(1, self.config.state_dim, device=self.device)
        dummy_rho = th.zeros(1, 1, device=self.device)

        bound_opts = {
            "perturb_bound": True,
            "optimize_bound_args": {
                "use_shared_alpha": False,
                "fix_interm_bounds": True,
            },
        }

        return BoundedModule(
            self.verifier,
            (dummy_x, dummy_rho),
            device=self.device,
            verbose=False,
            bound_opts=bound_opts,
        )

    def _get_suppress_ctx(self):
        """Return context manager controlling native LiRPA output."""
        if self.config.suppress_native_output:
            return suppress_native_output(suppress_stderr=True)
        return nullcontext()

    @staticmethod
    def _unpack_regions(bs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Unpack ``(N, 2, state_dim)`` regions into lower/upper bounds."""
        return bs[:, 0], bs[:, 1]

    def certify_boxes(
        self,
        bs: th.Tensor,
        rho: float,
        early_exit: bool = False,
    ) -> th.Tensor:
        """Conservatively certify packed regions with LiRPA bounds.

        Regions marked ``True`` are safe. ``False`` means only "unknown" and
        must be passed to a stronger verifier such as ABCrown.
        """
        if self.lirpa_model is None or not self.fallback_methods:
            raise RuntimeError("ConservativeLiRPAVerifier is not properly initialized.")

        num_regions = len(bs)
        if num_regions == 0:
            return th.empty((0,), dtype=th.bool, device=self.device)

        effective_batch_size = min(self.config.batch_size, num_regions)
        is_certified = th.zeros(num_regions, dtype=th.bool, device=self.device)
        lbs, ubs = self._unpack_regions(bs)
        centers = 0.5 * (lbs + ubs)

        for start_idx in range(0, num_regions, effective_batch_size):
            end_idx = min(start_idx + effective_batch_size, num_regions)
            batch_lbs = lbs[start_idx:end_idx]
            batch_ubs = ubs[start_idx:end_idx]
            batch_centers = centers[start_idx:end_idx]

            batch_len = batch_centers.shape[0]
            pad_len = self.config.batch_size - batch_len
            if pad_len > 0:
                batch_lbs = th.cat([batch_lbs, batch_lbs[-1:].expand(pad_len, -1)], dim=0)
                batch_ubs = th.cat([batch_ubs, batch_ubs[-1:].expand(pad_len, -1)], dim=0)
                batch_centers = th.cat(
                    [batch_centers, batch_centers[-1:].expand(pad_len, -1)],
                    dim=0,
                )

            ptb = PerturbationLpNorm(norm=float("inf"), x_L=batch_lbs, x_U=batch_ubs)
            bounded_input = BoundedTensor(batch_centers, ptb)
            batch_rho = th.full(
                (self.config.batch_size, 1),
                rho,
                dtype=th.float32,
                device=self.device,
            )

            lb_out = None
            ub_out = None
            certified_mask = None
            for candidate_method in self.fallback_methods:
                try:
                    with self._get_suppress_ctx():
                        if candidate_method == "alpha-crown":
                            lb_out, ub_out = self.lirpa_model.compute_bounds(
                                x=(bounded_input, batch_rho),
                                method=candidate_method,
                            )
                        else:
                            with th.no_grad():
                                lb_out, ub_out = self.lirpa_model.compute_bounds(
                                    x=(bounded_input, batch_rho),
                                    method=candidate_method,
                                )
                    certified_mask = self._evaluate_certified_mask(lb_out, ub_out, batch_len)
                    break
                except Exception as exc:
                    if (
                        candidate_method == "alpha-crown"
                        and self._is_alpha_shape_mismatch_error(exc)
                    ):
                        self._disable_alpha_crown_for_run(exc)
                    self.lirpa_model = self._build_bounded_module()
                    __logger__.warning(
                        "Conservative LiRPA method '%s' failed on batch: %s",
                        candidate_method,
                        exc,
                    )

            if certified_mask is not None:
                is_certified[start_idx:end_idx] = certified_mask

            del batch_lbs, batch_ubs, batch_centers, bounded_input, ptb, lb_out, ub_out, certified_mask

            if early_exit and not is_certified[start_idx:end_idx].all():
                __logger__.info("Early exit from conservative pre-verification due to uncertified batch.")
                return is_certified

        __logger__.info("Conservative pre-verification complete: %d/%d certified.", is_certified.sum(), num_regions)
        return is_certified
