from __future__ import annotations
import logging
import torch as th
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .certifier_base import BaseCertifier
from .models import LyapunovVerifier

__logger__ = logging.getLogger(__name__)


class LiRPACertifier(BaseCertifier):
    """Lyapunov certifier using auto_LiRPA bound propagation."""

    @staticmethod
    def _build_center_refined_axis_edges(
        lb: th.Tensor,
        ub: th.Tensor,
        num_bins: int,
        refinement_factor: float,
    ) -> th.Tensor:
        """Build 1D grid edges with optional geometric refinement toward zero."""
        if refinement_factor == 1.0 or lb >= 0.0 or ub <= 0.0:
            return th.linspace(lb, ub, steps=num_bins + 1, device=lb.device)

        neg_span = float((-lb).item())
        pos_span = float(ub.item())
        neg_bins = int(round(num_bins * neg_span / (neg_span + pos_span))) if neg_span + pos_span > 0 else 0
        neg_bins = max(1, min(num_bins - 1, neg_bins))
        pos_bins = num_bins - neg_bins

        def _side_edges(span: float, bins: int, sign: float, device: th.device) -> th.Tensor:
            if bins <= 0 or span <= 0.0:
                return th.zeros(1, device=device)
            if bins == 1:
                distances = th.tensor([span], dtype=th.float32, device=device)
            else:
                powers = th.arange(bins - 1, -1, -1, dtype=th.float32, device=device)
                weights = refinement_factor ** powers
                widths = span * weights / weights.sum()
                distances = th.cumsum(widths, dim=0)
            return sign * distances

        neg_edges = _side_edges(neg_span, neg_bins, -1.0, lb.device)
        pos_edges = _side_edges(pos_span, pos_bins, 1.0, ub.device)

        return th.cat([
            th.tensor([lb.item()], dtype=th.float32, device=lb.device),
            neg_edges.flip(0)[1:],
            th.zeros(1, dtype=th.float32, device=lb.device),
            pos_edges[:-1],
            th.tensor([ub.item()], dtype=th.float32, device=lb.device),
        ])

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
        self.verifier = self._setup_verifier()
        self.lirpa_model = self._get_bounded_module()
        self.fallback_methods = self._get_fallback_methods(self.config.cert_method)
        self._alpha_crown_disabled = False

    def _build_regions(self) -> th.Tensor:
        """Use the explicit external grid split required by LiRPA."""
        origin_exclusion = self._resolve_origin_exclusion()
        lbx, ubx = self.bounds[0], self.bounds[1]
        bin_counts = self.config.bins_per_dim
        refinement_factors = self.config.center_refinement_factor

        lb_axes = []
        ub_axes = []
        for idx in range(self.config.state_dim):
            edges = self._build_center_refined_axis_edges(
                lb=lbx[idx],
                ub=ubx[idx],
                num_bins=int(bin_counts[idx]),
                refinement_factor=float(refinement_factors[idx]),
            )
            lb_axes.append(edges[:-1])
            ub_axes.append(edges[1:])

        lb_mesh = th.meshgrid(*lb_axes, indexing="ij")
        ub_mesh = th.meshgrid(*ub_axes, indexing="ij")
        lbs = th.stack([axis.flatten() for axis in lb_mesh], dim=1)
        ubs = th.stack([axis.flatten() for axis in ub_mesh], dim=1)

        overlaps_origin_per_dim = (lbs < origin_exclusion.unsqueeze(0)) & (ubs > -origin_exclusion.unsqueeze(0))
        overlaps_origin = overlaps_origin_per_dim.all(dim=1)
        valid_mask = ~overlaps_origin

        __logger__.info(
            "Origin exclusion resolved to %s; filtered %d/%d root regions overlapping the centered exclusion box.",
            [float(value) for value in origin_exclusion.detach().cpu().tolist()],
            int(overlaps_origin.sum().item()),
            int(overlaps_origin.numel()),
        )

        return self._pack_regions(lbs[valid_mask], ubs[valid_mask])

    def _setup_verifier(self) -> LyapunovVerifier:
        """Construct the verifier with the global certification box.

        The LiRPA backend still perturbs inputs over each current subregion via
        ``x_L``/``x_U`` in ``_certify_batched_regions``. The closed-loop
        invariance check itself must however stay anchored to the global
        certification box ``B`` from ``self.bounds``.
        """
        global_lbx_batched = self.bounds[0].unsqueeze(0)
        global_ubx_batched = self.bounds[1].unsqueeze(0)

        verifier = LyapunovVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=global_lbx_batched,
            ubx=global_ubx_batched,
            kappa=self.config.kappa,
            sublevel_tolerance=self.config.sublevel_tolerance,
            condition_margin=self.config.condition_margin,
        )

        return verifier

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
            f"Disabling alpha-crown for this run due to backend shape mismatch: {reason}",
        )

    def _get_bounded_module(self) -> BoundedModule:
        """No additional setup needed for auto_LiRPA."""
        dummy_x = th.zeros(1, self.config.state_dim, device=self.device)
        dummy_rho = th.zeros(1, 1, device=self.device)

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
            (dummy_x, dummy_rho),
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

            ub_out = None
            for candidate_method in self.fallback_methods:
                try:
                    with self._get_suppress_ctx():
                        if candidate_method == "alpha-crown":
                            _, ub_out = self.lirpa_model.compute_bounds(
                                x=(bounded_input, b_rho),
                                method=candidate_method
                            )
                        else:
                            with th.no_grad():
                                _, ub_out = self.lirpa_model.compute_bounds(
                                    x=(bounded_input, b_rho),
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