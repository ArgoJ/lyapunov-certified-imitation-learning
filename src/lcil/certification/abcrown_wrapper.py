from __future__ import annotations
import logging

import torch as th
import torch.nn as nn

from abcrown import (
    ABCrownSolver, 
    VerificationSpec, 
    ConfigBuilder, 
    input_vars, 
    output_vars
)

from .conservative_preverifier import ConservativeLiRPAVerifier
from .certifier_base import BaseCertifier
from .models import LyapunovMultiOutputVerifier

__logger__ = logging.getLogger(__name__)

class _ABCrownModelWrapper(nn.Module):
    """
    A wrapper that freezes the dynamic rho parameter
    so that ABCrownSolver can evaluate a clean model x -> y.
    """
    def __init__(self, verifier: nn.Module, device: th.device):
        super().__init__()
        self.verifier = verifier
        self.register_buffer("rho", th.tensor(0.0, dtype=th.float32, device=device))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.verifier(x, self.rho)

class ABCrownCertifier(BaseCertifier):
    """
    Lyapunov certifier using the full Alpha-Beta-CROWN framework.
    Combines spatial branch-and-bound with neural activation branch-and-bound.
    """

    _CONSERVATIVE_PRECHECK_METHOD = "crown-ibp"
    _LARGE_BATCH_PRECHECK_SKIP_THRESHOLD = 128

    def __init__(
        self,
        policy_model,
        lyap_model,
        dyn_model,
        config,
        device: th.device = th.device("cpu"),
    ):
        super().__init__(policy_model, lyap_model, dyn_model, config, device)
        self.abcrown_config = None
        self.abcrown_verifier = None
        self.conservative_verifier = None
        self.skip_large_batch_precheck = False
        self.wrapped_model = None

    def setup_backend(self) -> None:
        """Set up the ABCrownSolver and its configuration."""
        self.abcrown_config = (
            ConfigBuilder.from_defaults()
            .set(general__device=self.device.type)
            .set(bab__branching__method="babsr")
            .set(solver__batch_size=4096)
            ()
        )
        
        self.verifier = self._setup_verifier()
        self.wrapped_model = _ABCrownModelWrapper(self.verifier, self.device)
        self.wrapped_model.eval()
        self.conservative_verifier = self._setup_conservative_verifier()

    @staticmethod
    def _is_verified_status(status: str) -> bool:
        normalized = str(status).strip().lower()
        return normalized == "verified" or normalized.startswith("safe")

    def _setup_verifier(self):
        lbx_batched = self.bounds[0].unsqueeze(0)
        ubx_batched = self.bounds[1].unsqueeze(0)
        
        return LyapunovMultiOutputVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=lbx_batched,
            ubx=ubx_batched,
            kappa=self.config.kappa,
            sublevel_tolerance=self.config.sublevel_tolerance,
            condition_margin=self.config.condition_margin,
        )

    def _setup_conservative_verifier(self) -> ConservativeLiRPAVerifier:
        """Build a cheap conservative pre-verifier for safe-region pruning."""
        return ConservativeLiRPAVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            bounds=self.bounds,
            config=self.config,
            device=self.device,
            method=self._CONSERVATIVE_PRECHECK_METHOD,
        )

    def _certify_with_conservative_verifier(
        self,
        bs: th.Tensor,
        rho: float,
    ) -> th.Tensor:
        """Conservatively pre-certify easy regions before invoking ABCrown."""
        if self.conservative_verifier is None:
            return th.zeros(len(bs), dtype=th.bool, device=self.device)
        if self.skip_large_batch_precheck and len(bs) >= self._LARGE_BATCH_PRECHECK_SKIP_THRESHOLD:
            return th.zeros(len(bs), dtype=th.bool, device=self.device)

        try:
            is_safe = self.conservative_verifier.certify_boxes(
                bs,
                rho,
                early_exit=False,
            )
            if (
                len(bs) >= self._LARGE_BATCH_PRECHECK_SKIP_THRESHOLD
                and not bool(is_safe.any().item())
            ):
                self.skip_large_batch_precheck = True
                __logger__.info(
                    "Conservative pre-verifier certified 0/%d on a large batch; skipping it for future large batches in this run.",
                    len(bs),
                )
            return is_safe
        except Exception as exc:
            __logger__.warning(
                "Conservative pre-verifier failed; falling back to ABCrown only: %s",
                exc,
            )
            self.conservative_verifier = None
            return th.zeros(len(bs), dtype=th.bool, device=self.device)

    def _solve_box_with_model(
        self,
        lb: th.Tensor,
        ub: th.Tensor,
    ) -> bool:
        if lb.ndim > 1:
            lb = lb.squeeze(0)
            ub = ub.squeeze(0)

        rho_value = float(self.wrapped_model.rho.item())

        with self._get_suppress_ctx():
            x = input_vars(self.config.state_dim)
            y = output_vars(2 + self.config.state_dim)

            input_constraint = (x >= lb) & (x <= ub)
            output_constraint = self._build_safe_output_constraint(
                y=y,
                rho=rho_value,
            )

            spec = VerificationSpec.build_spec(
                input_vars=x,
                output_vars=y,
                input_constraint=input_constraint,
                output_constraint=output_constraint,
            )

            solver = ABCrownSolver(
                spec=spec,
                computing_graph=self.wrapped_model,
                config=self.abcrown_config,
            )

            res = solver.solve()

        return self._is_verified_status(res.status)

    def _build_safe_output_constraint(
        self,
        y,
        rho: float,
    ):
        """Build the safe-region predicate for the multi-output verifier.

        The safe condition enforces the paper-style implication over the
        global certification box ``B``:

        ``V(x) > rho + tol_sublevel`` OR
        ``(V(x) >= -tol_cond) AND (decrease margin >= -tol_cond) AND (x_next in B)``.
        """
        safe_outside_sublevel = y[1] > (rho + self.config.sublevel_tolerance)
        safe_positive = y[1] > (-self.config.condition_tolerance)
        safe_decrease = y[0] > (-self.config.condition_tolerance)

        global_lb = self.bounds[0]
        global_ub = self.bounds[1]
        safe_x_next = None
        for idx in range(self.config.state_dim):
            coord_safe = (y[idx + 2] > (float(global_lb[idx]) - self.config.condition_tolerance)) & (
                y[idx + 2] < (float(global_ub[idx]) + self.config.condition_tolerance)
            )
            safe_x_next = coord_safe if safe_x_next is None else (safe_x_next & coord_safe)

        return safe_outside_sublevel | (safe_positive & safe_decrease & safe_x_next)


    def _certify_batched_regions(
            self,
            bs: th.Tensor,
            rho: float,
            early_exit: bool = True,
        ) -> th.Tensor:
        """
        Certifies a batch of regions using the ABCrown solver.

        Parameters
        ----------
        bs : th.Tensor
            Packed region bounds with shape ``(n, 2, state_dim)``.
        rho : float
            The rho parameter for the certification.
        early_exit : bool, optional
            Whether to exit early if a region is not certified. Defaults to True.

        Returns
        -------
        th.Tensor
            Boolean tensor indicating whether each region is certified.
        """
        if self.wrapped_model is None or self.abcrown_config is None:
            raise RuntimeError("ABCrownCertifier backend is not properly initialized.")
        
        num_regions = len(bs)
        if num_regions == 0:
            return th.empty((0,), dtype=th.bool, device=self.device)

        is_certified = th.zeros(num_regions, dtype=th.bool, device=self.device)
        conservative_safe = self._certify_with_conservative_verifier(bs, rho)
        is_certified |= conservative_safe

        remaining_idx = (~conservative_safe).nonzero(as_tuple=False).flatten()
        if remaining_idx.numel() == 0:
            return is_certified

        lbs, ubs = self._unpack_regions(bs)
        self.wrapped_model.rho.fill_(rho)

        for idx in remaining_idx.tolist():
            lb = lbs[idx]
            ub = ubs[idx]
            is_certified[idx] = self._solve_box_with_model(lb=lb, ub=ub)
            
            if early_exit and not is_certified[idx]:
                break

        return is_certified