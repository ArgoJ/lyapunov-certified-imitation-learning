from __future__ import annotations

import torch as th
import torch.nn as nn

import os
os.environ.setdefault("PYTORCH_JIT", "0")

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from abcrown import (
    ABCrownSolver, 
    VerificationSpec, 
    ConfigBuilder, 
    input_vars, 
    output_vars
)
from pkg_logger import get_package_logger

from .certifier_base import BaseCertifier


__logger__ = get_package_logger(__name__)


class _ABCrownModelWrapper(nn.Module):
    """
    A wrapper that freezes the dynamic parameters (rho, kappa)
    so that ABCrownSolver can evaluate a clean model x -> y.
    """
    def __init__(self, verifier: nn.Module, device: th.device):
        super().__init__()
        self.verifier = verifier
        self.register_buffer("rho", th.tensor([0.0], dtype=th.float32, device=device))
        self.register_buffer("kappa", th.tensor([0.0], dtype=th.float32, device=device))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.verifier(x, self.rho, self.kappa)


class _NegatedLyapunovModelWrapper(nn.Module):
    """Wrap ``V(x)`` as ``-V(x)`` for outside-sublevel proofs."""

    def __init__(self, lyap_model: nn.Module):
        super().__init__()
        self.lyap_model = lyap_model

    def forward(self, x: th.Tensor) -> th.Tensor:
        return -self.lyap_model(x)


class ABCrownCertifier(BaseCertifier):
    """
    Lyapunov certifier using the full Alpha-Beta-CROWN framework.
    Combines spatial branch-and-bound with neural activation branch-and-bound.
    """

    def setup_backend(self) -> None:
        """Set up the ABCrownSolver and its configuration."""
        self.abcrown_config = (
            ConfigBuilder.from_defaults()
            .set(general__device=self.device.type)
            ()
        )

        self.wrapped_model = _ABCrownModelWrapper(self.verifier, self.device)
        self.wrapped_model.kappa.fill_(self.config.kappa)
        self.wrapped_model.eval()
        self.negated_lyap_model = _NegatedLyapunovModelWrapper(self.lyap_model).to(self.device)
        self.negated_lyap_model.eval()
        dummy_x = th.zeros(1, self.config.state_dim, device=self.device)
        self.ibp_filter_model = BoundedModule(
            self.negated_lyap_model,
            (dummy_x,),
            device=self.device,
            verbose=False,
            bound_opts={"perturb_bound": True},
        )

    @staticmethod
    def _is_verified_status(status: str) -> bool:
        normalized = str(status).strip().lower()
        return normalized == "verified" or normalized.startswith("safe")

    def _solve_box_with_model(
        self,
        model: nn.Module,
        lb: th.Tensor,
        ub: th.Tensor,
        output_upper_bound: float,
        config: dict | None = None,
    ) -> bool:
        with self._get_suppress_ctx():
            x = input_vars(self.config.state_dim)
            y = output_vars(1)

            input_constraint = (x >= lb) & (x <= ub)
            output_constraint = (y[0] < output_upper_bound)

            spec = VerificationSpec.build_spec(
                input_vars=x,
                output_vars=y,
                input_constraint=input_constraint,
                output_constraint=output_constraint,
            )

            solver = ABCrownSolver(
                spec=spec,
                computing_graph=model,
                config=self.abcrown_config if config is None else config,
            )

            res = solver.solve()

        return self._is_verified_status(res.status)

    def _filter_sublevel_regions(
        self,
        lbs: th.Tensor,
        ubs: th.Tensor,
        rho: float,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Keep only boxes not proven to satisfy ``V(x) > rho`` everywhere."""
        if len(lbs) == 0:
            return lbs, ubs

        keep_mask = th.ones(len(lbs), dtype=th.bool, device=self.device)
        negated_threshold = -float(rho)
        max_batch_size = 2048

        for start_idx in range(0, len(lbs), max_batch_size):
            end_idx = min(start_idx + max_batch_size, len(lbs))
            batch_lbs = lbs[start_idx:end_idx]
            batch_ubs = ubs[start_idx:end_idx]
            batch_centers = 0.5 * (batch_lbs + batch_ubs)

            ptb = PerturbationLpNorm(norm=float("inf"), x_L=batch_lbs, x_U=batch_ubs)
            bounded_input = BoundedTensor(batch_centers, ptb)

            with th.no_grad():
                _, ub_out = self.ibp_filter_model.compute_bounds(
                    x=(bounded_input,),
                    method="ibp",
                )

            batch_is_outside = ub_out.flatten() <= negated_threshold
            keep_mask[start_idx:end_idx] = ~batch_is_outside

        if keep_mask.all():
            return lbs, ubs

        __logger__.info(
            "Pruned %d / %d regions proven outside V(x) <= %.6f.",
            int((~keep_mask).sum().item()),
            len(lbs),
            float(rho),
        )
        return lbs[keep_mask], ubs[keep_mask]

    def _certify_batched_regions(
            self,
            lbs: th.Tensor,
            ubs: th.Tensor,
            rho: float,
            early_exit: bool = True,
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Certifies a batch of regions using the ABCrown solver.

        Parameters
        ----------
        lbs : th.Tensor
            Lower bounds of the regions.
        ubs : th.Tensor
            Upper bounds of the regions.
        rho : float
            The rho parameter for the certification.
        early_exit : bool, optional
            Whether to exit early if a region is not certified. Defaults to True.

        Returns
        -------
        tuple[th.Tensor, th.Tensor]
            A tuple containing:
            - is_certified: A boolean tensor indicating whether each region is certified.
            - centers_out: A tensor containing the centers of the regions.
        """
        num_regions = len(lbs)
        is_certified = th.zeros(num_regions, dtype=th.bool, device=self.device)
        centers_out = (lbs + ubs) / 2.0

        self.wrapped_model.rho.fill_(rho)
        for idx in range(num_regions):
            lb = lbs[idx]
            ub = ubs[idx]
            is_certified[idx] = self._solve_box_with_model(
                model=self.wrapped_model,
                lb=lb,
                ub=ub,
                output_upper_bound=self.config.condition_tolerance,
                config=self.abcrown_config,
            )
            __logger__.debug(
                "Region %d: certified=%s, lb=%s, ub=%s, rho=%.6f",
                idx,
                bool(is_certified[idx].item()),
                lb.cpu().numpy(),
                ub.cpu().numpy(),
                rho,
            )

            if early_exit and not is_certified[idx]:
                break

        return is_certified, centers_out
