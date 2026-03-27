from __future__ import annotations

import torch as th
import torch.nn as nn

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

class ABCrownCertifier(BaseCertifier):
    """
    Lyapunov certifier using the full Alpha-Beta-CROWN framework.
    Combines spatial branch-and-bound with neural activation branch-and-bound.
    """
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
        self.wrapped_model = _ABCrownModelWrapper(self.verifier, self.device)
        self.wrapped_model.kappa.fill_(self.config.kappa)
        self.wrapped_model.eval()

    @staticmethod
    def _is_verified_status(status: str) -> bool:
        normalized = str(status).strip().lower()
        return normalized == "verified" or normalized.startswith("safe")

    def _solve_box_with_model(
        self,
        lb: th.Tensor,
        ub: th.Tensor,
    ) -> bool:
        if lb.ndim > 1:
            lb = lb.squeeze(0)
            ub = ub.squeeze(0)

        with self._get_suppress_ctx():
            x = input_vars(self.config.state_dim)
            y = output_vars(1)

            input_constraint = (x >= lb) & (x <= ub)
            output_constraint = (y[0] < self.config.condition_tolerance)

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


    def _certify_batched_regions(
            self,
            bs: th.Tensor,
            rho: float,
            early_exit: bool = True,
    ) -> tuple[th.Tensor, th.Tensor]:
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
        tuple[th.Tensor, th.Tensor]
            A tuple containing:
            - is_certified: A boolean tensor indicating whether each region is certified.
            - centers_out: A tensor containing the centers of the regions.
        """
        if self.wrapped_model is None or self.abcrown_config is None:
            raise RuntimeError("ABCrownCertifier backend is not properly initialized.")
        
        num_regions = len(bs)
        is_certified = th.zeros(num_regions, dtype=th.bool, device=self.device)

        lbs, ubs = self._unpack_regions(bs)
        centers_out = (lbs + ubs) / 2.0
        self.wrapped_model.rho.fill_(rho)

        for idx in range(num_regions):
            lb = lbs[idx]
            ub = ubs[idx]
            is_certified[idx] = self._solve_box_with_model(lb=lb, ub=ub)
            
            if early_exit and not is_certified[idx]:
                break

        return is_certified, centers_out