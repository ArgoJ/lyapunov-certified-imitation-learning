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

from .certifier_base import BaseCertifier
from ..utils.package_logger import get_package_logger


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

    def setup_backend(self) -> None:
        self.abcrown_config = (
            ConfigBuilder.from_defaults()
            .set(general__device=self.device.type)
            ()
        )

        self.wrapped_model = _ABCrownModelWrapper(self.verifier, self.device)
        self.wrapped_model.kappa.fill_(self.config.kappa)
        self.wrapped_model.eval()

    def _certify_batched_regions(
            self,
            lbs: th.Tensor,
            ubs: th.Tensor,
            rho: float,
            early_exit: bool = True,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        num_regions = len(lbs)
        is_certified = th.zeros(num_regions, dtype=th.bool, device=self.device)
        max_uppers = th.full((num_regions,), float("inf"), dtype=th.float32, device=self.device)
        centers_out = (lbs + ubs) / 2.0

        self.wrapped_model.rho.fill_(rho)
        for idx in range(num_regions):
            lb = lbs[idx]
            ub = ubs[idx]

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
                    config=self.abcrown_config
                )

                res = solver.solve()

            status = str(res.status).strip().lower()
            is_safe_status = status == "verified" or status.startswith("safe")
            __logger__.debug(f"Region {idx}: status={status}, lb={lb.cpu().numpy()}, ub={ub.cpu().numpy()}, rho={rho:.6f}")

            if is_safe_status:
                is_certified[idx] = True
                max_uppers[idx] = self.config.condition_tolerance - 1e-4
            else:
                is_certified[idx] = False
                max_uppers[idx] = float("inf")

            if early_exit and not is_certified[idx]:
                break

        return is_certified, centers_out, max_uppers