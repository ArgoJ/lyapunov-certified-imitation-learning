from __future__ import annotations

import torch as th
import torch.nn as nn
import numpy as np

# --- Neue Alpha-Beta-CROWN API Imports ---
from abcrown import (
    ABCrownSolver, 
    VerificationSpec, 
    ConfigBuilder, 
    input_vars, 
    output_vars
)

from .config import LyapunovCertificationConfig
from .certifier_base import BaseCertifier, RegionCertificationResult


class _ABCrownModelWrapper(nn.Module):
    """
    A wrapper that freezes the dynamic parameters (rho, kappa)
    so that ABCrownSolver can evaluate a clean model x -> y.
    """
    def __init__(self, verifier: nn.Module, rho: float, kappa: float, device: th.device):
        super().__init__()
        self.verifier = verifier
        self.rho = th.tensor([rho], dtype=th.float32, device=device)
        self.kappa = th.tensor([kappa], dtype=th.float32, device=device)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.verifier(x, self.rho, self.kappa)


class ABCrownCertifier(BaseCertifier):
    """
    Lyapunov certifier using the full Alpha-Beta-CROWN framework.
    Combines spatial branch-and-bound with neural activation branch-and-bound.
    """

    def build_regions(self) -> tuple[th.Tensor, th.Tensor]:
        train_diameter = th.max(th.abs(self.bounds)).item()
        excl = self.config.cert_origin_exclusion or min(train_diameter * 0.01, 0.1)

        lb_x, ub_x = self.bounds[0][0].item(), self.bounds[1][0].item()
        lb_y, ub_y = self.bounds[0][1].item(), self.bounds[1][1].item()

        # Wir zerlegen den 2D-Raum (ohne das Zentrum) in 4 große Makro-Rechtecke:
        # 1. Links vom Zentrum (volle Höhe)
        # 2. Rechts vom Zentrum (volle Höhe)
        # 3. Über dem Zentrum (nur Mittelstreifen)
        # 4. Unter dem Zentrum (nur Mittelstreifen)
        boxes = [
            ([lb_x, lb_y], [-excl, ub_y]),           # Links
            ([excl, lb_y], [ub_x, ub_y]),            # Rechts
            ([-excl, excl], [excl, ub_y]),           # Oben
            ([-excl, lb_y], [excl, -excl])           # Unten
        ]

        lbs = th.tensor([b[0] for b in boxes], dtype=th.float32, device=self.device)
        ubs = th.tensor([b[1] for b in boxes], dtype=th.float32, device=self.device)

        return lbs, ubs

    def setup_backend(self) -> None:
        self.abcrown_config = (
            ConfigBuilder.from_defaults()
            .set(general__device=self.device.type)
            ()
        )

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
        centers_out = th.empty_like(lbs)

        wrapped_model = _ABCrownModelWrapper(self.verifier, rho, self.config.kappa, self.device)
        for idx in range(num_regions):
            lb = lbs[idx]
            ub = ubs[idx]
            center = (lb + ub) / 2.0
            centers_out[idx] = center

            x = input_vars(self.config.state_dim)
            y = output_vars(1)

            input_constraint = (x >= lb) & (x <= ub)
            output_constraint = (y[0] > self.config.condition_tolerance)

            spec = VerificationSpec.build_spec(
                input_vars=x,
                output_vars=y,
                input_constraint=input_constraint,
                output_constraint=output_constraint,
            )

            solver = ABCrownSolver(
                spec=spec,
                computing_graph=wrapped_model,
                config=self.abcrown_config
            )

            res = solver.solve()

            if res.status in ["verified", "safe", "safe-incomplete"]:
                is_certified[idx] = True
                max_uppers[idx] = self.config.condition_tolerance - 1e-4
            else:
                is_certified[idx] = False
                max_uppers[idx] = float("inf")

            if early_exit and not is_certified[idx]:
                break

        return is_certified, centers_out, max_uppers

    def certify_regions(
            self, 
            rho: float,
            collect_details: bool = True
    ) -> RegionCertificationResult:
        lbs, ubs = self.regions
    
        if len(lbs) == 0:
            empty = th.empty((0, 2), device=self.device)
            return empty, empty, empty, empty, empty

        is_safe, centers, _ = self._certify_batched_regions(
            lbs, ubs, rho,
            early_exit=not collect_details
        )

        c_lbs = lbs[is_safe]
        c_ubs = ubs[is_safe]

        failed_mask = ~is_safe
        f_lbs = lbs[failed_mask]
        f_ubs = ubs[failed_mask]
        cex = centers[failed_mask]

        c_lbs_np = c_lbs.cpu().numpy() if c_lbs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)
        c_ubs_np = c_ubs.cpu().numpy() if c_ubs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)
        f_lbs_np = f_lbs.cpu().numpy() if f_lbs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)
        f_ubs_np = f_ubs.cpu().numpy() if f_ubs.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)

        certified_regions_np = np.stack([c_lbs_np, c_ubs_np], axis=1) if c_lbs_np.shape[0] > 0 else np.empty((0, 2, self.config.state_dim), dtype=np.float32)
        failed_regions_np = np.stack([f_lbs_np, f_ubs_np], axis=1) if f_lbs_np.shape[0] > 0 else np.empty((0, 2, self.config.state_dim), dtype=np.float32)
        counter_examples_np = cex.cpu().numpy() if cex.numel() > 0 else np.empty((0, self.config.state_dim), dtype=np.float32)

        return RegionCertificationResult(
            success=failed_regions_np.shape[0] == 0,
            counter_examples=counter_examples_np,
            failed_regions=failed_regions_np,
            certified_regions=certified_regions_np,
        )