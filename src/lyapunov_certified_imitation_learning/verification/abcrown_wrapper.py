"""
Lyapunov verification using auto_LiRPA (CROWN / alpha-CROWN) with
domain splitting for complete certification.

Replaces the CPLEX-based MIP verifier from the paper with neural-network
bound-propagation methods that scale to larger networks.

Verification targets
--------------------
1. **Positivity**  V(x) > 0  for  ||x||_inf >= err_origin
2. **Decrease**    V(f(x, pi(x))) - V(x) <= 0

Both conditions are checked per sub-region using auto_LiRPA's
``BoundedModule.compute_bounds``.  If the incomplete bound is
insufficient, the region is bisected recursively
(analogous to alpha-beta-CROWN branch-and-bound).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from ..utils.package_logger import PackageLogger

logger = PackageLogger.get_logger(__name__)


# ======================================================================
# Core bound computation
# ======================================================================
def compute_bounds(
    model: nn.Module,
    x_L: torch.Tensor,
    x_U: torch.Tensor,
    method: str = "IBP+backward",
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute output bounds of ``model(x)`` for x in [x_L, x_U].

    Parameters
    ----------
    model : nn.Module
    x_L, x_U : Tensor, shape ``(1, dim)`` or ``(dim,)``
    method : str
        ``'IBP'``, ``'IBP+backward'`` (IBP intermediates + CROWN output),
        ``'backward'`` (CROWN), or
        ``'CROWN-Optimized'`` (alpha-CROWN, may fail on complex graphs).
    device : str

    Returns
    -------
    lb, ub : Tensor, shape ``(1, output_dim)``
    """
    if x_L.dim() == 1:
        x_L = x_L.unsqueeze(0)
    if x_U.dim() == 1:
        x_U = x_U.unsqueeze(0)

    x_L = x_L.to(device).float()
    x_U = x_U.to(device).float()
    x = (x_L + x_U) / 2.0

    model = model.to(device).eval()
    lirpa_model = BoundedModule(model, x)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=x_L, x_U=x_U)
    bounded_x = BoundedTensor(x, ptb)

    try:
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=method)
    except RuntimeError:
        # Fall back to plain IBP for graphs with complex bivariate ops
        logger.debug("Method '%s' failed, falling back to IBP", method)
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method="IBP")
    return lb, ub


# ======================================================================
# LyapunovVerifier
# ======================================================================
class LyapunovVerifier:
    """
    Verifies Lyapunov conditions using auto_LiRPA bound propagation
    with recursive domain splitting.

    Parameters
    ----------
    device : str
    method : str
        Bound-propagation method (default ``'IBP+backward'``).
        Use ``'IBP+backward'`` for models with bivariate nonlinearities
        (e.g. sin/cos * state).  ``'CROWN-Optimized'`` can fail on such
        graphs due to intermediate-bound batching.
    """

    def __init__(self, device: str = "cpu",
                 method: str = "IBP+backward"):
        self.device = device
        self.method = method

    # ------------------------------------------------------------------
    # Elementary checks
    # ------------------------------------------------------------------
    def verify_decrease(
        self,
        closed_loop_model: nn.Module,
        x_L: torch.Tensor,
        x_U: torch.Tensor,
    ) -> tuple[bool, float]:
        """
        Verify V(x_next) - V(x) <= 0 over [x_L, x_U].

        Returns
        -------
        verified : bool
        upper_bound : float
            Global upper bound on V(x_next) - V(x).
        """
        _, ub = compute_bounds(
            closed_loop_model, x_L, x_U, self.method, self.device
        )
        max_ub = ub.max().item()
        return max_ub <= 0, max_ub

    def verify_positivity(
        self,
        lyap_model: nn.Module,
        x_L: torch.Tensor,
        x_U: torch.Tensor,
    ) -> tuple[bool, float]:
        """
        Verify V(x) >= 0 over [x_L, x_U].

        Returns
        -------
        verified : bool
        lower_bound : float
            Global lower bound on V(x).
        """
        lb, _ = compute_bounds(
            lyap_model, x_L, x_U, self.method, self.device
        )
        min_lb = lb.min().item()
        return min_lb >= 0, min_lb

    # ------------------------------------------------------------------
    # Region certification (positivity + decrease)
    # ------------------------------------------------------------------
    def certify_region(
        self,
        closed_loop_model: nn.Module,
        lyap_model: nn.Module,
        x_L: torch.Tensor,
        x_U: torch.Tensor,
        err_origin: float = 0.1,
    ) -> tuple[bool, float]:
        """
        Certify both Lyapunov conditions over a single region.

        1. Decrease: V(x_next) - V(x) <= 0
        2. Positivity: V(x) > 0  (only for regions away from origin)

        Returns
        -------
        satisfied : bool
        bound_value : float
        """
        # 1. Decrease
        dec_ok, dec_ub = self.verify_decrease(closed_loop_model, x_L, x_U)
        if not dec_ok:
            logger.debug("Decrease not verified: ub=%.6f", dec_ub)
            return False, dec_ub

        # 2. Positivity (skip if region overlaps the origin ball)
        x_L_np = x_L.cpu().numpy().flatten()
        x_U_np = x_U.cpu().numpy().flatten()
        near_origin = all(
            x_L_np[i] <= err_origin and x_U_np[i] >= -err_origin
            for i in range(len(x_L_np))
        )
        if not near_origin:
            pos_ok, pos_lb = self.verify_positivity(lyap_model, x_L, x_U)
            if not pos_ok:
                logger.debug("Positivity not verified: lb=%.6f", pos_lb)
                return False, pos_lb

        return True, dec_ub

    # ------------------------------------------------------------------
    # Domain-splitting verifier (BaB-style)
    # ------------------------------------------------------------------
    def verify_with_splitting(
        self,
        model: nn.Module,
        x_L: torch.Tensor,
        x_U: torch.Tensor,
        max_depth: int = 4,
    ) -> bool:
        """
        Recursively bisect the domain until the bound proves the
        property or ``max_depth`` is reached.

        The property checked is:  model(x) <= 0  for x in [x_L, x_U].
        """
        _, ub = compute_bounds(model, x_L, x_U, self.method, self.device)
        if ub.max().item() <= 0:
            return True
        if max_depth == 0:
            return False

        # Split along widest dimension
        widths = x_U - x_L
        if widths.dim() == 2:
            widths = widths.squeeze(0)
        split_dim = torch.argmax(widths).item()
        mid = (x_L.flatten()[split_dim] + x_U.flatten()[split_dim]) / 2.0

        x_L_flat = x_L.flatten().clone()
        x_U_flat = x_U.flatten().clone()

        ub_left = x_U_flat.clone()
        ub_left[split_dim] = mid

        lb_right = x_L_flat.clone()
        lb_right[split_dim] = mid

        return (
            self.verify_with_splitting(model, x_L_flat, ub_left, max_depth - 1)
            and self.verify_with_splitting(model, lb_right, x_U_flat, max_depth - 1)
        )

    # ------------------------------------------------------------------
    # Batch certification over a list of sub-regions
    # ------------------------------------------------------------------
    def certify_all_regions(
        self,
        closed_loop_model: nn.Module,
        lyap_model: nn.Module,
        certify_list: Sequence[list[float]],
        max_split_depth: int = 3,
    ) -> tuple[bool, list, torch.Tensor]:
        """
        Certify the Lyapunov decrease condition over every sub-region
        in ``certify_list``.

        Each element of *certify_list* is a 12-element list:
            [lb_x, ub_x, lb_y, ub_y, lb_theta, ub_theta,
             lb_xd, ub_xd, lb_yd, ub_yd, lb_thetad, ub_thetad]

        For regions that fail the direct check, domain splitting (up to
        ``max_split_depth`` bisections) is attempted.

        Returns
        -------
        all_verified : bool
        failed_regions : list[list[float]]
        counterexamples : Tensor, shape (M, 6)  (centres of failed regions)
        """
        failed_regions: list[list[float]] = []
        counterexamples: list[torch.Tensor] = []

        for idx, element in enumerate(certify_list):
            lb = torch.tensor(
                [element[0], element[2], element[4],
                 element[6], element[8], element[10]],
                dtype=torch.float32,
            )
            ub = torch.tensor(
                [element[1], element[3], element[5],
                 element[7], element[9], element[11]],
                dtype=torch.float32,
            )

            # Try direct + splitting verification
            verified = self.verify_with_splitting(
                closed_loop_model, lb, ub, max_depth=max_split_depth
            )

            if not verified:
                failed_regions.append(element)
                counterexamples.append(((lb + ub) / 2.0).unsqueeze(0))
                logger.info("Region %d / %d FAILED", idx, len(certify_list))
            else:
                if (idx + 1) % 50 == 0:
                    logger.info(
                        "Region %d / %d verified", idx + 1, len(certify_list)
                    )

        all_verified = len(failed_regions) == 0
        if counterexamples:
            ce_tensor = torch.cat(counterexamples, dim=0)
        else:
            ce_tensor = torch.empty(0, 6)

        return all_verified, failed_regions, ce_tensor

