"""
Lyapunov training loop with counterexample augmentation.

Matches the paper's pipeline:

1. **Pre-train** the Lyapunov network using auto_LiRPA positivity bounds
   plus PGD-found counterexamples for the decrease condition.
2. **Main training** alternates gradient updates with formal certification
   (via ``LyapunovVerifier``).  Failed regions produce counterexamples
   that are fed back into the next training round.
3. **Outer loop** with learning-rate decay and re-certification until
   all sub-regions are verified.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from ..models.dynamics import STATE_DIM
from ..verification.counterexample import find_counterexamples, lyap_diff
from ..verification.abcrown_wrapper import LyapunovVerifier
from ..utils.package_logger import PackageLogger

__logger__ = PackageLogger.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants matching the paper
# ---------------------------------------------------------------------------
TRAIN_DIAMETER = 1.0
CONST_C_DEFAULT = 0.12


# ---------------------------------------------------------------------------
# Build the grid of verification sub-regions
# ---------------------------------------------------------------------------
def build_certify_list(train_diameter: float = TRAIN_DIAMETER) -> list[list[float]]:
    """
    Build the verification sub-regions matching the paper's grid.

    Returns a list of 512 elements, each a 12-element list::

        [lb_x, ub_x, lb_y, ub_y, lb_theta, ub_theta,
         lb_xd, ub_xd, lb_yd, ub_yd, lb_thetad, ub_thetad]
    """
    split_vel = [-1.0, -0.5, 0.0, 0.5, 1.0]
    split_theta = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

    certify_list: list[list[float]] = []
    d = train_diameter
    for ii in range(len(split_theta) - 1):
        for jj in range(len(split_vel) - 1):
            for kk in range(len(split_vel) - 1):
                for ll in range(len(split_vel) - 1):
                    certify_list.append([
                        -d, d,                                 # x
                        -d, d,                                 # y
                        split_theta[ii], split_theta[ii + 1],  # theta
                        split_vel[jj],   split_vel[jj + 1],   # x_dot
                        split_vel[kk],   split_vel[kk + 1],   # y_dot
                        split_vel[ll],   split_vel[ll + 1],    # theta_dot
                    ])
    return certify_list


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------
def pre_train(
    lyap_model: torch.nn.Module,
    policy_model: torch.nn.Module,
    device: str,
    max_iters: int = 1000,
    lr: float = 0.00625,
    const_c: float = CONST_C_DEFAULT,
) -> None:
    """
    Pre-train the Lyapunov network (policy weights are frozen).

    Uses auto_LiRPA CROWN lower bound on V(x) over the full domain
    to penalise negative Lyapunov values, and PGD counterexamples
    for the decrease condition.
    """
    __logger__.info("Starting pre-training")
    N = 500
    optimizer = torch.optim.Adam(list(lyap_model.parameters()), lr=lr)

    bound = torch.tensor(
        [[TRAIN_DIAMETER] * STATE_DIM], dtype=torch.float32
    ).to(device)

    x = (torch.zeros(N, STATE_DIM).uniform_(-1, 1) * bound.squeeze(0)).to(device)
    x_all = x.clone()
    x_0 = torch.zeros([1, STATE_DIM]).to(device)

    for it in range(max_iters):
        # --- auto_LiRPA lower bound on V --------------------------------
        model_bound = BoundedModule(lyap_model, x_0)
        ptb = PerturbationLpNorm(norm=np.inf, x_L=-bound, x_U=bound)
        state_range = BoundedTensor(x_0, ptb)
        lyap_lb, _ = model_bound.compute_bounds(
            x=(state_range,), method="backward"
        )

        # --- sample-based decrease --------------------------------------
        diff, _ = lyap_diff(policy_model, lyap_model, x)

        loss = (
            F.relu(-lyap_lb) + 1.2 * F.relu(diff + const_c)
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- periodic PGD counter-example search -------------------------
        if it % 10 == 0:
            ce = find_counterexamples(
                policy_model, lyap_model, device, bounds=bound.squeeze(0)
            )
            if len(ce) > 0:
                __logger__.debug("Iter %d: %d counterexamples", it, ce.size(0))

            idx = np.random.choice(x_all.size(0), size=min(x_all.size(0), 512))
            x = torch.cat((ce, x_all[idx]), dim=0) if len(ce) else x_all[idx]
            x_all = torch.cat((ce, x_all), dim=0) if len(ce) else x_all

        if it % 10 == 0 and loss.item() == 0:
            break

    __logger__.info("Pre-training done (%d iters, loss=%.6f)", it + 1, loss.item())


# ---------------------------------------------------------------------------
# Main Lyapunov training iteration
# ---------------------------------------------------------------------------
def lyap_train_main(
    policy_model: torch.nn.Module,
    lyap_model: torch.nn.Module,
    closed_loop_model: torch.nn.Module,
    certify_counter_examples: list[torch.Tensor],
    certify_list: Sequence[list[float]],
    lr: float,
    device: str,
    const_c: float = CONST_C_DEFAULT,
    max_iters: int = 1000,
) -> tuple[list[torch.Tensor], list, bool, list[float]]:
    """
    One round of Lyapunov training + certification.

    Returns
    -------
    certify_counter_examples : list[Tensor]
        Updated list (may have new entries).
    failed_regions : list[list[float]]
        Sub-regions that failed certification.
    all_verified : bool
    ce_diffs : list[float]
        Lyapunov-difference values at the new counterexamples.
    """
    __logger__.info("Training round  CONST_C=%.4f  lr=%.6f", const_c, lr)
    N = 500
    optimizer = torch.optim.Adam(
        list(policy_model.parameters()) + list(lyap_model.parameters()),
        lr=lr,
    )

    bound = torch.tensor(
        [[TRAIN_DIAMETER] * STATE_DIM], dtype=torch.float32
    ).to(device)

    x = (torch.zeros(N, STATE_DIM).uniform_(-1, 1) * bound.squeeze(0)).to(device)
    if certify_counter_examples:
        ce_t = torch.cat(certify_counter_examples, dim=0).to(device)
        x = torch.cat((x, ce_t), dim=0)
    else:
        ce_t = None

    x_all = x.clone()
    x_0 = torch.zeros([1, STATE_DIM]).to(device)

    for it in range(max_iters):
        # Lower bound on V via CROWN
        model_bound = BoundedModule(lyap_model, x_0)
        ptb = PerturbationLpNorm(norm=np.inf, x_L=-bound, x_U=bound)
        state_range = BoundedTensor(x_0, ptb)
        lyap_lb, _ = model_bound.compute_bounds(
            x=(state_range,), method="backward"
        )

        diff, _ = lyap_diff(policy_model, lyap_model, x)
        loss = F.relu(-lyap_lb).mean() + 1.2 * F.relu(diff + const_c).mean()

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping on policy (matching the paper)
        for param in policy_model.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        if it % 10 == 0:
            ce = find_counterexamples(
                policy_model, lyap_model, device, bounds=bound.squeeze(0)
            )
            if len(ce) > 0:
                __logger__.debug("Iter %d: %d counterexamples", it, ce.size(0))

            idx = np.random.choice(x_all.size(0), size=min(x_all.size(0), 512))
            x = torch.cat((ce, x_all[idx]), dim=0) if len(ce) else x_all[idx]
            if ce_t is not None:
                x = torch.cat((x, ce_t), dim=0)
            x_all = torch.cat((ce, x_all), dim=0) if len(ce) else x_all

        if it % 10 == 0 and loss.item() == 0:
            break

    __logger__.info("Training done (%d iters, loss=%.6f)", it + 1, loss.item())

    # --- Certification ---------------------------------------------------
    verifier = LyapunovVerifier(device=device)
    all_verified, failed_regions, new_ce = verifier.certify_all_regions(
        closed_loop_model, lyap_model, certify_list
    )

    ce_diffs: list[float] = []
    if len(new_ce) > 0:
        with torch.no_grad():
            for i in range(new_ce.size(0)):
                d, _ = lyap_diff(
                    policy_model, lyap_model, new_ce[i : i + 1].to(device)
                )
                ce_diffs.append(d.item())
                certify_counter_examples.append(new_ce[i : i + 1])

    return certify_counter_examples, failed_regions, all_verified, ce_diffs


# ---------------------------------------------------------------------------
# Outer training loop (with LR decay + re-certification)
# ---------------------------------------------------------------------------
def train_main(
    policy_model: torch.nn.Module,
    lyap_model: torch.nn.Module,
    closed_loop_model: torch.nn.Module,
    certify_list: Sequence[list[float]],
    certify_counter_examples: list[torch.Tensor],
    device: str,
    const_c: float = CONST_C_DEFAULT,
) -> int:
    """
    Outer loop: train, certify, decay LR, repeat until success.

    Returns the number of training rounds needed.
    """
    num_trains = 0
    success = False

    while not success:
        num_trains += 1
        lr = 0.00625
        __logger__.info("=== Round %d  CONST_C=%.4f ===", num_trains, const_c)

        (certify_counter_examples, failed_regions,
         success, ce_diffs) = lyap_train_main(
            policy_model, lyap_model, closed_loop_model,
            certify_counter_examples, certify_list, lr, device, const_c,
        )

        if ce_diffs:
            const_c = max(0.12, -min(ce_diffs) + 0.01)
        else:
            const_c = 0.12

        if success:
            return num_trains

        # LR-decay retries on the failed sub-regions
        for retry in range(10):
            lr *= 0.9
            __logger__.info("  Retry %d/10  lr=%.6f  CONST_C=%.4f",
                         retry + 1, lr, const_c)
            (certify_counter_examples, _, success,
             ce_diffs) = lyap_train_main(
                policy_model, lyap_model, closed_loop_model,
                certify_counter_examples, failed_regions, lr, device, const_c,
            )
            if ce_diffs:
                const_c = max(0.12, -min(ce_diffs) + 0.01)
            else:
                const_c = 0.12
            if success:
                break

        if success:
            # Re-certify on the *full* region list
            verifier = LyapunovVerifier(device=device)
            success, _, _ = verifier.certify_all_regions(
                closed_loop_model, lyap_model, certify_list
            )

    return num_trains
