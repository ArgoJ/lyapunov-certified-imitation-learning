from __future__ import annotations

import os
import time
import numpy as np
import torch as th
import torch.nn as nn

from typing import Any
from tqdm import tqdm

from .lyapunov_config import LyapunovTrainingConfig
from ..utils.package_logger import DEFAULT_MODULE_NAME, PackageLogger
from ..verification.abcrown_wrapper import certify_rho_max
from ..verification.counterexample import (
    estimate_rho_from_boundary,
    find_counter_examples,
    lyapunov_condition_violation,
    sample_uniform_box,
)

__logger__ = PackageLogger.get_logger(__name__)


def _parameter_l1_norm(model_params: list[nn.Parameter]) -> th.Tensor:
    return th.stack([param.abs().sum() for param in model_params]).sum()


def _build_roa_candidates(
    sample_size: int,
    bounds: th.Tensor,
    device: th.device,
) -> th.Tensor:
    """Create diverse candidate states near the boundary of B."""
    state_dim = bounds.numel()
    directions = th.randn(sample_size, state_dim, device=device)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
    radii = th.rand(sample_size, 1, device=device) * 0.4 + 0.6
    return directions * radii * bounds.unsqueeze(0)


def train_lyapunov(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
    models_prefix: str = "lyap_crown",
    results_path: str = "lyap_crown_result.txt",
) -> dict[str, Any]:
    """Train a Lyapunov-stable neural controller with a CEGIS-style loop."""
    if len(config.state_bounds) != config.state_dim:
        raise ValueError("state_bounds must match state_dim.")

    if config.seed is not None:
        th.manual_seed(config.seed)
        np.random.seed(config.seed)

    optimizer = th.optim.Adam(
        list(policy_model.parameters()) + list(lyap_model.parameters()),
        lr=config.learning_rate,
    )

    bounds = th.tensor(config.state_bounds, dtype=th.float32, device=device)
    training_pool = sample_uniform_box(config.sample_size, bounds, device)
    roa_candidates = _build_roa_candidates(config.roa_candidate_size, bounds, device)
    origin = th.zeros(1, config.state_dim, dtype=th.float32, device=device)

    trainable_params = list(policy_model.parameters()) + list(lyap_model.parameters())

    mining_interval = max(1, config.counterexample_every // max(1, config.steps_per_epoch))
    rho_estimate = config.rho_min
    num_mined_counterexamples = 0

    start_time = time.time()
    total_steps = config.outer_epochs * config.steps_per_epoch
    with PackageLogger.tqdm_progress(total=total_steps, desc="Iterations", unit="step") as pbar:
        for outer_iter in range(config.outer_epochs):
            rho_estimate = estimate_rho_from_boundary(
                lyap_model=lyap_model,
                config=config,
                device=device,
            )

            if (outer_iter + 1) % mining_interval == 0:
                new_cex = find_counter_examples(
                    policy_model=policy_model,
                    lyap_model=lyap_model,
                    dyn_model=dyn_model,
                    config=config,
                    rho=rho_estimate,
                    device=device,
                )
                if new_cex.numel() > 0:
                    num_mined_counterexamples += new_cex.shape[0]
                    training_pool = th.cat((training_pool, new_cex), dim=0)
                    if training_pool.shape[0] > config.max_buffer:
                        keep_idx = th.randperm(training_pool.shape[0], device=device)[
                            : config.max_buffer
                        ]
                        training_pool = training_pool[keep_idx]

            for _ in range(config.steps_per_epoch):
                batch_size = min(config.batch_size, training_pool.shape[0])
                batch_idx = th.randint(
                    low=0,
                    high=training_pool.shape[0],
                    size=(batch_size,),
                    device=device,
                )
                x_batch = training_pool[batch_idx]
                violation = lyapunov_condition_violation(
                    policy_model=policy_model,
                    lyap_model=lyap_model,
                    dyn_model=dyn_model,
                    state=x_batch,
                    bounds=bounds,
                    kappa=config.kappa,
                    invariance_weight=config.invariance_weight,
                    rho=rho_estimate,
                )
                loss_condition = violation.mean()

                v_candidates = lyap_model(roa_candidates)
                loss_roa = th.relu(v_candidates / max(rho_estimate, config.rho_min) - 1.0).mean()

                # Keep V(0) near zero for generic Lyapunov parameterizations.
                loss_origin = lyap_model(origin).pow(2).mean()
                loss_l1 = _parameter_l1_norm(trainable_params)

                loss = (
                    loss_condition
                    + config.roa_weight * loss_roa
                    + config.l1_weight * loss_l1
                    + config.pos_scale * loss_origin
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "rho": f"{rho_estimate:.4f}",
                        "pool": int(training_pool.shape[0]),
                    }
                )

    train_time = time.time() - start_time
    __logger__.info("Training finished in %.2fs", train_time)

    results: dict[str, Any] = {
        "counter_examples": [],
        "failed_regions": [],
        "certified_regions": [],
        "success": None,
        "rho_estimate": rho_estimate,
        "rho_certified": None,
        "num_mined_counterexamples": num_mined_counterexamples,
    }

    if config.run_certification:
        __logger__.info(
            "Starting certification with alpha-beta-CROWN (rho estimate %.6f)...",
            rho_estimate,
        )
        try:
            rho_certified, cert_result = certify_rho_max(
                policy_model=policy_model,
                lyap_model=lyap_model,
                dyn_model=dyn_model,
                config=config,
                rho_estimate=rho_estimate,
                device=device,
            )
            results["counter_examples"] = cert_result.counter_examples
            results["failed_regions"] = cert_result.failed_regions
            results["certified_regions"] = cert_result.certified_regions
            results["success"] = cert_result.success
            results["rho_certified"] = rho_certified
            __logger__.info(
                "Certification success=%s, failed_regions=%d, rho_certified=%.6f",
                cert_result.success,
                len(cert_result.failed_regions),
                rho_certified,
            )
        except Exception as exc:
            __logger__.warning("Certification failed or timed out: %s", exc)
            import traceback

            traceback.print_exc()
    else:
        __logger__.info("Certification skipped by config.")

    parent_dir = os.path.abspath(os.path.dirname(models_prefix))
    os.makedirs(parent_dir, exist_ok=True)

    seed_suffix = f"_{config.seed}" if config.seed is not None else ""
    th.save(
        {"model_state_dict": lyap_model.state_dict()},
        f"{models_prefix}_lyap{seed_suffix}.pt",
    )
    th.save(
        {"model_state_dict": policy_model.state_dict()},
        f"{models_prefix}_policy{seed_suffix}.pt",
    )

    if results_path and results["success"] is not None:
        seed_value = config.seed if config.seed is not None else -1
        with open(results_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"Seed: {seed_value}, Success: {results['success']}, "
                f"Training Time: {train_time:.2f}, "
                f"Failed Regions: {len(results['failed_regions'])}, "
                f"Rho Estimate: {results['rho_estimate']:.6f}, "
                f"Rho Certified: {results['rho_certified']:.6f}\n"
            )

    return results
