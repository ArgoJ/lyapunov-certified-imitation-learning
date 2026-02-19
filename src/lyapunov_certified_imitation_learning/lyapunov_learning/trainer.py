from __future__ import annotations

import os
import time
import numpy as np
import torch as th
import torch.nn as nn
from pathlib import Path

from dataclasses import dataclass

from .config import LyapunovTrainingConfig
from ..utils.package_logger import get_package_logger
from ..utils.base_models import save_model_checkpoint
from ..certification.models import ClosedLoopLyapunovConditionVerifier
from ..certification.counterexample import (
    estimate_rho_from_boundary,
    find_counter_examples,
    sample_uniform_box,
)

__logger__ = get_package_logger(__name__)


@dataclass
class LyapunovTrainingResult:
    rho_estimate: float
    num_mined_counterexamples: int
    train_time: float
    lyap_model_path: os.PathLike[str] | None = None
    policy_model_path: os.PathLike[str] | None = None


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
    models_folder: os.PathLike[str] | None = None,
) -> LyapunovTrainingResult:
    """Train a Lyapunov-stable neural controller with a CEGIS-style loop."""
    if len(config.state_bounds) != config.state_dim:
        raise ValueError("state_bounds must match state_dim.")

    if config.seed is not None:
        th.manual_seed(config.seed)
        np.random.seed(config.seed)

    policy_model.to(device).train()
    lyap_model.to(device).train()
    dyn_model.to(device)

    lyap_params = list(lyap_model.parameters())
    policy_params = list(policy_model.parameters()) if config.train_policy_model else []
    trainable_params = policy_params + lyap_params

    if not trainable_params:
        raise ValueError("No trainable parameters found. Check model definitions and config.")

    if not config.train_policy_model:
        policy_model.eval()
        __logger__.info("Policy model updates disabled; training only Lyapunov model parameters.")

    optimizer = th.optim.Adam(trainable_params, lr=config.learning_rate)

    bounds = th.tensor(config.state_bounds, dtype=th.float32, device=device)
    training_pool = sample_uniform_box(config.sample_size, bounds, device)
    roa_candidates = _build_roa_candidates(config.roa_candidate_size, bounds, device)
    origin = th.zeros(1, config.state_dim, dtype=th.float32, device=device)
    verifier = ClosedLoopLyapunovConditionVerifier(
        policy_model=policy_model,
        lyap_model=lyap_model,
        dyn_model=dyn_model,
        bounds=bounds,
        kappa=config.kappa,
        invariance_weight=config.invariance_weight,
        rho=config.rho_min,
    ).to(device)

    mining_interval = max(1, config.counterexample_every // max(1, config.steps_per_epoch))
    rho_estimate = config.rho_min
    num_mined_counterexamples = 0

    # TODO: füge CONST_C = max(0.12, -np.min(counter_example_diff_list) + 0.01) ein, 
    # damit die Bedingung nicht zu locker wird, wenn die ersten Counterexamples sehr schlecht sind.
    start_time = time.time()
    total_steps = config.outer_epochs * config.steps_per_epoch
    with __logger__.tqdm(total=total_steps, desc="Train iterations", unit="step") as pbar:
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

                # L_Vdot = ReLU(verifier(x))
                x_batch = training_pool[batch_idx]
                verifier.set_rho(rho_estimate)
                loss_condition = th.relu(verifier(x_batch)).mean()

                # L_roa = ReLU(V(x) / rho - 1)
                v_candidates = lyap_model(roa_candidates)
                loss_roa = th.relu(v_candidates / max(rho_estimate, config.rho_min) - 1.0).sum()

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

    results = LyapunovTrainingResult(
        rho_estimate=rho_estimate,
        num_mined_counterexamples=num_mined_counterexamples,
        train_time=train_time,
    )

    if models_folder is None:
        return results

    parent_dir = os.path.abspath(os.path.dirname(models_folder))
    os.makedirs(parent_dir, exist_ok=True)

    seed_suffix = f"_{config.seed}" if config.seed is not None else ""

    results.lyap_model_path = f"{models_folder}/lyapunov_model{seed_suffix}.pt"
    results.policy_model_path = f"{models_folder}/policy_model{seed_suffix}.pt"
    save_model_checkpoint(lyap_model, results.lyap_model_path)
    save_model_checkpoint(policy_model, results.policy_model_path)
    __logger__.info("Saved Lyapunov model to %s", results.lyap_model_path)
    __logger__.info("Saved policy model to %s", results.policy_model_path)
    
    if not config.train_policy_model:
        __logger__.info("Policy model was not trained; saved initial parameters.")

    return results