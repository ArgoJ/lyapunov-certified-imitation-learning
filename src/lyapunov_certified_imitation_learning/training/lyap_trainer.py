import logging
import time
from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .lyapunov_config import LyapunovTrainingConfig
from ..utils.package_logger import DEFAULT_MODULE_NAME, PackageLogger
from ..verification.counterexample import find_counter_examples, lyap_diff_calculation
from ..verification.abcrown_wrapper import certify_list_all

__logger__ = PackageLogger.get_logger(__name__)


def train_lyapunov(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
    output_prefix: str = "lyap_crown",
    results_path: str = "lyap_crown_result.txt",
) -> dict[str, Any]:
    """Train a Lyapunov network and optionally certify it.

    Parameters
    ----------
    policy_model : nn.Module
        Policy network mapping state to action.
    lyap_model : nn.Module
        Lyapunov function approximator.
    dyn_model : nn.Module
        Dynamics model mapping (state, action) to next state.
    config : LyapunovTrainingConfig
        Training and certification configuration.
    device : th.device
        Torch device.
    output_prefix : str
        Prefix for checkpoint filename.
    results_path : str
        Path to append certification results.
    """
    if len(config.state_bounds) != config.state_dim:
        raise ValueError("state_bounds must match state_dim.")

    if config.seed is not None:
        th.manual_seed(config.seed)
        np.random.seed(config.seed)

    optimizer = th.optim.Adam(
        list(policy_model.parameters()) + list(lyap_model.parameters()),
        lr=config.learning_rate,
    )

    bounds = th.tensor(config.state_bounds, device=device)
    x_all = th.zeros(config.sample_size, config.state_dim).uniform_(-1, 1).to(device)
    x_all = x_all * bounds

    start_time = time.time()
    tqdm_handler = None
    restored_handlers = []
    tqdm_handler, restored_handlers = PackageLogger.add_tqdm_handler()
    pbar = tqdm(range(config.outer_epochs), desc="Epochs", unit="epoch")
    try:
        for epoch in pbar:
            running_loss = 0.0
            for step in range(config.steps_per_epoch):
                batch_size = min(x_all.size(0), config.batch_size)
                batch_indices = np.random.choice(x_all.size(0), batch_size, replace=False)
                x_batch = x_all[batch_indices]

                dv, reg = lyap_diff_calculation(
                    policy_model,
                    lyap_model,
                    dyn_model,
                    x_batch,
                    reg_clamp_max=config.reg_clamp_max,
                )
                v_val = lyap_model(x_batch)
                v_0 = lyap_model(th.zeros(1, config.state_dim, device=device))

                loss_stab = F.relu(dv + config.reg_scale * reg).mean()
                loss_pos = F.relu(
                    -v_val + config.pos_scale * th.norm(x_batch, dim=1, keepdim=True)
                ).mean()
                loss_origin = v_0.pow(2).sum()

                loss = loss_stab + loss_pos + loss_origin

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (step + 1) % config.counterexample_every == 0 or step == 0:
                    avg_loss = running_loss / (step + 1)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                if step % config.counterexample_every == 0:
                    new_cex = find_counter_examples(
                        policy_model,
                        lyap_model,
                        dyn_model,
                        config,
                        device=device,
                    )
                    if len(new_cex) > 0:
                        x_all = th.cat((x_all, new_cex), dim=0)
                        if x_all.size(0) > config.max_buffer:
                            x_all = x_all[-config.max_buffer :]
    finally:
        if tqdm_handler:
            PackageLogger.restore_handlers(DEFAULT_MODULE_NAME, tqdm_handler, restored_handlers)

    train_time = time.time() - start_time
    __logger__.info("Training finished in %.2fs", train_time)

    results: dict[str, Any] = {
        "counter_examples": [],
        "failed_regions": [],
        "success": None,
    }

    if config.run_certification:
        __logger__.info("Starting Certification...")
        try:
            cex_list, failed_regions, certified_regions, success = certify_list_all(
                policy_model,
                lyap_model,
                dyn_model,
                config,
                device=device,
            )
            results["counter_examples"] = cex_list
            results["failed_regions"] = failed_regions
            results["certified_regions"] = certified_regions
            results["success"] = success
            __logger__.info("Certification Success: %s", success)
            __logger__.info("Number of uncertified sub-regions: %d", len(failed_regions))
        except Exception as exc:
            __logger__.warning("Certification failed: %s", exc)
    else:
        __logger__.info("Certification skipped by config.")

    seed_suffix = f"_{config.seed}" if config.seed is not None else ""
    th.save(
        {"model_state_dict": lyap_model.state_dict()},
        f"{output_prefix}{seed_suffix}.pt",
    )

    if results["success"] is not None:
        status_msg = "rigorously certified" if results["success"] else "not certified"
        __logger__.info("Model is %s.", status_msg)

    if results_path and results["success"] is not None:
        seed_value = config.seed if config.seed is not None else -1
        with open(results_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"Seed: {seed_value}, Success: {results['success']}, "
                f"Failed Regions: {len(results['failed_regions'])}, "
                f"Certified Regions: {len(results['certified_regions'])}\n"
            )

    return results