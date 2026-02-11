import os
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

# Hyperparameter from nlc_discrete (inverted_pendulum.py)
# These control the strictness of the constraints.
# V(x_{k+1}) - V(x_k) <= -margin
STAB_MARGIN = 0.12 # Often tunable in the paper, here hardcoded from the example
REG_PARAM = 0.0001
LAMBDA_STAB = 1.2
LAMBDA_V0 = 1.2


def train_lyapunov(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
    models_prefix: str = "lyap_crown",
    results_path: str = "lyap_crown_result.txt",
) -> dict[str, Any]:
    """
    Trainiert ein Lyapunov-Netzwerk basierend auf der Methodik von Wu et al. (nlc_discrete),
    nutzt aber alpha-beta-CROWN für die Zertifizierung.
    """
    if len(config.state_bounds) != config.state_dim:
        raise ValueError("state_bounds must match state_dim.")

    if config.seed is not None:
        th.manual_seed(config.seed)
        np.random.seed(config.seed)

    # Define Optimizer
    optimizer = th.optim.Adam(
        list(policy_model.parameters()) + list(lyap_model.parameters()),
        lr=config.learning_rate,
    )

    # Bounds and initial data
    bounds = th.tensor(config.state_bounds, device=device)
    
    # x_all serves as a pool of random samples (Uniform Distribution)
    # Corresponds to 'x' in inverted_pendulum.py before the loop
    x_random_pool = th.zeros(config.sample_size, config.state_dim).uniform_(-1, 1).to(device)
    x_random_pool = x_random_pool * bounds

    # Buffer for found counterexamples (Falsification buffer)
    x_counterexamples = th.tensor([], device=device)

    start_time = time.time()
    
    # Logging Setup
    tqdm_handler = None
    restored_handlers = []
    tqdm_handler, restored_handlers = PackageLogger.add_tqdm_handler()
    
    # Training loop (Iterations instead of epochs, similar to inverted_pendulum.py)
    # We use config.outer_epochs * config.steps_per_epoch as total_iters
    total_iters = config.outer_epochs * config.steps_per_epoch
    pbar = tqdm(range(total_iters), desc="Iterations", unit="step")

    try:
        iter_num = 0
        valid_num = 0 # Counts how often no counterexamples were found

        for iter_num in pbar:
            
            # --- Batch Construction ---
            # Mix counterexamples with random samples.
            n_batch = config.batch_size
            
            # Select random samples
            rand_indices = np.random.choice(x_random_pool.size(0), size=min(x_random_pool.size(0), n_batch), replace=False)
            x_batch = x_random_pool[rand_indices]

            # Add counterexamples (if any)
            if x_counterexamples.size(0) > 0:
                x_batch = th.cat((x_counterexamples, x_batch), dim=0)

            x_batch = x_batch.to(device)
            x_0 = th.zeros(1, config.state_dim, device=device)

            # --- Loss ---
            
            # Lyapunov Differenz berechnen: V(f(x)) - V(x)
            # lyap_diff_calculation should return (V_next - V_curr, regularization)
            lyap_value_diff, reg = lyap_diff_calculation(
                policy_model,
                lyap_model,
                dyn_model,
                x_batch,
                reg_clamp_max=config.reg_clamp_max, # z.B. 0.0005
            )

            v_val = lyap_model(x_batch)
            v_0 = lyap_model(x_0)
            
            # Positivity: V(x) - V(0) > 0  =>  Loss: ReLU(-(V(x) - V(0)) + epsilon)
            lyap_pos = v_val - v_0
            
            # Lyapunov_risk = (F.relu(-lyap_pos + 0.0001 * reg) + 1.2*F.relu(lyap_value_diff + 0.12)).mean()+ 1.2*(V0).pow(2)
            loss_positivity = F.relu(-lyap_pos + REG_PARAM * reg)
            loss_stability = LAMBDA_STAB * F.relu(lyap_value_diff + STAB_MARGIN)
            loss_origin = LAMBDA_V0 * (v_0).pow(2)

            loss = (loss_positivity + loss_stability).mean() + loss_origin

            # --- Optimization Step ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Falsification (Counter-Example Search) ---
            # Find all N steps
            if (iter_num + 1) % config.counterexample_every == 0:
                new_cex = find_counter_examples(
                    policy_model,
                    lyap_model,
                    dyn_model,
                    config,
                    device=device,
                )
                
                if len(new_cex) > 0:
                    x_counterexamples = th.cat((x_counterexamples, new_cex), dim=0)
                    if x_counterexamples.size(0) > config.max_buffer:
                        x_counterexamples = x_counterexamples[-config.max_buffer:]
                else:
                    valid_num += 1
                
                # Update Progress Bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "num_cex": f"{x_counterexamples.size(0)}"
                })

    except KeyboardInterrupt:
        __logger__.info("Training interrupted by user.")
        exit(0)
    finally:
        if tqdm_handler:
            PackageLogger.restore_handlers(DEFAULT_MODULE_NAME, tqdm_handler, restored_handlers)

    train_time = time.time() - start_time
    __logger__.info("Training finished in %.2fs", train_time)

    # --- Certification (alpha-beta-CROWN) ---
    results: dict[str, Any] = {
        "counter_examples": [],
        "failed_regions": [],
        "success": None,
    }

    if config.run_certification:
        __logger__.info("Starting Certification with alpha-beta-CROWN...")
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
            __logger__.warning("Certification failed or timed out: %s", exc)
            import traceback
            traceback.print_exc()
    else:
        __logger__.info("Certification skipped by config.")

    # Save models
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

    # Results Logging
    if results_path and results["success"] is not None:
        seed_value = config.seed if config.seed is not None else -1
        with open(results_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"Seed: {seed_value}, Success: {results['success']}, "
                f"Training Time: {train_time:.2f}, "
                f"Failed Regions: {len(results['failed_regions'])}\n"
            )

    return results