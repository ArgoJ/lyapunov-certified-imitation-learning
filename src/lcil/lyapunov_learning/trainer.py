from __future__ import annotations

import os
import time
import numpy as np
import torch as th
import torch.nn as nn

from numpy.typing import NDArray
from pathlib import Path
from dataclasses import dataclass
from pkg_logger import get_package_logger

from .config import LyapunovTrainingConfig
from .buffer import DynamicStateBuffer
from .counterexample import (
    estimate_rho_from_boundary,
    find_counter_examples,
    sample_uniform_box,
)
from ..certification.models import ClosedLoopLyapunovConditionVerifier
from ..utils.base_models import save_model_checkpoint

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

class LyapunovTrainer:
    """Trainer class for Lyapunov-stable neural controllers utilizing a CEGIS-style loop."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovTrainingConfig,
        device: th.device | str = "cpu"
    ) -> None:
        self.config = self._resolve_config(config)
        self.device = th.device(device)

        self.policy_model = policy_model.to(self.device)
        self.lyap_model = lyap_model.to(self.device)
        self.dyn_model = dyn_model.to(self.device)

        self.origin = th.zeros(1, self.config.state_dim, dtype=th.float32, device=self.device)
        self.lbx, self.ubx = self._resolve_bounds(self.config.state_bounds, self.device)

        self.optimizer: th.optim.Optimizer | None = None
        self.verifier: ClosedLoopLyapunovConditionVerifier | None = None
        self.trainable_params: list[nn.Parameter] = []
        self.results: LyapunovTrainingResult | None = None

    @staticmethod
    def _resolve_config(config: LyapunovTrainingConfig) -> LyapunovTrainingConfig:
        """Utility to resolve and validate the training configuration."""
        if config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if config.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if config.roa_candidate_size <= 0:
            raise ValueError("ROA candidate size must be positive.")
        if config.outer_epochs <= 0:
            raise ValueError("Outer epochs must be positive.")
        if config.steps_per_epoch <= 0:
            raise ValueError("Steps per epoch must be positive.")
        if config.counterexample_every < 0:
            raise ValueError("Counterexample mining interval must be non-negative.")
        if config.rho_min <= 0:
            raise ValueError("Minimum rho estimate must be positive.")
        if len(config.state_bounds) != config.state_dim:
            raise ValueError(
                "state_bounds must match state_dim. "
                f"Expected {config.state_dim}, got {len(config.state_bounds)}"
            )
        if config.seed is not None:
            th.manual_seed(config.seed)
            np.random.seed(config.seed)
        return config

    @staticmethod
    def _resolve_bounds(state_bounds: NDArray, device: th.device) -> tuple[th.Tensor, th.Tensor]:
        """Utility to convert state bounds from config into tensors."""
        lbx = th.tensor(state_bounds[0], dtype=th.float32, device=device)
        ubx = th.tensor(state_bounds[1], dtype=th.float32, device=device)
        
        if (lbx >= 0).any() or (ubx <= 0).any() or (lbx > ubx).any():
            raise ValueError(
                "State bounds do not appear to include the origin. " 
                "Ensure that state_bounds are correctly specified for Lyapunov training."
            )

        return lbx, ubx

    def _set_adam_optimizer(self) -> None:
        """Configure the trainer to use the Adam optimizer based on the config."""
        self.lyap_model.train()
        self.dyn_model.eval()

        if not self.config.train_policy_model:
            self.policy_model.eval()
            policy_params = []
            __logger__.debug("Policy model updates disabled; training only Lyapunov model parameters.")
        else:
            self.policy_model.train()
            policy_params = list(self.policy_model.parameters())
            __logger__.debug("Training both policy and Lyapunov model parameters.")

        lyap_params = list(self.lyap_model.parameters())
        self.trainable_params = policy_params + lyap_params

        if not self.trainable_params:
            raise ValueError("No trainable parameters found. Check model definitions and config.")

        self.optimizer = th.optim.Adam(self.trainable_params, lr=self.config.learning_rate)

    def _set_verifier(self) -> None:
        """Utility to create and configure the Lyapunov condition verifier."""
        self.verifier = ClosedLoopLyapunovConditionVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=self.lbx,
            ubx=self.ubx,
            invariance_weight=self.config.invariance_weight,
        ).to(self.device)

    def _to_tensor(self, value: float) -> th.Tensor:
        """Utility to convert scalar values to tensors on the correct device."""
        return th.tensor([value], dtype=th.float32, device=self.device)

    def _loss_fn(self, x_batch: th.Tensor, roa_candidates: th.Tensor, rho_estimate: float, current_kappa: float) -> th.Tensor:
        """Compute the combined loss for a training batch."""
        if self.verifier is None:
            raise ValueError("Verifier not initialized. Call _set_verifier() before computing loss.")
        
        rho_t = self._to_tensor(rho_estimate)
        kappa_t = self._to_tensor(current_kappa)

        # L_Vdot = ReLU(verifier(x, rho, kappa))
        loss_condition = th.relu(self.verifier(x_batch, rho_t, kappa_t)).mean()

        # L_roa = ReLU(V(x) / rho - 1)
        v_candidates = self.lyap_model(roa_candidates)
        loss_roa = th.relu(v_candidates / max(rho_estimate, self.config.rho_min) - 1.0).mean()

        # Keep V(0) near zero for generic Lyapunov parameterizations.
        loss_origin = self.lyap_model(self.origin).pow(2).mean()
        loss_l1 = _parameter_l1_norm(self.trainable_params)

        total_loss = (
            loss_condition
            + self.config.roa_weight * loss_roa
            + self.config.l1_weight * loss_l1
            + self.config.pos_scale * loss_origin
        )

        return total_loss

    def _mine_new_counterexamples(self, rho_estimate: float, current_kappa: float) -> th.Tensor:
        """Mine new counterexamples using the verifier and add them to the training buffer."""
        if self.verifier is None:
            raise ValueError("Verifier not initialized. Call _set_verifier() before mining counterexamples.")
        
        new_cex = find_counter_examples(
            verifier=self.verifier,
            config=self.config,
            rho=rho_estimate,
            kappa=current_kappa,
            device=self.device,
        )
        
        rho_t = self._to_tensor(rho_estimate)
        kappa_t = self._to_tensor(current_kappa)
        
        new_kappa = current_kappa
        if new_cex.numel() > 0:
            # Margin dynamically adjusted based on the worst violation in the new counterexamples
            with th.no_grad():
                cex_diffs = self.verifier(new_cex, rho_t, kappa_t).detach().cpu().numpy()
            new_kappa = max(self.config.kappa, -np.min(cex_diffs) + 0.01)
        
        return new_cex, new_kappa

    def _build_roa_candidates(self) -> th.Tensor:
        """Create diverse candidate states near the boundary of the asymmetric B."""
        directions = th.randn(self.config.roa_candidate_size, self.config.state_dim, device=self.device)
        directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
        radii = th.rand(self.config.roa_candidate_size, 1, device=self.device) * 0.4 + 0.6
        z_candidates = directions * radii  # between 0.6 and 1.0 in random directions
        half_width = (self.ubx - self.lbx) / 2.0
        center = (self.ubx + self.lbx) / 2.0
        return z_candidates * half_width + center

    def train(self) -> LyapunovTrainingResult:
        """Execute the CEGIS-style training loop."""
        self._set_adam_optimizer()
        self._set_verifier()

        roa_candidates = self._build_roa_candidates()
        
        # Initial training pool sampled uniformly from the state space bounds
        initial_x = sample_uniform_box(self.config.initial_sample_size, self.lbx, self.ubx, self.device)
        state_buffer = DynamicStateBuffer(initial_states=initial_x, max_size=self.config.max_buffer)

        mining_interval = max(1, self.config.counterexample_every // max(1, self.config.steps_per_epoch))
        rho_estimate = self.config.rho_min
        current_kappa = self.config.kappa
        num_mined_counterexamples = 0
        total_steps = self.config.outer_epochs * self.config.steps_per_epoch
        
        start_time = time.time()
        with __logger__.tqdm(
            total=total_steps, 
            desc="Lyapunov Training Iterations", 
            unit="step",
            suppress_native_output=True,
        ) as pbar:
            for outer_iter in range(self.config.outer_epochs):
                
                # Estimate current Region of Attraction
                rho_estimate = estimate_rho_from_boundary(
                    lyap_model=self.lyap_model,
                    config=self.config,
                    device=self.device,
                )

                # Mine counterexamples (CEGIS)
                if (outer_iter + 1) % mining_interval == 0:
                    new_cex, current_kappa = self._mine_new_counterexamples(rho_estimate, current_kappa)
                    num_mined_counterexamples += new_cex.shape[0]
                    state_buffer.add(new_cex)

                # Inner training loop
                for _ in range(self.config.steps_per_epoch):
                    x_batch = state_buffer.sample(self.config.batch_size)
                    loss = self._loss_fn(x_batch, roa_candidates, rho_estimate, current_kappa)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Update Progress Bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "Loss": f"{loss.item():.4f}",
                        "Rho": f"{rho_estimate:.4f}",
                        "Pool": int(state_buffer.__len__()),
                    })

        train_time = time.time() - start_time
        __logger__.debug("Lyapunov training finished in %.2fs", train_time)

        self.results = LyapunovTrainingResult(
            rho_estimate=rho_estimate,
            num_mined_counterexamples=num_mined_counterexamples,
            train_time=train_time,
        )

        return self.results

    def save(
        self,
        save_folder: os.PathLike[str],
    ) -> None:
        """Utility function to save training results and model checkpoints.

        Parameters
        ----------
        save_folder : PathLike[str]
            Folder where the models and config should be saved.
        """
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        lyap_model_path = save_folder / f"lyapunov_model.pt"
        policy_model_path = save_folder / f"policy_model.pt"

        save_model_checkpoint(self.lyap_model, lyap_model_path)
        save_model_checkpoint(self.policy_model, policy_model_path)

        # Optional: Update the results object with the paths if it exists
        if self.results is not None:
            self.results.lyap_model_path = str(lyap_model_path)
            self.results.policy_model_path = str(policy_model_path)
        
        __logger__.info("Saved lyapunov results to %s", save_folder)
