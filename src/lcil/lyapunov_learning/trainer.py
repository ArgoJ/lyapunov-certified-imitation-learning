from __future__ import annotations

import os
import time
import logging
import numpy as np
import torch as th
import torch.nn as nn

from numpy.typing import NDArray
from pathlib import Path
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from .config import LyapunovTrainingConfig
from .buffer import DynamicStateBuffer
from .loss import (
    LyapunovTrainingLoss,
)
from .counterexample import (
    BoundaryRhoDiagnostics,
    estimate_rho_from_boundary_diagnostics,
    find_counter_examples,
    sample_uniform_box,
)
from ..utils.base_models import save_model_checkpoint
from ..utils.helpers import none_to_float

__logger__ = logging.getLogger(__name__)


@dataclass
class LyapunovTrainingResult:
    rho_estimate: float
    num_mined_counterexamples: int
    train_time: float
    lyap_model_path: os.PathLike | None = None
    policy_model_path: os.PathLike | None = None


@dataclass
class LyapunovTrainingMetrics:
    """Per-outer-iteration training metrics for Lyapunov optimization."""

    loss: NDArray
    rho_estimate: NDArray
    rho_boundary_quantile: NDArray
    rho_boundary_mean: NDArray
    rho_feature_term_quantile: NDArray
    rho_linear_term_quantile: NDArray
    rho_feature_term_mean: NDArray
    rho_linear_term_mean: NDArray
    rho_feature_term_mean_share: NDArray
    rho_linear_term_mean_share: NDArray
    r_factor_fro_norm: NDArray
    buffer_size: NDArray
    num_mined_counterexamples: NDArray
    outer_iterations_completed: int = 0

    @classmethod
    def from_num_outer_epochs(cls, num_outer_epochs: int) -> LyapunovTrainingMetrics:
        """Create NaN-initialized metric arrays for a fixed outer-epoch budget."""
        if num_outer_epochs <= 0:
            raise ValueError("num_outer_epochs must be positive.")

        nan_array = np.full((num_outer_epochs,), np.nan, dtype=np.float64)
        return cls(
            loss=nan_array.copy(),
            rho_estimate=nan_array.copy(),
            rho_boundary_quantile=nan_array.copy(),
            rho_boundary_mean=nan_array.copy(),
            rho_feature_term_quantile=nan_array.copy(),
            rho_linear_term_quantile=nan_array.copy(),
            rho_feature_term_mean=nan_array.copy(),
            rho_linear_term_mean=nan_array.copy(),
            rho_feature_term_mean_share=nan_array.copy(),
            rho_linear_term_mean_share=nan_array.copy(),
            r_factor_fro_norm=nan_array.copy(),
            buffer_size=nan_array.copy(),
            num_mined_counterexamples=nan_array.copy(),
            outer_iterations_completed=0,
        )

    def fill_outer(
        self,
        outer_iter: int,
        loss_value: float,
        state_buffer: list,
        num_mined_counterexamples: int,
        rho_diagnostics: BoundaryRhoDiagnostics,
    ):
        self.loss[outer_iter] = loss_value
        self.buffer_size[outer_iter] = float(len(state_buffer))
        self.num_mined_counterexamples[outer_iter] = float(num_mined_counterexamples)
        self.outer_iterations_completed = outer_iter + 1

        self.rho_estimate[outer_iter] = rho_diagnostics.rho
        self.rho_boundary_quantile[outer_iter] = rho_diagnostics.boundary_quantile
        self.rho_boundary_mean[outer_iter] = rho_diagnostics.boundary_mean
        self.rho_feature_term_quantile[outer_iter] = rho_diagnostics.feature_term_quantile
        self.rho_linear_term_quantile[outer_iter] = rho_diagnostics.linear_term_quantile
        self.rho_feature_term_mean[outer_iter] = rho_diagnostics.feature_term_mean
        self.rho_linear_term_mean[outer_iter] = rho_diagnostics.linear_term_mean
        self.rho_feature_term_mean_share[outer_iter] = rho_diagnostics.feature_term_mean_share
        self.rho_linear_term_mean_share[outer_iter] = rho_diagnostics.linear_term_mean_share
        self.r_factor_fro_norm[outer_iter] = rho_diagnostics.r_factor_fro_norm

    def save(self, path: os.PathLike) -> None:
        metrics_path = Path(path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            metrics_path,
            loss=self.loss,
            rho_estimate=self.rho_estimate,
            rho_boundary_quantile=self.rho_boundary_quantile,
            rho_boundary_mean=self.rho_boundary_mean,
            rho_feature_term_quantile=self.rho_feature_term_quantile,
            rho_linear_term_quantile=self.rho_linear_term_quantile,
            rho_feature_term_mean=self.rho_feature_term_mean,
            rho_linear_term_mean=self.rho_linear_term_mean,
            rho_feature_term_mean_share=self.rho_feature_term_mean_share,
            rho_linear_term_mean_share=self.rho_linear_term_mean_share,
            r_factor_fro_norm=self.r_factor_fro_norm,
            buffer_size=self.buffer_size,
            num_mined_counterexamples=self.num_mined_counterexamples,
            outer_iterations_completed=np.asarray(self.outer_iterations_completed, dtype=np.int64),
        )
def _tb_writer_add_metrics(tb_writer: SummaryWriter, metrics: LyapunovTrainingMetrics) -> None:
    outer_iter = metrics.outer_iterations_completed - 1
    tb_writer.add_scalar("Lyapunov/Loss", metrics.loss[outer_iter], outer_iter)
    tb_writer.add_scalar("Lyapunov/Rho", metrics.rho_estimate[outer_iter], outer_iter)
    tb_writer.add_scalar("Lyapunov/RhoBoundaryQuantile", metrics.rho_boundary_quantile[outer_iter], outer_iter)
    tb_writer.add_scalar("Lyapunov/RhoBoundaryMean", metrics.rho_boundary_mean[outer_iter], outer_iter)
    tb_writer.add_scalar("Lyapunov/RhoFeatureTermQuantile", metrics.rho_feature_term_quantile[outer_iter], outer_iter)
    tb_writer.add_scalar("Lyapunov/RhoLinearTermQuantile", metrics.rho_linear_term_quantile[outer_iter], outer_iter)
    tb_writer.add_scalar("Lyapunov/RhoFeatureTermMeanShare", metrics.rho_feature_term_mean_share[outer_iter], outer_iter)
    tb_writer.add_scalar("Lyapunov/RhoLinearTermMeanShare", metrics.rho_linear_term_mean_share[outer_iter], outer_iter)
    tb_writer.add_scalar("Lyapunov/RFactorFroNorm", metrics.r_factor_fro_norm[outer_iter], outer_iter)
    tb_writer.add_scalar(
        "Lyapunov/NumMinedCounterexamples",
        metrics.num_mined_counterexamples[outer_iter],
        outer_iter,
    )


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
        self.config = config
        self.device = th.device(device)
        if self.config.seed is not None:
            th.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)

        self.policy_model = policy_model.to(self.device)
        self.lyap_model = lyap_model.to(self.device)
        self.dyn_model = dyn_model.to(self.device)
        self._set_train_modes()

        self.optimizer = th.optim.Adam(
            self._get_train_params(), 
            self.config.learning_rate
        )
        self.loss_module = LyapunovTrainingLoss(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            config=self.config,
            device=self.device,
        )
        self.lbx = self.loss_module.lbx
        self.ubx = self.loss_module.ubx
        self.center = self.loss_module.center

        self.results: LyapunovTrainingResult | None = None
        self.metrics: LyapunovTrainingMetrics | None = None

    def _set_train_modes(self) -> None:
        for param in self.lyap_model.parameters():
            param.requires_grad_(True)
        self.lyap_model.train()

        for param in self.dyn_model.parameters():
            param.requires_grad_(False)
        self.dyn_model.eval()

        if not self.config.train_policy_model:
            for param in self.policy_model.parameters():
                param.requires_grad_(False)
            self.policy_model.eval()
        else:
            for param in self.policy_model.parameters():
                param.requires_grad_(True)
            self.policy_model.train()

    def _get_train_params(self) -> tuple[nn.Parameter, ...]:
        """Utility to gather trainable parameters based on the config."""
        return tuple(
            param
            for model in (self.lyap_model, self.policy_model)
            for param in model.parameters()
            if param.requires_grad
        )

    def _mine_new_counterexamples(self, rho_estimate: float) -> th.Tensor:
        """Mine rho-gated counterexamples using the external training semantics."""
        return find_counter_examples(
            objective=lambda x: self.loss_module.mining_objective(
                x_batch=x,
                rho_estimate=rho_estimate,
            ),
            config=self.config,
            device=self.device,
        )

    def _build_roa_candidates(self) -> th.Tensor:
        """Create diverse candidate states near the boundary of the asymmetric B."""
        directions = th.randn(self.config.roa_candidate_size, self.config.state_dim, device=self.device)
        directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
        radii = th.rand(self.config.roa_candidate_size, 1, device=self.device) * 0.4 + 0.6
        z_candidates = directions * radii  # between 0.6 and 1.0 in random directions
        half_width = 0.5 * (self.ubx - self.lbx)
        return z_candidates * half_width + self.center

    def train(self) -> LyapunovTrainingResult:
        """Execute the CEGIS-style training loop."""
        metrics = LyapunovTrainingMetrics.from_num_outer_epochs(self.config.outer_epochs)
        
        # Initial training pool sampled uniformly from the state space bounds
        initial_x = sample_uniform_box(self.config.initial_sample_size, self.lbx, self.ubx, self.device)
        state_buffer = DynamicStateBuffer(initial_states=initial_x, max_size=self.config.max_buffer, device=self.device)
        roa_candidates = self._build_roa_candidates()

        mining_interval = max(1, int(self.config.counterexample_every))
        rho_estimate = self.config.rho_min
        num_mined_counterexamples = 0
        total_steps = self.config.outer_epochs * self.config.steps_per_epoch

        tb_writer = SummaryWriter(log_dir=self.config.tb_log_dir) \
            if SummaryWriter is not None and self.config.tb_log_dir is not None else None

        start_time = time.time()
        with Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss: {task.fields[loss]:.4f}"),
            TextColumn("ρ: {task.fields[rho]:.4f}"),
            TextColumn("pool: {task.fields[pool]:.0f}"),
            TextColumn("cex: {task.fields[cex]:.0f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "Lyapunov Training Iterations",
                total=float(total_steps),
                loss=float("nan"),
                rho=float(rho_estimate),
                pool=float(len(state_buffer)),
                cex=float(num_mined_counterexamples),
            )
            for outer_iter in range(self.config.outer_epochs):
                last_loss_value = np.nan
                
                # Estimate current Region of Attraction
                rho_diagnostics = estimate_rho_from_boundary_diagnostics(
                    lyap_model=self.lyap_model,
                    config=self.config,
                    device=self.device,
                )
                rho_estimate = rho_diagnostics.rho # TODO: Consider smoothing or just constant

                # Mine counterexamples (CEGIS)
                if (outer_iter + 1) % mining_interval == 0:
                    new_cex = self._mine_new_counterexamples(rho_estimate=rho_estimate)
                    num_mined_counterexamples += new_cex.shape[0]
                    state_buffer.register_cex(new_cex)
                    roa_candidates = self._build_roa_candidates()

                # Inner training loop
                for _ in range(self.config.steps_per_epoch):
                    x_batch = state_buffer.sample(self.config.batch_size)
                    loss = self.loss_module(
                        x_batch=x_batch,
                        roa_candidates=roa_candidates,
                        rho_estimate=rho_estimate,
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Update Progress Bar
                    progress.update(
                        task,
                        advance=1.0,
                        loss=none_to_float(loss.item()),
                        rho=none_to_float(rho_estimate),
                        pool=none_to_float(len(state_buffer)),
                        cex=none_to_float(num_mined_counterexamples),
                    )
                    last_loss_value = float(loss.item())

                metrics.fill_outer(
                    outer_iter=outer_iter,
                    loss_value=last_loss_value,
                    state_buffer=state_buffer,
                    num_mined_counterexamples=num_mined_counterexamples,
                    rho_diagnostics=rho_diagnostics,
                )

                if tb_writer is not None:
                    _tb_writer_add_metrics(tb_writer, metrics)

        train_time = time.time() - start_time
        __logger__.debug("Lyapunov training finished in %.2fs", train_time)

        self.results = LyapunovTrainingResult(
            rho_estimate=rho_estimate,
            num_mined_counterexamples=num_mined_counterexamples,
            train_time=train_time,
        )
        self.metrics = metrics

        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

        return self.results

    def save(
        self,
        save_folder: os.PathLike,
    ) -> None:
        """Utility function to save training results and model checkpoints.

        Parameters
        ----------
        save_folder : PathLike[str]
            Folder where the models and config should be saved.
        """
        save_folder = Path(save_folder).resolve()
        save_folder.mkdir(parents=True, exist_ok=True)

        # Config saving
        config_path = save_folder / "training_config.json"
        self.config.save(config_path)

        # Models saving 
        lyap_model_path = save_folder / f"lyapunov_model.pt"
        policy_model_path = save_folder / f"policy_model.pt"

        save_model_checkpoint(self.lyap_model, lyap_model_path)
        save_model_checkpoint(self.policy_model, policy_model_path)

        # Metric saving
        if self.metrics is not None:
            metrics_path = save_folder / "training_metrics.npz"
            self.metrics.save(metrics_path)

        # Optional: Update the results object with the paths if it exists
        if self.results is not None:
            self.results.lyap_model_path = str(lyap_model_path)
            self.results.policy_model_path = str(policy_model_path)
        
        __logger__.info(f"Saved lyapunov results to {save_folder}")
