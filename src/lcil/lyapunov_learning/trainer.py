from __future__ import annotations

import os
import time
import logging
import numpy as np
import torch as th
import torch.nn as nn

from collections.abc import Sequence
from typing import Literal
from numpy.typing import NDArray
from pathlib import Path
from dataclasses import dataclass, replace
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
from .buffer import BoundaryStateBuffer, DynamicStateBuffer
from .loss import (
    LyapunovTrainingLoss,
    LyapunovTrainingLossParts,
)
from .counterexample import (
    BoundaryRhoDiagnostics,
    estimate_rho_from_boundary_diagnostics,
    find_counter_examples,
    sample_uniform_box,
)
from .utils import ThresholdMonitor
from ..utils import (
    JsonDataclass,
    GracefulInterruptHandler,
    save_model_checkpoint,
    build_generator,
    none_to_float,
)
from ..utils.constants import *

__logger__ = logging.getLogger(__name__)


@dataclass
class LyapunovTrainingResult(JsonDataclass):
    rho_estimate: float
    num_mined_counterexamples: int
    train_time: float
    aborted: bool = False
    abort_reason: str | None = None
    lyap_model_path: os.PathLike | None = None
    policy_model_path: os.PathLike | None = None

    DEFAULT_FILE_NAME = TRAINING_RESULTS_FILENAME

    @property
    def completed(self) -> bool:
        return not self.aborted


@dataclass
class LyapunovTrainingCurriculumStage(JsonDataclass):
    stage_index: int
    state_bounds: NDArray
    scale: NDArray
    result: LyapunovTrainingResult

    NP_ARRAY_FIELDS = ("state_bounds", "scale")
    DEFAULT_FILE_NAME = TRAINING_CURRICULUM_STAGE_FILENAME


@dataclass
class LyapunovTrainingCurriculumResult(JsonDataclass):
    stages: list[LyapunovTrainingCurriculumStage]
    aborted_result: LyapunovTrainingResult | None = None
    aborted_stage_index: int | None = None

    DEFAULT_FILE_NAME = TRAINING_CURRICULUM_RESULT_FILENAME

    @property
    def aborted(self) -> bool:
        return self.aborted_result is not None

    @property
    def abort_reason(self) -> str | None:
        if self.aborted_result is None:
            return None
        return self.aborted_result.abort_reason

    @property
    def final_result(self) -> LyapunovTrainingResult | None:
        if self.aborted_result is not None:
            return self.aborted_result
        if not self.stages:
            return None
        return self.stages[-1].result

    @property
    def last_completed_result(self) -> LyapunovTrainingResult | None:
        if not self.stages:
            return None
        return self.stages[-1].result

    @property
    def final_state_bounds(self) -> NDArray | None:
        if not self.stages:
            return None
        return self.stages[-1].state_bounds


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
    inner_loss: NDArray
    inner_condition_raw: NDArray
    inner_roa_raw: NDArray
    inner_l1_raw: NDArray
    inner_equilibrium_raw: NDArray
    inner_formal_positivity_raw: NDArray
    inner_scale_raw: NDArray
    inner_policy_regularization_raw: NDArray
    inner_condition: NDArray
    inner_roa: NDArray
    inner_l1: NDArray
    inner_equilibrium: NDArray
    inner_formal_positivity: NDArray
    inner_scale: NDArray
    inner_policy_regularization: NDArray
    steps_per_epoch: int
    outer_iterations_completed: int = 0
    inner_iterations_completed: int = 0

    @classmethod
    def from_num_steps(
        cls,
        num_outer_epochs: int,
        steps_per_epoch: int,
    ) -> LyapunovTrainingMetrics:
        """Create NaN-initialized metric arrays for a fixed outer-epoch budget."""
        if num_outer_epochs <= 0:
            raise ValueError("num_outer_epochs must be positive.")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive.")

        outer_nan_array = np.full((num_outer_epochs,), np.nan, dtype=np.float64)
        inner_nan_array = np.full((num_outer_epochs * steps_per_epoch,), np.nan, dtype=np.float64)
        return cls(
            loss=outer_nan_array.copy(),
            rho_estimate=outer_nan_array.copy(),
            rho_boundary_quantile=outer_nan_array.copy(),
            rho_boundary_mean=outer_nan_array.copy(),
            rho_feature_term_quantile=outer_nan_array.copy(),
            rho_linear_term_quantile=outer_nan_array.copy(),
            rho_feature_term_mean=outer_nan_array.copy(),
            rho_linear_term_mean=outer_nan_array.copy(),
            rho_feature_term_mean_share=outer_nan_array.copy(),
            rho_linear_term_mean_share=outer_nan_array.copy(),
            r_factor_fro_norm=outer_nan_array.copy(),
            buffer_size=outer_nan_array.copy(),
            num_mined_counterexamples=outer_nan_array.copy(),
            inner_loss=inner_nan_array.copy(),
            inner_condition_raw=inner_nan_array.copy(),
            inner_roa_raw=inner_nan_array.copy(),
            inner_l1_raw=inner_nan_array.copy(),
            inner_equilibrium_raw=inner_nan_array.copy(),
            inner_formal_positivity_raw=inner_nan_array.copy(),
            inner_scale_raw=inner_nan_array.copy(),
            inner_policy_regularization_raw=inner_nan_array.copy(),
            inner_condition=inner_nan_array.copy(),
            inner_roa=inner_nan_array.copy(),
            inner_l1=inner_nan_array.copy(),
            inner_equilibrium=inner_nan_array.copy(),
            inner_formal_positivity=inner_nan_array.copy(),
            inner_scale=inner_nan_array.copy(),
            inner_policy_regularization=inner_nan_array.copy(),
            steps_per_epoch=steps_per_epoch,
            outer_iterations_completed=0,
            inner_iterations_completed=0,
        )

    def inner_tb_step(self, inner_iter: int) -> int:
        return inner_iter

    def outer_tb_step(self, outer_iter: int) -> int:
        return (outer_iter + 1) * self.steps_per_epoch - 1

    def fill_inner(
        self,
        inner_iter: int,
        loss_parts: LyapunovTrainingLossParts,
    ) -> None:
        self.inner_loss[inner_iter] = float(loss_parts.total.item())
        self.inner_condition_raw[inner_iter] = float(loss_parts.condition_raw.item())
        self.inner_roa_raw[inner_iter] = float(loss_parts.roa_raw.item())
        self.inner_l1_raw[inner_iter] = float(loss_parts.l1_raw.item())
        self.inner_equilibrium_raw[inner_iter] = float(loss_parts.equilibrium_raw.item())
        self.inner_formal_positivity_raw[inner_iter] = float(loss_parts.formal_positivity_raw.item())
        self.inner_scale_raw[inner_iter] = float(loss_parts.scale_raw.item())
        self.inner_policy_regularization_raw[inner_iter] = float(loss_parts.policy_regularization_raw.item())

        self.inner_condition[inner_iter] = float(loss_parts.condition.item())
        self.inner_roa[inner_iter] = float(loss_parts.roa.item())
        self.inner_l1[inner_iter] = float(loss_parts.l1.item())
        self.inner_equilibrium[inner_iter] = float(loss_parts.equilibrium.item())
        self.inner_formal_positivity[inner_iter] = float(loss_parts.formal_positivity.item())
        self.inner_scale[inner_iter] = float(loss_parts.scale.item())
        self.inner_policy_regularization[inner_iter] = float(loss_parts.policy_regularization.item())
        self.inner_iterations_completed = inner_iter + 1

    def fill_outer(
        self,
        outer_iter: int,
        loss_value: float,
        state_buffer: DynamicStateBuffer,
        num_mined_counterexamples: int,
        rho_diagnostics: BoundaryRhoDiagnostics,
    ) -> None:
        self.loss[outer_iter] = loss_value

        self.buffer_size[outer_iter] = float(state_buffer.state_count)
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
            inner_loss=self.inner_loss,
            inner_condition_raw=self.inner_condition_raw,
            inner_roa_raw=self.inner_roa_raw,
            inner_l1_raw=self.inner_l1_raw,
            inner_equilibrium_raw=self.inner_equilibrium_raw,
            inner_formal_positivity_raw=self.inner_formal_positivity_raw,
            inner_scale_raw=self.inner_scale_raw,
            inner_policy_regularization_raw=self.inner_policy_regularization_raw,
            inner_condition=self.inner_condition,
            inner_roa=self.inner_roa,
            inner_l1=self.inner_l1,
            inner_equilibrium=self.inner_equilibrium,
            inner_formal_positivity=self.inner_formal_positivity,
            inner_scale=self.inner_scale,
            inner_policy_regularization=self.inner_policy_regularization,
            steps_per_epoch=np.asarray(self.steps_per_epoch, dtype=np.int64),
            outer_iterations_completed=np.asarray(self.outer_iterations_completed, dtype=np.int64),
            inner_iterations_completed=np.asarray(self.inner_iterations_completed, dtype=np.int64),
        )


def _tb_writer_add_metrics(
    tb_writer: SummaryWriter | None,
    metrics: LyapunovTrainingMetrics,
    metric_scope: Literal["inner", "outer"],
) -> None:
    inner_raw_loss_str = "IRLoss/"
    inner_weighted_loss_str = "IWLoss/"

    rho_str = "Rho/"
    if tb_writer is not None and metric_scope == "outer":
        outer_iter = metrics.outer_iterations_completed - 1
        tb_step = metrics.outer_tb_step(outer_iter)
        tb_writer.add_scalar("LossTotal", metrics.loss[outer_iter], tb_step)
        tb_writer.add_scalar(rho_str + "Estimate", metrics.rho_estimate[outer_iter], tb_step)
        tb_writer.add_scalar(rho_str + "BoundaryQuantile", metrics.rho_boundary_quantile[outer_iter], tb_step)
        tb_writer.add_scalar(rho_str + "BoundaryMean", metrics.rho_boundary_mean[outer_iter], tb_step)
        tb_writer.add_scalar(rho_str + "FeatureTermQuantile", metrics.rho_feature_term_quantile[outer_iter], tb_step)
        tb_writer.add_scalar(rho_str + "LinearTermQuantile", metrics.rho_linear_term_quantile[outer_iter], tb_step)
        tb_writer.add_scalar(rho_str + "FeatureTermMeanShare", metrics.rho_feature_term_mean_share[outer_iter], tb_step)
        tb_writer.add_scalar(rho_str + "LinearTermMeanShare", metrics.rho_linear_term_mean_share[outer_iter], tb_step)
        tb_writer.add_scalar(rho_str + "RFactorFroNorm", metrics.r_factor_fro_norm[outer_iter], tb_step)
        tb_writer.add_scalar(
            "NumMinedCounterexamples",
            metrics.num_mined_counterexamples[outer_iter],
            tb_step,
        )
    
    if tb_writer is not None and metric_scope == "inner":
        inner_iter = metrics.inner_iterations_completed - 1
        tb_step = metrics.inner_tb_step(inner_iter)
        tb_writer.add_scalar(inner_raw_loss_str + "Total", metrics.inner_loss[inner_iter], tb_step)
        tb_writer.add_scalar(inner_raw_loss_str + "Condition", metrics.inner_condition_raw[inner_iter], tb_step)
        tb_writer.add_scalar(inner_raw_loss_str + "Roa", metrics.inner_roa_raw[inner_iter], tb_step)
        tb_writer.add_scalar(inner_raw_loss_str + "L1", metrics.inner_l1_raw[inner_iter], tb_step)
        tb_writer.add_scalar(inner_raw_loss_str + "Equilibrium", metrics.inner_equilibrium_raw[inner_iter], tb_step)
        tb_writer.add_scalar(inner_raw_loss_str + "FormalPositivity", metrics.inner_formal_positivity_raw[inner_iter], tb_step)
        tb_writer.add_scalar(inner_raw_loss_str + "Scale", metrics.inner_scale_raw[inner_iter], tb_step)
        tb_writer.add_scalar(
            inner_raw_loss_str + "PolicyRegularization",
            metrics.inner_policy_regularization_raw[inner_iter],
            tb_step,
        )
        tb_writer.add_scalar(inner_weighted_loss_str + "Condition", metrics.inner_condition[inner_iter], tb_step)
        tb_writer.add_scalar(inner_weighted_loss_str + "Roa", metrics.inner_roa[inner_iter], tb_step)
        tb_writer.add_scalar(inner_weighted_loss_str + "L1", metrics.inner_l1[inner_iter], tb_step)
        tb_writer.add_scalar(inner_weighted_loss_str + "Equilibrium", metrics.inner_equilibrium[inner_iter], tb_step)
        tb_writer.add_scalar(
            inner_weighted_loss_str + "FormalPositivity",
            metrics.inner_formal_positivity[inner_iter],
            tb_step,
        )
        tb_writer.add_scalar(inner_weighted_loss_str + "Scale", metrics.inner_scale[inner_iter], tb_step)
        tb_writer.add_scalar(
            inner_weighted_loss_str + "PolicyRegularization",
            metrics.inner_policy_regularization[inner_iter],
            tb_step,
        )

def _tb_writer_close(tb_writer: SummaryWriter | None) -> None:
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

def _tb_writer_build(log_dir: os.PathLike | None) -> SummaryWriter | None:
    if log_dir is not None:
        return SummaryWriter(log_dir=log_dir)
    return None


class LyapunovTrainer:
    """Trainer class for Lyapunov-stable neural controllers utilizing a CEGIS-style loop."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovTrainingConfig,
        rho_monitor: ThresholdMonitor | None = None,
        device: th.device | str = "cpu"
    ) -> None:
        self.config = config
        self.device = th.device(device)
        self.torch_gen = build_generator(self.config.seed, self.device)

        self.policy_model = policy_model.to(self.device)
        self.lyap_model = lyap_model.to(self.device)
        self.dyn_model = dyn_model.to(self.device)

        self._policy_start_epoch = self.config.outer_epochs - self.config.policy_epochs if self.config.policy_epochs is not None else float("inf")
        self._curr_policy_train_status = False
        self._set_train_modes()

        self.optimizer = self._build_optimizer()
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
    
        self.rho_monitor = rho_monitor
        self.results: LyapunovTrainingResult | None = None
        self.metrics: LyapunovTrainingMetrics | None = None

    @staticmethod
    def _build_scaled_state_bounds(
        base_bounds: NDArray,
        bound_scales: Sequence[float | Sequence[float] | NDArray],
    ) -> list[tuple[NDArray, NDArray]]:
        """Build a sequence of scaled training boxes from a base bounds array.

        Parameters
        ----------
        base_bounds : NDArray
            Bounds of shape ``(2, nx)`` containing ``[lbx, ubx]``.
        bound_scales : sequence
            Stage-wise scaling factors. Each element can be either a scalar
            scale applied to all dimensions or a per-dimension scale array of
            shape ``(nx,)``.

        Returns
        -------
        list[tuple[NDArray, NDArray]]
            Pairs ``(stage_bounds, stage_scale)`` for each curriculum stage.
        """
        bounds = np.asarray(base_bounds, dtype=np.float32)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("base_bounds must have shape (2, nx).")
        if len(bound_scales) == 0:
            raise ValueError("bound_scales must contain at least one stage.")

        state_dim = bounds.shape[1]
        scaled_bounds: list[tuple[NDArray, NDArray]] = []
        for scale in bound_scales:
            scale_arr = np.asarray(scale, dtype=np.float32)
            if scale_arr.ndim == 0:
                scale_arr = np.full((state_dim,), float(scale_arr), dtype=np.float32)
            elif scale_arr.shape != (state_dim,):
                raise ValueError(
                    f"Each bound scale must be a scalar or shape ({state_dim},), got {scale_arr.shape}."
                )

            if np.any(scale_arr <= 0.0):
                raise ValueError("All bound scales must be positive.")

            scaled_bounds.append((bounds * scale_arr.reshape(1, -1), scale_arr.copy()))

        return scaled_bounds

    def _set_train_modes(self) -> None:
        for param in self.lyap_model.parameters():
            param.requires_grad_(True)
        self.lyap_model.train()

        for param in self.dyn_model.parameters():
            param.requires_grad_(False)
        self.dyn_model.eval()
        self._set_policy_train_mode(train_mode=False)
        
    def _set_policy_train_mode(self, train_mode: bool) -> None:
        for param in self.policy_model.parameters():
            param.requires_grad_(train_mode)
        self.policy_model.train(train_mode)

    def _get_train_params(self) -> tuple[nn.Parameter, ...]:
        """Utility to gather trainable parameters based on the config."""
        return tuple(
            param
            for model in (self.lyap_model, self.policy_model)
            for param in model.parameters()
            if param.requires_grad
        )

    def _build_optimizer(self) -> th.optim.Adam:
        return th.optim.Adam(self._get_train_params(), self.config.learning_rate)

    def _enable_policy_training(self) -> None:
        self._curr_policy_train_status = True
        self._set_policy_train_mode(train_mode=True)
        self.optimizer.add_param_group({"params": self.policy_model.parameters()})

    def _mine_new_counterexamples(self, rho_estimate: float) -> th.Tensor:
        """Mine rho-gated counterexamples using the external training semantics."""
        return find_counter_examples(
            objective=lambda x: self.loss_module.mining_objective(
                x_batch=x,
                rho_estimate=rho_estimate,
            ),
            config=self.config,
            device=self.device,
            generator=self.torch_gen,
        )

    def _build_roa_candidates(self) -> th.Tensor:
        """Create diverse candidate states near the boundary of the asymmetric B."""
        directions = th.randn(
            self.config.roa_candidate_size,
            self.config.state_dim,
            device=self.device,
            generator=self.torch_gen,
        )
        directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
        radii = th.rand(
            self.config.roa_candidate_size,
            1,
            device=self.device,
            generator=self.torch_gen
        ) * 0.4 + 0.6
        z_candidates = directions * radii  # between 0.6 and 1.0 in random directions
        half_width = 0.5 * (self.ubx - self.lbx)
        return z_candidates * half_width + self.center

    def train(self) -> LyapunovTrainingResult:
        """Execute the CEGIS-style training loop."""
        metrics = LyapunovTrainingMetrics.from_num_steps(
            num_outer_epochs=self.config.outer_epochs,
            steps_per_epoch=self.config.steps_per_epoch,
        )
        
        # Initial training pool sampled uniformly from the state space bounds
        initial_x = sample_uniform_box(self.config.initial_sample_size, self.lbx, self.ubx, self.device)
        state_buffer = DynamicStateBuffer(
            initial_states=initial_x,
            state_buffer_limit=self.config.state_buffer_limit,
            cex_buffer_limit=self.config.cex_buffer_limit,
            device=self.device,
            min_cex_fraction=self.config.cex_fraction_min,
            max_cex_fraction=self.config.cex_fraction_max,
            generator=self.torch_gen,
        )
        boundary_buffer = BoundaryStateBuffer(
            state_dim=self.config.state_dim,
            max_size=int(self.config.rho_boundary_buffer_size),
            device=self.device,
            dtype=initial_x.dtype,
        )
        roa_candidates = self._build_roa_candidates()

        mining_interval = max(1, int(self.config.counterexample_every))
        rho_estimate = self.config.rho_min
        cex_fraction_ema = 0.0
        cex_fraction = 0.0
        total_steps = self.config.outer_epochs * self.config.steps_per_epoch

        tb_writer = _tb_writer_build(self.config.tb_log_dir)
        start_time = time.time()

        with GracefulInterruptHandler(logger=__logger__) as interrupt_handler:
            with Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("loss: {task.fields[loss]:.4f}"),
                TextColumn("ρ: {task.fields[rho]:.4f}"),
                TextColumn("cex_pool: {task.fields[cex_pool]:.0f}"),
                TextColumn("cex_samples/batch: {task.fields[cex_samples]:.0f}/{task.fields[batch_size]:.0f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    "Lyapunov Training Iterations",
                    total=float(total_steps),
                    loss=float("nan"),
                    rho=float(rho_estimate),
                    cex_pool=float(0.0),
                    cex_samples=float(0.0),
                    batch_size=float(self.config.batch_size),
                )
                for outer_iter in range(self.config.outer_epochs):
                    start_policy_training = (outer_iter >= self._policy_start_epoch)
                    if start_policy_training and not self._curr_policy_train_status:
                        __logger__.info("Starting policy training at outer iteration %d.", outer_iter)
                        self._enable_policy_training()

                    last_loss_value = np.nan
                    
                    # Estimate current Region of Attraction
                    rho_diagnostics = estimate_rho_from_boundary_diagnostics(
                        lyap_model=self.lyap_model,
                        config=self.config,
                        device=self.device,
                        boundary_buffer=boundary_buffer,
                        # cex_buffer=state_buffer,
                        generator=self.torch_gen,
                    )
                    rho_estimate = rho_diagnostics.rho # TODO: Consider smoothing or just constant
                    if self.rho_monitor is not None and self.rho_monitor.update(rho_estimate):
                        train_time = time.time() - start_time
                        abort_reason = (
                            "Lyapunov training aborted after "
                            f"{self.rho_monitor.consecutive_low} consecutive rho estimates "
                            f"below {self.rho_monitor.threshold:.3f}."
                        )
                        self.results = LyapunovTrainingResult(
                            rho_estimate=rho_estimate,
                            num_mined_counterexamples=state_buffer.cex_count,
                            train_time=train_time,
                            aborted=True,
                            abort_reason=abort_reason,
                        )
                        self.metrics = metrics
                        _tb_writer_close(tb_writer)
                        __logger__.info(
                            "Aborting Lyapunov training after %d consecutive rho estimates below %.3f.",
                            self.rho_monitor.consecutive_low,
                            self.rho_monitor.threshold,
                        )
                        return self.results

                    # Mine counterexamples (CEGIS)
                    if (outer_iter + 1) % mining_interval == 0:
                        new_cex = self._mine_new_counterexamples(rho_estimate=rho_estimate)
                        state_buffer.register_cex(
                            new_cex,
                            objective=self.loss_module.buffer_sorting_objective,
                        )
                        if new_cex.numel() == 0:
                            __logger__.info("No new counterexamples mined at outer iteration %d.", outer_iter)
                        roa_candidates = self._build_roa_candidates()

                        frac_yield = new_cex.shape[0] / self.config.adversarial_samples
                        cex_fraction_ema = self.config.cex_fraction_ema_decay * cex_fraction_ema + (1 - self.config.cex_fraction_ema_decay) * frac_yield
                        cex_fraction = self.config.cex_fraction_min + cex_fraction_ema * (self.config.cex_fraction_max - self.config.cex_fraction_min)

                    # Inner training loop
                    for inner_step in range(self.config.steps_per_epoch):
                        x_batch = state_buffer.sample(self.config.batch_size, cex_fraction=cex_fraction)
                        loss = self.loss_module(
                            x_batch=x_batch,
                            roa_candidates=roa_candidates,
                            rho_estimate=rho_estimate,
                            active_policy_regularization=self._curr_policy_train_status,
                        )

                        loss_parts = self.loss_module.last_loss_parts
                        if loss_parts is None:
                            raise RuntimeError("Lyapunov loss parts were not computed during the forward pass.")

                        global_inner_iter = outer_iter * self.config.steps_per_epoch + inner_step
                        metrics.fill_inner(
                            inner_iter=global_inner_iter,
                            loss_parts=loss_parts,
                        )
                        _tb_writer_add_metrics(tb_writer, metrics, metric_scope="inner")

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        # Update Progress Bar
                        progress.update(
                            task,
                            advance=1.0,
                            loss=none_to_float(loss.item()),
                            rho=none_to_float(rho_estimate),
                            cex_pool=none_to_float(state_buffer.cex_count),
                            cex_samples=none_to_float(cex_fraction * self.config.batch_size),
                        )
                        last_loss_value = float(loss.item())

                    metrics.fill_outer(
                        outer_iter=outer_iter,
                        loss_value=last_loss_value,
                        state_buffer=state_buffer,
                        num_mined_counterexamples=state_buffer.cex_count,
                        rho_diagnostics=rho_diagnostics,
                    )
                    _tb_writer_add_metrics(tb_writer, metrics, metric_scope="outer")

        train_time = time.time() - start_time
        __logger__.debug("Lyapunov training finished in %.2fs", train_time)

        if interrupt_handler.aborted:
            self.results = LyapunovTrainingResult(
                rho_estimate=rho_estimate,
                num_mined_counterexamples=state_buffer.cex_count,
                train_time=train_time,
                aborted=True,
                abort_reason="Lyapunov training interrupted by user.",
            )
            __logger__.info("Lyapunov training interrupted by user after %d outer iterations.", metrics.outer_iterations_completed)
        else:
            self.results = LyapunovTrainingResult(
                rho_estimate=rho_estimate,
                num_mined_counterexamples=state_buffer.cex_count,
                train_time=train_time,
            )

        self.metrics = metrics
        _tb_writer_close(tb_writer)
        return self.results

    def train_with_scaled_bounds(
        self,
        bound_scales: Sequence[float | Sequence[float] | NDArray]
    ) -> LyapunovTrainingCurriculumResult:
        """Train on a curriculum of progressively scaled state bounds.

        The same policy and Lyapunov model instances are reused across stages,
        so each stage warm-starts from the weights learned on the previous one.
        The trainer instance is updated in-place to the final curriculum stage.
        """
        base_bounds = self.config.state_bounds
        scaled_bounds = self._build_scaled_state_bounds(
            base_bounds=base_bounds,
            bound_scales=bound_scales,
        )

        stage_records: list[LyapunovTrainingCurriculumStage] = []
        final_stage_trainer: LyapunovTrainer | None = None
        aborted_result: LyapunovTrainingResult | None = None
        aborted_stage_index: int | None = None

        for stage_index, (stage_bounds, stage_scale) in enumerate(scaled_bounds):
            stage_tb_log_dir = None
            if self.config.tb_log_dir is not None:
                stage_tb_log_dir = Path(self.config.tb_log_dir) / f"curriculum_stage_{stage_index:02d}"

            stage_seed = None if self.config.seed is None else self.config.seed + stage_index
            stage_config = replace(
                self.config,
                state_bounds=stage_bounds,
                seed=stage_seed,
                tb_log_dir=stage_tb_log_dir,
            )
            stage_trainer = type(self)(
                policy_model=self.policy_model,
                lyap_model=self.lyap_model,
                dyn_model=self.dyn_model,
                config=stage_config,
                rho_monitor=self.rho_monitor,
                device=self.device,
            )
            stage_result = stage_trainer.train()
            final_stage_trainer = stage_trainer

            if stage_result.aborted:
                aborted_result = stage_result
                aborted_stage_index = stage_index
                __logger__.info(
                    "Stopping curriculum after stage %d aborted: %s",
                    stage_index,
                    stage_result.abort_reason,
                )
                break

            stage_records.append(
                LyapunovTrainingCurriculumStage(
                    stage_index=stage_index,
                    state_bounds=stage_bounds.copy(),
                    scale=stage_scale.copy(),
                    result=stage_result,
                )
            )

        if final_stage_trainer is None:
            return LyapunovTrainingCurriculumResult(stages=[])

        self.config = final_stage_trainer.config
        self.optimizer = final_stage_trainer.optimizer
        self.loss_module = final_stage_trainer.loss_module
        self.lbx = final_stage_trainer.lbx
        self.ubx = final_stage_trainer.ubx
        self.center = final_stage_trainer.center
        self.results = final_stage_trainer.results
        self.metrics = final_stage_trainer.metrics

        return LyapunovTrainingCurriculumResult(
            stages=stage_records,
            aborted_result=aborted_result,
            aborted_stage_index=aborted_stage_index,
        )

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
        config_path = save_folder / TRAINING_CONFIG_FILENAME
        self.config.save(config_path)

        # Models saving 
        lyap_model_path = save_folder / LYAPUNOV_MODEL_FILENAME
        policy_model_path = save_folder / POLICY_MODEL_FILENAME

        save_model_checkpoint(self.lyap_model, lyap_model_path)
        save_model_checkpoint(self.policy_model, policy_model_path)

        # Metric saving
        if self.metrics is not None:
            metrics_path = save_folder / TRAINING_METRICS_FILENAME
            self.metrics.save(metrics_path)

        # Optional: Update the results object with the paths if it exists
        if self.results is not None:
            self.results.lyap_model_path = str(lyap_model_path)
            self.results.policy_model_path = str(policy_model_path)
            self.results.save(save_folder)
        
        __logger__.info(f"Saved lyapunov results to {save_folder}")
