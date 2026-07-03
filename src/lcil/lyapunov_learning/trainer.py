from __future__ import annotations

import os
import time
import logging
import numpy as np
import torch as th
import torch.nn as nn

from collections.abc import Sequence
from numpy.typing import NDArray
from pathlib import Path
from dataclasses import replace
from mpc_datagen.mpc_data import MPCConfig
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
from .loss import LyapunovTrainingLoss
from .counterexample import (
    BoundaryRhoDiagnostics,
    estimate_rho_from_boundary,
    find_counter_examples,
    sample_uniform_box,
    sample_boundary_points,
)
from .results import (
    LyapunovTrainingResult,
    LyapunovTrainingMetrics,
    LyapunovTrainingCurriculumResult,
    LyapunovTrainingCurriculumStage,
)
from .tensorboard import (
    tb_writer_build,
    tb_writer_close,
    tb_writer_add_metrics,
    tb_writer_add_parallel_coordinates,
)
from .utils import (
    ThresholdMonitor,
    get_th_lbx_ubx,
    get_center,
    get_ema,
)
from ..utils import (
    GracefulInterruptHandler,
    RegionBuilder,
    save_model_checkpoint,
    build_generator,
    none_to_float,
    timeit,
    save_mpc_config_for_run,
)
from ..utils.constants import *

__logger__ = logging.getLogger(__name__)


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
        self.loss_module = LyapunovTrainingLoss(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            config=self.config,
            device=self.device,
        )

        self._policy_start_epoch = self.config.outer_epochs - self.config.policy_epochs if self.config.policy_epochs is not None else float("inf")
        self._curr_policy_train_status = False
        self._set_train_modes()
        self.optimizer = self._build_optimizer()
        
        self.lbx_train, self.ubx_train = get_th_lbx_ubx(self.config.train_bounds, self.device)
        self.center_train = get_center(self.lbx_train, self.ubx_train)

        self.region_builder = RegionBuilder(
            bounds=self.config.train_bounds,
            bins_per_dim=self.config.bins_per_dim,
            origin_exclusion=self.config.origin_exclusion,
            device=self.device,
        )
    
        self.rho_monitor = rho_monitor
        self.results: LyapunovTrainingResult | None = None
        self.metrics: LyapunovTrainingMetrics | None = None

    @staticmethod
    def _build_scaled_train_bounds(
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

    def _enable_policy_training(self, at_iter: int = 0) -> None:
        self._curr_policy_train_status = True
        self._set_policy_train_mode(train_mode=True)

        policy_lr = self.config.learning_rate * self.config.policy_lr_factor
        self.optimizer.add_param_group({
            "params": self.policy_model.parameters(),
            "lr": policy_lr,
        })
        self.loss_module.set_explicit_l1_params(list(self.policy_model.parameters()) + list(self.lyap_model.parameters()))

        __logger__.info(
            "Policy training enabled at iteration %d! Added to optimizer with LR: %.2e (Factor: %s)",
            at_iter, policy_lr, self.config.policy_lr_factor
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
            generator=self.torch_gen,
        )

    def _build_roa_candidates(self, injection_states: th.Tensor) -> th.Tensor:
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
        half_width = 0.5 * (self.ubx_train - self.lbx_train)
        random_candidates = z_candidates * half_width + self.center_train

        if injection_states is not None and injection_states.numel() > 0:
            return th.cat((random_candidates, injection_states), dim=0)
        
        return random_candidates

    def _get_boundary_buffer(self):
        boundary_buffer = BoundaryStateBuffer(
            state_dim=self.config.state_dim,
            max_size=int(self.config.rho_boundary_buffer_size),
            max_age=self.config.roa_max_age,
            device=self.device,
        )
        init_boundary_x, _, _ = sample_boundary_points(
            sample_size=int(self.config.rho_boundary_buffer_size),
            lb=self.lbx_train,
            ub=self.ubx_train,
            device=self.device,
            generator=self.torch_gen,
        )
        boundary_buffer.update(init_boundary_x, value_fn=self.lyap_model)
        return boundary_buffer

    def _get_cegis_buffer(self) -> DynamicStateBuffer:
        initial_x = sample_uniform_box(self.config.initial_sample_size, self.lbx_train, self.ubx_train, self.device)
        cegis_buffer = DynamicStateBuffer(
            initial_states=initial_x,
            state_buffer_limit=self.config.state_buffer_limit,
            cex_buffer_limit=self.config.cex_buffer_limit,
            min_cex_fraction=self.config.cex_fraction_min,
            max_cex_fraction=self.config.cex_fraction_max,
            max_cex_age=self.config.cex_max_age,
            generator=self.torch_gen,
            device=self.device,
        )
        return cegis_buffer
    
    # ==========================================
    # --- TRAINING HELPER METHODS ---
    # ==========================================

    def _init_training_components(
        self
    ) -> tuple[LyapunovTrainingMetrics, DynamicStateBuffer, BoundaryStateBuffer, th.Tensor]:
        metrics = LyapunovTrainingMetrics.from_num_steps(
            num_outer_epochs=self.config.outer_epochs,
            steps_per_epoch=self.config.steps_per_epoch,
        )
        cegis_buffer = self._get_cegis_buffer()
        boundary_buffer = self._get_boundary_buffer()
        lirpa_regions = self.region_builder.build_regions()
        return metrics, cegis_buffer, boundary_buffer, lirpa_regions

    def _build_progress_bar(self) -> Progress:
        return Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss: {task.fields[loss]:.4f}"),
            TextColumn("ρ: {task.fields[rho]:.4f}"),
            TextColumn("cex_pool: {task.fields[cex_pool]:.0f}"),
            TextColumn("cex_samples/batch: {task.fields[cex_samples]:.0f}/{task.fields[batch_size]:.0f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

    def _init_progress_task(self, progress: Progress, total_steps: int, description: str) -> int:
        return progress.add_task(
            description=description,
            total=float(total_steps),
            loss=float("nan"),
            rho=float(0.0),
            cex_pool=float(0.0),
            cex_samples=float(0.0),
            batch_size=float(self.config.batch_size),
        )

    def _update_policy_training_status(self, outer_iter: int) -> None:
        start_policy_training = (outer_iter >= self._policy_start_epoch)
        if start_policy_training and not self._curr_policy_train_status:
            self._enable_policy_training(at_iter=outer_iter * self.config.steps_per_epoch)

    def _evaluate_boundary_and_roa(
        self, boundary_buffer: BoundaryStateBuffer, current_rho_estimate: float | None
    ) -> tuple[float, th.Tensor, BoundaryRhoDiagnostics]:
        
        rho_diagnostics, boundary_states = estimate_rho_from_boundary(
            lyap_model=self.lyap_model,
            config=self.config,
            device=self.device,
            generator=self.torch_gen,
        )
        boundary_buffer.update(boundary_states, value_fn=self.lyap_model)
        roa_candidates = self._build_roa_candidates(injection_states=boundary_buffer.states)
        
        new_rho_estimate = get_ema(current_rho_estimate, rho_diagnostics.rho, self.config.rho_ema_decay)
        return new_rho_estimate, roa_candidates, rho_diagnostics

    def _should_abort_training(self, rho_estimate: float) -> bool:
        if self.rho_monitor is not None:
            return self.rho_monitor.update(rho_estimate)
        return False

    def _is_mining_step(self, outer_iter: int, mining_interval: int) -> bool:
        return (outer_iter + 1) % mining_interval == 0

    def _mine_cegis_step(
        self,
        outer_iter: int,
        mining_interval: int,
        rho_estimate: float,
        state_buffer: DynamicStateBuffer,
        cex_fraction_ema: float | None,
    ) -> tuple[float, float | None]:
        
        # Only mine at intervals
        if not self._is_mining_step(outer_iter, mining_interval):
            if cex_fraction_ema is None:
                return 0.0, None
            current_fraction = self.config.cex_fraction_min + \
                cex_fraction_ema * (self.config.cex_fraction_max - self.config.cex_fraction_min)
            return current_fraction, cex_fraction_ema

        # New mining
        new_cex = self._mine_new_counterexamples(rho_estimate=rho_estimate)
        state_buffer.register_cex(new_cex, objective=self.loss_module.buffer_sorting_objective)
        
        if new_cex.numel() == 0:
            __logger__.info("No new counterexamples mined at outer iteration %d.", outer_iter)

        frac_yield = new_cex.shape[0] / self.config.adversarial_samples
        new_ema = get_ema(cex_fraction_ema, frac_yield, self.config.cex_fraction_ema_decay)
        new_fraction = self.config.cex_fraction_min + new_ema * (self.config.cex_fraction_max - self.config.cex_fraction_min)

        return new_fraction, new_ema

    def _finalize_training(
        self, 
        cex_count: int,
        rho_estimate: float,
        start_time: float,
        abort_reason: str | None = None,
    ) -> LyapunovTrainingResult:
        """Handles final training results for both successes and aborted runs."""
        train_time = time.time() - start_time
        if abort_reason:
            __logger__.info(abort_reason)
        else:
            __logger__.info("Lyapunov training completed successfully in %.2fs " \
                "with final rho estimate: %.4f and %d mined counterexamples.",
                train_time, rho_estimate, cex_count
            )
            
        self.results = LyapunovTrainingResult(
            rho_estimate=rho_estimate,
            num_mined_counterexamples=cex_count,
            train_time=train_time,
            aborted=(abort_reason is not None),
            abort_reason=abort_reason,
        )
        return self.results

    def train(self, description: str = "Lyapunov Learning") -> LyapunovTrainingResult:
        """Execute the CEGIS-style training loop."""
        self.metrics, cegis_buffer, boundary_buffer, lirpa_regions = self._init_training_components()

        mining_interval = max(1, int(self.config.cex_every))
        rho_estimate = None
        cex_fraction_ema = None
        cex_fraction = 0.0
        total_steps = self.config.outer_epochs * self.config.steps_per_epoch

        tb_writer = tb_writer_build(self.config.tb_log_dir)
        start_time = time.time()

        with GracefulInterruptHandler(logger=__logger__) as interrupt_handler:
            with self._build_progress_bar() as progress:
                task = self._init_progress_task(progress, total_steps, description)

                for outer_iter in range(self.config.outer_epochs):
                    self._update_policy_training_status(outer_iter)

                    rho_estimate, roa_candidates, rho_diagnostics = self._evaluate_boundary_and_roa(
                        boundary_buffer, rho_estimate
                    )

                    if self._should_abort_training(rho_estimate):
                        abort_reason = (
                            "Lyapunov training aborted after "
                            f"{self.rho_monitor.consecutive_low} consecutive rho estimates "
                            f"below {self.rho_monitor.threshold:.3f}."
                        )
                        tb_writer_close(tb_writer)
                        return self._finalize_training(
                            cex_count=cegis_buffer.cex_count,
                            rho_estimate=none_to_float(rho_estimate),
                            start_time=start_time,
                            abort_reason=abort_reason
                        )

                    cex_fraction, cex_fraction_ema = self._mine_cegis_step(
                        outer_iter, mining_interval, rho_estimate, cegis_buffer, cex_fraction_ema
                    )

                    # Inner training loop
                    for inner_step in range(self.config.steps_per_epoch):
                        x_batch = cegis_buffer.sample(self.config.batch_size, cex_fraction=cex_fraction)
                        loss = self.loss_module(
                            x_batch=x_batch,
                            roa_candidates=roa_candidates,
                            rho_estimate=rho_estimate,
                            lirpa_regions=lirpa_regions,
                            active_policy_regularization=self._curr_policy_train_status,
                        )

                        loss_parts = self.loss_module.last_loss_parts
                        if loss_parts is None:
                            raise RuntimeError("Lyapunov loss parts were not computed during the forward pass.")

                        self.metrics.fill_inner(
                            inner_iter=outer_iter * self.config.steps_per_epoch + inner_step,
                            loss_parts=loss_parts,
                        )

                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        th.nn.utils.clip_grad_norm_(self._get_train_params(), max_norm=1.0)
                        self.optimizer.step()

                        # Update Progress Bar
                        progress.update(
                            task,
                            advance=1.0,
                            loss=none_to_float(loss.item()),
                            rho=none_to_float(rho_estimate),
                            cex_pool=none_to_float(cegis_buffer.cex_count),
                            cex_samples=none_to_float(cex_fraction * self.config.batch_size),
                        )

                    self.metrics.fill_outer(
                        outer_iter=outer_iter,
                        state_buffer=cegis_buffer,
                        num_mined_counterexamples=cegis_buffer.cex_count,
                        rho_diagnostics=rho_diagnostics,
                    )
                    tb_writer_add_metrics(tb_writer, self.metrics)
                    if self._is_mining_step(outer_iter, mining_interval) and cegis_buffer.cexs.numel() > 0:
                        tb_writer_add_parallel_coordinates(
                            tb_writer,
                            tag="Counterexamples",
                            states=cegis_buffer.cexs,
                            state_bounds=self.config.train_bounds,
                            origin_exclusion=self.config.origin_exclusion,
                            global_step=outer_iter + 1,
                            max_lines=512,
                        )
                    

        tb_writer_close(tb_writer)
        final_abort_reason = None
        if interrupt_handler.aborted:
            final_abort_reason = "Lyapunov training interrupted by user."

        return self._finalize_training(
            cex_count=cegis_buffer.cex_count,
            rho_estimate=none_to_float(rho_estimate),
            start_time=start_time,
            abort_reason=final_abort_reason,
        )

    def train_with_scaled_bounds(
        self,
        bound_scales: Sequence[float | Sequence[float] | NDArray]
    ) -> LyapunovTrainingCurriculumResult:
        """Train on a curriculum of progressively scaled state bounds.

        The same policy and Lyapunov model instances are reused across stages,
        so each stage warm-starts from the weights learned on the previous one.
        The trainer instance is updated in-place to the final curriculum stage.
        """
        base_bounds = self.config.train_bounds
        scaled_bounds = self._build_scaled_train_bounds(
            base_bounds=base_bounds,
            bound_scales=bound_scales,
        )

        stage_records: list[LyapunovTrainingCurriculumStage] = []
        final_stage_trainer: LyapunovTrainer | None = None
        aborted_result: LyapunovTrainingResult | None = None
        aborted_stage_index: int | None = None
        last_stage_idx = len(scaled_bounds) - 1

        for stage_index, (stage_bounds, stage_scale) in enumerate(scaled_bounds):
            stage_tb_log_dir = None
            if self.config.tb_log_dir is not None:
                stage_tb_log_dir = Path(self.config.tb_log_dir) / f"curriculum_stage_{stage_index:02d}"

            stage_seed = None if self.config.seed is None else self.config.seed + stage_index
            rho_min = stage_records[-1].result.rho_estimate if stage_records else self.config.rho_min
            stage_config = replace(
                self.config,
                train_bounds=stage_bounds,
                seed=stage_seed,
                tb_log_dir=stage_tb_log_dir,
                rho_min=rho_min,
                outer_epochs=int(self.config.outer_epochs * 0.5) if stage_index != last_stage_idx else self.config.outer_epochs,
                policy_epochs=int(self.config.policy_epochs * 0.5) if stage_index != last_stage_idx else self.config.policy_epochs,

            )
            
            stage_trainer = type(self)(
                policy_model=self.policy_model,
                lyap_model=self.lyap_model,
                dyn_model=self.dyn_model,
                config=stage_config,
                rho_monitor=self.rho_monitor,
                device=self.device,
            )
            stage_result = stage_trainer.train(description=f"Lyapunov Learning Stage [{stage_index + 1}/{len(scaled_bounds)}]")
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
                    train_bounds=stage_bounds.copy(),
                    scale=stage_scale.copy(),
                    result=stage_result,
                )
            )

        if final_stage_trainer is None:
            return LyapunovTrainingCurriculumResult(stages=[])

        self.config = replace(
            final_stage_trainer.config,
            rho_min=self.config.rho_min,
            seed=self.config.seed,
        )
        self.optimizer = final_stage_trainer.optimizer
        self.loss_module = final_stage_trainer.loss_module
        self.lbx_train = final_stage_trainer.lbx_train
        self.ubx_train = final_stage_trainer.ubx_train
        self.center_train = final_stage_trainer.center_train
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
        mpc_config: MPCConfig | None = None,
    ) -> None:
        """Utility function to save training results and model checkpoints.

        Parameters
        ----------
        save_folder : PathLike[str]
            Folder where the models and config should be saved.
        mpc_config : MPCConfig | None, optional
            Optional MPC configuration saved alongside the copied policy checkpoint.
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

        if mpc_config is not None:
            save_mpc_config_for_run(mpc_config, save_folder)
        

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
