from __future__ import annotations

import os
import numpy as np
import torch as th

from numpy.typing import NDArray
from pathlib import Path
from dataclasses import dataclass

from .buffer import CEGISBuffer
from .loss import LyapunovTrainingLossParts
from .counterexample import BoundaryRhoEvaluation
from ..utils import (
    JsonDataclass,
    timeit,
)
from ..utils.constants import *

@dataclass
class MiningStepResult:
    cex_fraction: float
    cex_fraction_ema: float | None
    mined_cex_states: th.Tensor | None
    mined_cex_violations: th.Tensor | None

    @classmethod
    def empty(cls):
        return cls(0.0, None, None, None)


@dataclass
class BoundaryStepResult:
    rho_estimate: float
    roa_candidates: th.Tensor
    rho_evaluation: BoundaryRhoEvaluation


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
    train_bounds: NDArray
    scale: NDArray
    result: LyapunovTrainingResult

    NP_ARRAY_FIELDS = ("train_bounds", "scale")
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
    def final_train_bounds(self) -> NDArray | None:
        if not self.stages:
            return None
        return self.stages[-1].train_bounds


@dataclass
class LyapunovTrainingMetrics:
    """Per-outer-iteration training metrics for Lyapunov optimization."""

    rho_estimate: NDArray
    rho_boundary_quantile: NDArray
    rho_boundary_mean: NDArray
    rho_feature_term_quantile: NDArray
    rho_linear_term_quantile: NDArray
    rho_feature_term_mean: NDArray
    rho_linear_term_mean: NDArray
    rho_feature_term_mean_share: NDArray
    rho_linear_term_mean_share: NDArray
    buffer_size: NDArray
    num_mined_counterexamples: NDArray
    loss: NDArray
    condition_raw: NDArray
    roa_raw: NDArray
    condition_lirpa_raw: NDArray
    l1_raw: NDArray
    equilibrium_raw: NDArray
    formal_positivity_raw: NDArray
    scale_raw: NDArray
    policy_regularization_raw: NDArray
    r_factor_fro_norm_raw: NDArray
    condition: NDArray
    roa: NDArray
    condition_lirpa: NDArray
    l1: NDArray
    equilibrium: NDArray
    formal_positivity: NDArray
    scale: NDArray
    policy_regularization: NDArray
    r_factor_fro_norm: NDArray
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
            rho_estimate=outer_nan_array.copy(),
            rho_boundary_quantile=outer_nan_array.copy(),
            rho_boundary_mean=outer_nan_array.copy(),
            rho_feature_term_quantile=outer_nan_array.copy(),
            rho_linear_term_quantile=outer_nan_array.copy(),
            rho_feature_term_mean=outer_nan_array.copy(),
            rho_linear_term_mean=outer_nan_array.copy(),
            rho_feature_term_mean_share=outer_nan_array.copy(),
            rho_linear_term_mean_share=outer_nan_array.copy(),
            buffer_size=outer_nan_array.copy(),
            num_mined_counterexamples=outer_nan_array.copy(),
            loss=inner_nan_array.copy(),
            condition_raw=inner_nan_array.copy(),
            roa_raw=inner_nan_array.copy(),
            condition_lirpa_raw=inner_nan_array.copy(),
            l1_raw=inner_nan_array.copy(),
            equilibrium_raw=inner_nan_array.copy(),
            formal_positivity_raw=inner_nan_array.copy(),
            scale_raw=inner_nan_array.copy(),
            policy_regularization_raw=inner_nan_array.copy(),
            r_factor_fro_norm_raw=inner_nan_array.copy(),
            condition=inner_nan_array.copy(),
            roa=inner_nan_array.copy(),
            condition_lirpa=inner_nan_array.copy(),
            l1=inner_nan_array.copy(),
            equilibrium=inner_nan_array.copy(),
            formal_positivity=inner_nan_array.copy(),
            scale=inner_nan_array.copy(),
            policy_regularization=inner_nan_array.copy(),
            r_factor_fro_norm=inner_nan_array.copy(),
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
        self.loss[inner_iter] = float(loss_parts.total.item())
        self.condition_raw[inner_iter] = float(loss_parts.condition_raw.item())
        self.roa_raw[inner_iter] = float(loss_parts.roa_raw.item())
        self.condition_lirpa_raw[inner_iter] = float(loss_parts.condition_lirpa_raw.item())
        self.l1_raw[inner_iter] = float(loss_parts.l1_raw.item())
        self.equilibrium_raw[inner_iter] = float(loss_parts.equilibrium_raw.item())
        self.formal_positivity_raw[inner_iter] = float(loss_parts.formal_positivity_raw.item())
        self.scale_raw[inner_iter] = float(loss_parts.scale_raw.item())
        self.policy_regularization_raw[inner_iter] = float(loss_parts.policy_regularization_raw.item())
        self.r_factor_fro_norm_raw[inner_iter] = float(loss_parts.r_factor_fro_norm_raw.item())

        self.condition[inner_iter] = float(loss_parts.condition.item())
        self.roa[inner_iter] = float(loss_parts.roa.item())
        self.condition_lirpa[inner_iter] = float(loss_parts.condition_lirpa.item())
        self.l1[inner_iter] = float(loss_parts.l1.item())
        self.equilibrium[inner_iter] = float(loss_parts.equilibrium.item())
        self.formal_positivity[inner_iter] = float(loss_parts.formal_positivity.item())
        self.scale[inner_iter] = float(loss_parts.scale.item())
        self.policy_regularization[inner_iter] = float(loss_parts.policy_regularization.item())
        self.r_factor_fro_norm[inner_iter] = float(loss_parts.r_factor_fro_norm.item())
        self.inner_iterations_completed = inner_iter + 1

    def fill_outer(
        self,
        outer_iter: int,
        state_buffer: CEGISBuffer,
        num_mined_counterexamples: int,
        boundary_eval: BoundaryRhoEvaluation,
    ) -> None:
        self.buffer_size[outer_iter] = float(state_buffer.state_count)
        self.num_mined_counterexamples[outer_iter] = float(num_mined_counterexamples)
        self.outer_iterations_completed = outer_iter + 1

        self.rho_estimate[outer_iter] = boundary_eval.rho.rho
        self.rho_boundary_quantile[outer_iter] = boundary_eval.rho.boundary_quantile
        self.rho_boundary_mean[outer_iter] = boundary_eval.rho.boundary_mean
        self.rho_feature_term_quantile[outer_iter] = boundary_eval.terms.feature_term_quantile
        self.rho_linear_term_quantile[outer_iter] = boundary_eval.terms.linear_term_quantile
        self.rho_feature_term_mean[outer_iter] = boundary_eval.terms.feature_term_mean
        self.rho_linear_term_mean[outer_iter] = boundary_eval.terms.linear_term_mean
        self.rho_feature_term_mean_share[outer_iter] = boundary_eval.terms.feature_term_mean_share
        self.rho_linear_term_mean_share[outer_iter] = boundary_eval.terms.linear_term_mean_share

    def save(self, path: os.PathLike) -> None:
        metrics_path = Path(path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            metrics_path,
            rho_estimate=self.rho_estimate,
            rho_boundary_quantile=self.rho_boundary_quantile,
            rho_boundary_mean=self.rho_boundary_mean,
            rho_feature_term_quantile=self.rho_feature_term_quantile,
            rho_linear_term_quantile=self.rho_linear_term_quantile,
            rho_feature_term_mean=self.rho_feature_term_mean,
            rho_linear_term_mean=self.rho_linear_term_mean,
            rho_feature_term_mean_share=self.rho_feature_term_mean_share,
            rho_linear_term_mean_share=self.rho_linear_term_mean_share,
            buffer_size=self.buffer_size,
            num_mined_counterexamples=self.num_mined_counterexamples,
            loss=self.loss,
            condition_raw=self.condition_raw,
            roa_raw=self.roa_raw,
            condition_lirpa_raw=self.condition_lirpa_raw,
            l1_raw=self.l1_raw,
            equilibrium_raw=self.equilibrium_raw,
            formal_positivity_raw=self.formal_positivity_raw,
            scale_raw=self.scale_raw,
            policy_regularization_raw=self.policy_regularization_raw,
            r_factor_fro_norm_raw=self.r_factor_fro_norm_raw,
            condition=self.condition,
            roa=self.roa,
            condition_lirpa=self.condition_lirpa,
            l1=self.l1,
            equilibrium=self.equilibrium,
            formal_positivity=self.formal_positivity,
            scale=self.scale,
            policy_regularization=self.policy_regularization,
            r_factor_fro_norm=self.r_factor_fro_norm,
            steps_per_epoch=np.asarray(self.steps_per_epoch, dtype=np.int64),
            outer_iterations_completed=np.asarray(self.outer_iterations_completed, dtype=np.int64),
            inner_iterations_completed=np.asarray(self.inner_iterations_completed, dtype=np.int64),
        )