import os
import torch as th
import numpy as np
from numpy.typing import NDArray
from typing import Sequence

from torch.utils.tensorboard import SummaryWriter

from .results import LyapunovTrainingMetrics
from ..utils.lcil_plt.parallel_coodrdinates import parallel_coordinates_matplot


def tb_writer_add_metrics(
    tb_writer: SummaryWriter | None,
    metrics: LyapunovTrainingMetrics,
) -> None:
    if tb_writer is None:
        return 
    
    raw_loss_str = "RawLoss/"
    weighted_loss_str = "WeightedLoss/"
    rho_str = "Rho/"
    cex_str = "Cex/"

    outer_iter = metrics.outer_iterations_completed - 1
    tb_step = metrics.outer_tb_step(outer_iter)
    tb_writer.add_scalar(rho_str + "Estimate", metrics.rho_estimate[outer_iter], tb_step)
    tb_writer.add_scalar(rho_str + "BoundaryQuantile", metrics.rho_boundary_quantile[outer_iter], tb_step)
    tb_writer.add_scalar(rho_str + "BoundaryMean", metrics.rho_boundary_mean[outer_iter], tb_step)
    tb_writer.add_scalar(rho_str + "FeatureTermQuantile", metrics.rho_feature_term_quantile[outer_iter], tb_step)
    tb_writer.add_scalar(rho_str + "LinearTermQuantile", metrics.rho_linear_term_quantile[outer_iter], tb_step)
    tb_writer.add_scalar(rho_str + "FeatureTermMeanShare", metrics.rho_feature_term_mean_share[outer_iter], tb_step)
    tb_writer.add_scalar(rho_str + "LinearTermMeanShare", metrics.rho_linear_term_mean_share[outer_iter], tb_step)
    tb_writer.add_scalar(
        cex_str + "NumMinedCounterexamples",
        metrics.num_mined_counterexamples[outer_iter],
        tb_step,
    )

    end_iter = metrics.inner_iterations_completed
    start_iter = max(0, end_iter - metrics.steps_per_epoch)

    for inner_iter in range(start_iter, end_iter):
        tb_step = metrics.inner_tb_step(inner_iter)
        tb_writer.add_scalar(raw_loss_str + "Condition", metrics.condition_raw[inner_iter], tb_step)
        tb_writer.add_scalar(raw_loss_str + "Roa", metrics.roa_raw[inner_iter], tb_step)
        tb_writer.add_scalar(raw_loss_str + "ConditionLiRPA", metrics.condition_lirpa_raw[inner_iter], tb_step)
        tb_writer.add_scalar(raw_loss_str + "L1", metrics.l1_raw[inner_iter], tb_step)
        tb_writer.add_scalar(raw_loss_str + "Equilibrium", metrics.equilibrium_raw[inner_iter], tb_step)
        tb_writer.add_scalar(raw_loss_str + "FormalPositivity", metrics.formal_positivity_raw[inner_iter], tb_step)
        tb_writer.add_scalar(raw_loss_str + "Scale", metrics.scale_raw[inner_iter], tb_step)
        tb_writer.add_scalar(raw_loss_str + "RFactorFroNorm", metrics.r_factor_fro_norm_raw[inner_iter], tb_step)
        tb_writer.add_scalar(
            raw_loss_str + "PolicyRegularization",
            metrics.policy_regularization_raw[inner_iter],
            tb_step,
        )

        tb_writer.add_scalar(weighted_loss_str + "Total", metrics.loss[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "Condition", metrics.condition[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "Roa", metrics.roa[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "ConditionLiRPA", metrics.condition_lirpa[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "L1", metrics.l1[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "Equilibrium", metrics.equilibrium[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "FormalPositivity", metrics.formal_positivity[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "Scale", metrics.scale[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "RFactorFroNorm", metrics.r_factor_fro_norm[inner_iter], tb_step)
        tb_writer.add_scalar(
            weighted_loss_str + "PolicyRegularization",
            metrics.policy_regularization[inner_iter],
            tb_step,
        )


def tb_writer_add_parallel_coordinates(
    tb_writer: SummaryWriter | None,
    *,
    tag: str,
    global_step: int,
    states: th.Tensor | NDArray | None,
    state_bounds: th.Tensor | NDArray | None = None,
    state_labels: Sequence[str] | None = None,
    state_order: Sequence[int] | None = None,
    origin_exclusion: float | Sequence[float] | None = None,
    max_lines: int = 64,
) -> None:
    if tb_writer is None or states is None:
        return

    if isinstance(states, th.Tensor):
        x_np = states.detach().cpu().numpy()
    else:
        x_np = np.asarray(states)

    bounds_np = None
    if state_bounds is not None:
        if isinstance(state_bounds, th.Tensor):
            bounds_np = state_bounds.detach().cpu().numpy()
        else:
            bounds_np = np.asarray(state_bounds)

    fig = parallel_coordinates_matplot(
        states=x_np,
        state_bounds=bounds_np,
        state_labels=state_labels,
        state_order=state_order,
        origin_exclusion=origin_exclusion,
        max_lines=max_lines,
    )
    tb_writer.add_figure(tag, fig, global_step=global_step, close=True)


def tb_writer_close(tb_writer: SummaryWriter | None) -> None:
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


def tb_writer_build(log_dir: os.PathLike | None) -> SummaryWriter | None:
    if log_dir is not None:
        return SummaryWriter(log_dir=log_dir)
    return None