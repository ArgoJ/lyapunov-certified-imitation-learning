import os
from torch.utils.tensorboard import SummaryWriter

from .results import LyapunovTrainingMetrics


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
    tb_writer.add_scalar(rho_str + "RFactorFroNorm", metrics.r_factor_fro_norm[outer_iter], tb_step)
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
        tb_writer.add_scalar(raw_loss_str + "ConditionIBP", metrics.condition_ibp_raw[inner_iter], tb_step)
        tb_writer.add_scalar(raw_loss_str + "L1", metrics.l1_raw[inner_iter], tb_step)
        tb_writer.add_scalar(raw_loss_str + "Equilibrium", metrics.equilibrium_raw[inner_iter], tb_step)
        tb_writer.add_scalar(raw_loss_str + "FormalPositivity", metrics.formal_positivity_raw[inner_iter], tb_step)
        tb_writer.add_scalar(raw_loss_str + "Scale", metrics.scale_raw[inner_iter], tb_step)
        tb_writer.add_scalar(
            raw_loss_str + "PolicyRegularization",
            metrics.policy_regularization_raw[inner_iter],
            tb_step,
        )
        tb_writer.add_scalar(weighted_loss_str + "Total", metrics.loss[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "Condition", metrics.condition[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "Roa", metrics.roa[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "ConditionIBP", metrics.condition_ibp[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "L1", metrics.l1[inner_iter], tb_step)
        tb_writer.add_scalar(weighted_loss_str + "Equilibrium", metrics.equilibrium[inner_iter], tb_step)
        tb_writer.add_scalar(
            weighted_loss_str + "FormalPositivity",
            metrics.formal_positivity[inner_iter],
            tb_step,
        )
        tb_writer.add_scalar(weighted_loss_str + "Scale", metrics.scale[inner_iter], tb_step)
        tb_writer.add_scalar(
            weighted_loss_str + "PolicyRegularization",
            metrics.policy_regularization[inner_iter],
            tb_step,
        )

def tb_writer_close(tb_writer: SummaryWriter | None) -> None:
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

def tb_writer_build(log_dir: os.PathLike | None) -> SummaryWriter | None:
    if log_dir is not None:
        return SummaryWriter(log_dir=log_dir)
    return None