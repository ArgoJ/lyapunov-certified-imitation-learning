import argparse
import torch as th
import numpy as np
import logging

from dataclasses import replace, dataclass
from datetime import datetime
from pathlib import Path
from torch import nn

from lcil.lyapunov_learning import (
    LyapunovTrainingConfig,
    NeuralLinearLyapunovCandidate,
    NeuralQuadraticLyapunovCandidate,
    LyapunovTrainer,
    ThresholdMonitor,
    FromRolloutsPolicyWrapper,
    compute_closed_loop_jacobian,
    compute_closed_loop_lyapunov_matrix,
    compute_polyhedral_value_matrix,
    calculate_r_factor_from_riccati,
)
from lcil.utils import GridSearchHelper, MLP, IntegrationMethod, config_field, ArgumentParserConfig
from lcil.utils.lcil_plt.parallel_coodrdinates import parallel_coordinates_plotly
from mpc_datagen import MPCDataset, MPCConfig, mdg_plt

from . import (
    DoubleIntegratorDynamics,
    Q,
    load_mpc_config,
    load_policy_model,
    build_lyapunov_func,
    discover_latest_policy_dir,
)
from ..constants import (
    LYAPUNOV_DIRNAME,
    POLICY_MODEL_FILENAME,
    POLICY_ROLLOUT_FILENAME,
    LYAPUNOV_ROLLOUT_FILENAME,
)

__logger__ = logging.getLogger("lcil.examples.double_integrator.learn_lyapunov")



@dataclass(frozen=True)
class LyapunovLearningScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(help=f"Policy run directory containing {POLICY_MODEL_FILENAME}.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    activation: str = config_field(default="relu", help="Activation function for Lyapunov feature net.", display_alias="act")
    hidden_size: int = config_field(default=32, help="Number of neurons in each hidden layer of the Lyapunov feature net.", display_alias="n_hidden")
    layers: int = config_field(default=2, help="Number of hidden layers in the Lyapunov feature net.", display_alias="n_layers")
    candidate_type: str = config_field(
        default="quadratic",
        help="Type of residual Lyapunov candidate: 'quadratic' (ellipsoidal delta^T P delta) or 'linear' (polyhedral ||P delta||_1).",
        display_alias="candidate",
    )
    fix_r_factor: bool = config_field(default=True, help="Whether to fix the R factor in the Lyapunov candidate.")
    riccati_scale: str | float = config_field(
        default="spectral",
        help="How to scale the Riccati P matrix before seeding R. "
             "'none' = use raw P, 'spectral' = divide by spectral norm, "
             "'frobenius' = divide by Frobenius norm, or a float for a custom divisor.",
    )
    eps: float = config_field(default=0.0, help="Epsilon added to the Riccati matrix P inside NeuralResidualLyapunovCandidate.")

def _build_script_defaults() -> LyapunovLearningScriptConfig:
    default_policy_dir = discover_latest_policy_dir()
    return LyapunovLearningScriptConfig(
        policy_dir=str(default_policy_dir),
    )

def _build_training_defaults() -> LyapunovTrainingConfig:
    return LyapunovTrainingConfig(
        state_dim=2,
        state_bounds=np.array([[-10.0, -10.0], [10.0, 10.0]], dtype=float),
        batch_size=2048,
        learning_rate=0.001,
        outer_epochs=2000,
        steps_per_epoch=20,
        policy_epochs=200,
        policy_lr_factor=0.05,
        policy_update_interval=2,
        kappa=0.01,
        seed=1674653,
        regularization_num_samples=4096,
        regularization_resample_interval=20,
        origin_exclusion=[0.05, 0.15],
        bins_per_dim=50,

        condition_weight=100.0,
        roa_weight=5.0,
        condition_lirpa_weight=15.0,
        l1_weight=0.0,
        weight_decay=0.0,
        scale_weight=0.0,
        equilibrium_weight=0.0,
        formal_positivity_weight=0.0,
        policy_regularization_weight=1.0,
        r_factor_regularization_weight=100.0,
        condition_margin=0.005,

        roa_candidate_size=4096,
        rho_estimation_samples=8192,
        rho_estimate_quantile=0.005,
        rho_descent_steps=10,
        rho_growth_gamma=1.2,
        rho_ema_decay=0.9,
        rho_resample_margin=1.3,
        cex_every=2,
        cex_max_age=3,
        cex_fraction_max=0.4,
        cex_descent_steps=20,
        state_buffer_limit=16384,
        cex_step_size=0.005,
        cex_axis_antiphase_samples=512,
        cex_eigenvector_antiphase_samples=256,
        cex_axis_scale_factor=1.0,
        enable_diagnosis=True,
    )


def parse_cli_args() -> GridSearchHelper[tuple[LyapunovLearningScriptConfig, LyapunovTrainingConfig]]:
    """Parse command-line arguments for Lyapunov learning with a fixed policy."""
    parser = argparse.ArgumentParser(
        description="Train and certify a Lyapunov candidate for a fixed double-integrator policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    script_defaults = _build_script_defaults()
    script_defaults.add_to_argparse(
        parser,
        nargs_fields={"activation", "hidden_size", "layers", "candidate_type"}
    )
    training_defaults = _build_training_defaults()
    training_defaults.add_to_argparse(
        parser,
        nargs_fields={
            "learning_rate",
            "kappa",
            "*weight",
            "rho_growth_gamma",
            "rho_estimate_quantile",
            "policy_epochs",
        }
    )
    
    args = parser.parse_args()
    sweep: GridSearchHelper[tuple[LyapunovLearningScriptConfig, LyapunovTrainingConfig]] = GridSearchHelper.from_namespace(
        args,
        script_defaults,
        training_defaults,
        sweep_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    return sweep


def load_policy_and_dataset(
    policy_dir: Path | str, 
    device: th.device
) -> tuple[nn.Module, MPCDataset | None, MPCConfig]:
    policy_dir = Path(policy_dir)
    policy_path = policy_dir / POLICY_MODEL_FILENAME
    
    __logger__.info(f"Loading policy model from {policy_path}...")
    policy_model = load_policy_model(policy_path, device)
    
    policy_sequence_length = int(getattr(policy_model, "max_seq_len", 1))
    uses_sequence_policy = policy_sequence_length > 1
    
    rollout_dataset_path = policy_dir / POLICY_ROLLOUT_FILENAME
    rollout_dataset = None
    
    if rollout_dataset_path.exists():
        __logger__.info(f"Loading rollout dataset from {rollout_dataset_path}...")
        rollout_dataset = MPCDataset.load(rollout_dataset_path)

    if uses_sequence_policy:
        if rollout_dataset is None:
            raise FileNotFoundError(
                "Sequence policy detected but no rollout dataset was found at "
                f"'{rollout_dataset_path}'. Run the policy rollout first so Lyapunov learning "
                "can reconstruct transformer history with FromRolloutsPolicyWrapper."
            )
        __logger__.info(
            "Sequence policy detected (max_seq_len=%d). Wrapping policy with rollout-based history.",
            policy_sequence_length,
        )
        policy_model = FromRolloutsPolicyWrapper.from_rollouts(
            policy=policy_model,
            rollout_source=rollout_dataset,
            sequence_length=policy_sequence_length,
        )
    else:
        __logger__.info("Standard policy detected. Using directly for Lyapunov learning.")

    return policy_model, rollout_dataset, load_mpc_config(policy_dir)


def main() -> None:
    sweep = parse_cli_args()

    init_script_config = sweep.configs[0][0]
    device = th.device(init_script_config.device)

    init_policy_dir = init_script_config.policy_dir
    actual_output_root = Path(init_policy_dir) / LYAPUNOV_DIRNAME
    sweep.set_output_root(actual_output_root)

    policy_model, rollout_dataset, policy_global_config = load_policy_and_dataset(init_policy_dir, device)
    state_bounds = np.vstack([policy_global_config.constraints.lbx, policy_global_config.constraints.ubx])

    __logger__.info(f"Starting grid search over {len(sweep)} configurations...")
    for run_idx, run in enumerate(sweep):
        script_config, train_config = run.config
        __logger__.info("%s", run.progress_message())

        policy_dir = script_config.policy_dir
        if policy_dir != init_policy_dir:
            __logger__.info(f"Loading policy model from {policy_dir} for this run...")
            policy_model, rollout_dataset, policy_global_config = load_policy_and_dataset(policy_dir, device)
            state_bounds = np.vstack([policy_global_config.constraints.lbx, policy_global_config.constraints.ubx])

        base_path = run.output_dir.resolve()
        dyn_model = DoubleIntegratorDynamics(
            dt=policy_global_config.dt,
            method=IntegrationMethod.EXPLICIT_EULER,
            abcrown_compatible_ops=True,
        )
        dyn_model.eval()

        # ---------------------------------------------------------------------
        # 1. Initialize fresh Lyapunov Model
        # ---------------------------------------------------------------------
        seed = train_config.seed + run_idx if train_config.seed is not None else None
        lyap_feature = MLP(
            [policy_global_config.nx] + [script_config.hidden_size] * script_config.layers + [1],
            [script_config.activation] * script_config.layers + ["identity"],
            dropout=train_config.dropout,
            normalization='none',
            seed=seed,
        )
        candidate_type = script_config.candidate_type.strip().lower()
        if candidate_type == "quadratic":
            candidate_cls = NeuralQuadraticLyapunovCandidate
        elif candidate_type == "linear":
            candidate_cls = NeuralLinearLyapunovCandidate
        else:
            raise ValueError(
                f"Unknown candidate_type '{script_config.candidate_type}'. "
                "Expected 'quadratic' or 'linear'."
            )

        a_cl = compute_closed_loop_jacobian(
            dyn_model, policy_model, state_dim=policy_global_config.nx, device=device
        ).detach().cpu().numpy()

        q_matrix = (
            policy_global_config.cost.W[:policy_global_config.nx, :policy_global_config.nx]
            if hasattr(policy_global_config, "cost") and hasattr(policy_global_config.cost, "W")
            else Q
        )
        closed_loop_p = compute_closed_loop_lyapunov_matrix(
            a_closed_loop=a_cl,
            q_matrix=q_matrix,
            kappa=train_config.kappa,
            scale_mode=script_config.riccati_scale,
        )

        if candidate_type == "linear":
            seed_p = compute_polyhedral_value_matrix(
                a_closed_loop=a_cl,
                p_initial=closed_loop_p,
                kappa=train_config.kappa,
                scale_mode=script_config.riccati_scale,
            )
        else:
            seed_p = closed_loop_p
            __logger__.info("Seeded quadratic Lyapunov matrix P from closed-loop Jacobian A_cl.")

        eps_init = 0.0 if script_config.fix_r_factor else script_config.eps
        r_init, adjusted_eps = calculate_r_factor_from_riccati(seed_p, eps=eps_init)
        lyap_model = candidate_cls(
            feature_net=lyap_feature,
            state_dim=policy_global_config.nx,
            eps=adjusted_eps,
            r_factor=r_init,
            fixed_r_factor=script_config.fix_r_factor,
            kappa=train_config.kappa,
        )

        # ---------------------------------------------------------------------
        # 2. Setup Configs with current grid parameters
        # ---------------------------------------------------------------------
        training_config = replace(
            train_config,
            state_dim=policy_global_config.nx,
            state_bounds=state_bounds,
            seed=seed,
            tb_log_dir=actual_output_root / "tb" / sweep.sweep_id / run.run_name,
        )

        # ---------------------------------------------------------------------
        # 3. Train
        # ---------------------------------------------------------------------
        trainer = LyapunovTrainer(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            config=training_config,
            rho_monitor=ThresholdMonitor(
                threshold=training_config.rho_min,
                patience=5,
            ),
            device=device,
        )
        
        train_results = trainer.train()
        trainer.save(base_path, mpc_config=policy_global_config)
        if train_results.aborted:
            __logger__.warning(f"Skipping run {run.run_name}: {train_results.abort_reason}")
            continue

        # ---------------------------------------------------------------------
        # 4. Plot & Save
        # ---------------------------------------------------------------------
        if rollout_dataset is not None:
            lyapunov_func = build_lyapunov_func(lyap_model, device)
            mdg_plt.lyapunov(
                lyapunov_func=lyapunov_func,
                dataset=rollout_dataset[:100],
                state_indices=[0, 1],
                state_labels=[r"$p$", r"$v$"],
                html_path=(base_path / LYAPUNOV_ROLLOUT_FILENAME).with_suffix(".html")
            )
            
        if not train_results.aborted:
            __logger__.info("Mining final counterexamples for visualization...")
            eval_rho_diag, _ = trainer.estimate_rho(
                gamma=1.0, 
                estimate_quantile=0.0,
            )
            final_rho = eval_rho_diag.rho.rho
            __logger__.info(
                "Estimated final rho without margins: %.6f (trained EMA: %.6f)",
                final_rho,
                train_results.rho_estimate,
            )
            final_cex, final_violations = trainer.mine_counterexamples(
                rho_estimate=final_rho,
                with_margin=False,
            )
            if final_cex.numel() > 0:
                parallel_coordinates_plotly(
                    states=final_cex.cpu().numpy(),
                    state_bounds=training_config.train_bounds,
                    state_labels=[r"$p$", r"$v$"],
                    origin_exclusion=training_config.origin_exclusion,
                    title="Counterexamples",
                    html_path=(base_path / "final_counterexamples.html"),
                    cond_violations=final_violations.cpu().numpy(),
                )

    __logger__.info(f"Grid search complete. All results saved to: {sweep.output_root}")

if __name__ == "__main__":
    main()