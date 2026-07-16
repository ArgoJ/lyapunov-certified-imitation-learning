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
    NeuralLyapunovCandidate,
    LyapunovTrainer,
    ThresholdMonitor,
    FromRolloutsPolicyWrapper,
)
from lcil.lyapunov_learning.counterexample import find_counter_examples
from lcil.utils import GridSearchHelper, MLP, IntegrationMethod, config_field, ArgumentParserConfig
from lcil.utils.lcil_plt.parallel_coodrdinates import parallel_coordinates_plotly
from mpc_datagen import MPCDataset, MPCConfig, mdg_plt

from . import (
    DoubleIntegratorDynamics,
    load_mpc_config,
    load_policy_model,
    compute_riccati_value_matrix,
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
    fix_r_factor: bool = config_field(default=False, help="Whether to fix the R factor in the Lyapunov candidate to 1.0.")
    last_layer_std: float = config_field(default=0.001, help="Standard deviation for the last layer of the Lyapunov feature net.")

def _build_script_defaults() -> LyapunovLearningScriptConfig:
    default_policy_dir = discover_latest_policy_dir()
    return LyapunovLearningScriptConfig(
        policy_dir=str(default_policy_dir),
    )

def _build_training_defaults() -> LyapunovTrainingConfig:
    return LyapunovTrainingConfig(
        state_dim=2,
        state_bounds=np.array([[-10.0, -10.0], [10.0, 10.0]], dtype=float),
        batch_size=1024,
        learning_rate=0.0005,
        outer_epochs=500,
        steps_per_epoch=10,
        policy_epochs=400,
        policy_lr_factor=0.5,
        kappa=0.01,
        seed=1674653,
        regularization_num_samples=1024,
        regularization_resample_interval=100,
        origin_exclusion=[0.05, 0.15],
        bins_per_dim=50,

        condition_weight=10.0,
        roa_weight=0.05,
        condition_lirpa_weight=0.1,
        l1_weight=0.00001,
        scale_weight=0.0,
        equilibrium_weight=0.0,
        formal_positivity_weight=0.0,
        policy_regularization_weight=0.1,
        r_factor_fro_norm_weight=100.0,

        roa_candidate_size=2048,
        rho_estimation_samples=2048,
        rho_growth_gamma=1.1,
        cex_every=10,
        cex_descent_steps=20,
        state_buffer_limit=2048,
        cex_step_size=0.001,
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
        nargs_fields={"activation", "hidden_size", "layers"}
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
    riccati_p = compute_riccati_value_matrix(float(policy_global_config.dt))

    __logger__.info(f"Starting grid search over {len(sweep)} configurations...")
    for run_idx, run in enumerate(sweep):
        script_config, train_config = run.config
        __logger__.info("%s", run.progress_message())

        policy_dir = script_config.policy_dir
        if policy_dir != init_policy_dir:
            __logger__.info(f"Loading policy model from {policy_dir} for this run...")
            policy_model, rollout_dataset, policy_global_config = load_policy_and_dataset(policy_dir, device)
            state_bounds = np.vstack([policy_global_config.constraints.lbx, policy_global_config.constraints.ubx])
            riccati_p = compute_riccati_value_matrix(float(policy_global_config.dt))

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
        lyap_model = NeuralLyapunovCandidate(
            feature_net=lyap_feature,
            state_dim=policy_global_config.nx,
            riccati_p=riccati_p,
            fixed_r_factor=script_config.fix_r_factor,
            feature_last_init_std=script_config.last_layer_std,
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
            final_cex, final_violation = find_counter_examples(
                objective=lambda x: trainer.loss_module.mining_objective(x, train_results.rho_estimate),
                condition_evaluator=lambda x: trainer.loss_module.get_counterexample_mask(x, train_results.rho_estimate),
                config=training_config,
                device=device,
                generator=trainer.torch_gen,
            )
            if final_cex.numel() > 0:
                parallel_coordinates_plotly(
                    states=final_cex.cpu().numpy(),
                    state_bounds=training_config.train_bounds,
                    state_labels=[r"$p$", r"$v$"],
                    origin_exclusion=training_config.origin_exclusion,
                    title="Counterexamples",
                    html_path=(base_path / "counterexamples.html"),
                    cond_violations=final_violation.cpu().numpy(),
                )

    __logger__.info(f"Grid search complete. All results saved to: {sweep.output_root}")

if __name__ == "__main__":
    main()