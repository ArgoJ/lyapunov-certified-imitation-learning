import argparse
import torch as th
import numpy as np
import logging

from dataclasses import replace, dataclass
from datetime import datetime
from pathlib import Path

from lcil.lyapunov_learning import (
    LyapunovTrainingConfig,
    NeuralLyapunovCandidate,
    LyapunovTrainer,
    ThresholdMonitor,
    FromRolloutsPolicyWrapper,
)
from lcil.utils import GridSearchHelper, lcil_plt, MLP, IntegrationMethod, config_field, ArgumentParserConfig
from mpc_datagen import MPCDataset

from . import (
    DoubleIntegratorDynamics,
    load_policy_model,
    compute_riccati_value_matrix,
    build_lyapunov_func,
    discover_latest_policy_dir,
)
from ..constants import *

__logger__ = logging.getLogger("lcil.examples.double_integrator.learn_lyapunov")



@dataclass(frozen=True)
class LyapunovLearningScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(help=f"Policy run directory containing {POLICY_MODEL_FILENAME}.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    activation: str = config_field(default="relu", help="Activation function for Lyapunov feature net.")
    hidden_size: int = config_field(default=32, help="Number of neurons in each hidden layer of the Lyapunov feature net.")
    layers: int = config_field(default=2, help="Number of hidden layers in the Lyapunov feature net.")

def _build_script_defaults() -> LyapunovLearningScriptConfig:
    default_policy_dir = discover_latest_policy_dir()
    return LyapunovLearningScriptConfig(
        policy_dir=str(default_policy_dir),
    )

def _build_training_defaults() -> LyapunovTrainingConfig:
    return LyapunovTrainingConfig(
        state_dim=2,
        state_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=float),
        initial_sample_size=1000,
        batch_size=512,
        outer_epochs=100,
        steps_per_epoch=5,
        counterexample_every=10,
        train_policy_model=False,
        seed=5912354,
    )


def parse_cli_args() -> tuple[
    LyapunovLearningScriptConfig, 
    GridSearchHelper[LyapunovTrainingConfig]
]:
    """Parse command-line arguments for Lyapunov learning with a fixed policy."""
    parser = argparse.ArgumentParser(
        description="Train and certify a Lyapunov candidate for a fixed double-integrator policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    script_defaults = _build_script_defaults()
    script_defaults.add_to_argparse(parser)
    training_defaults = _build_training_defaults()
    training_defaults.add_to_argparse(
        parser,
        nargs_fields={
            "learning_rate",
            "kappa",
            "invariance_weight",
            "rho_growth_gamma",
            "roa_weight",
            "l1_weight",
            "rho_estimate_quantile",
            "condition_margin",
        }
    )
    
    args = parser.parse_args()
    script_config = script_defaults.from_namespace(args)
    sweep: GridSearchHelper[LyapunovTrainingConfig] = GridSearchHelper.from_namespace(
        training_defaults,
        args,
        output_root=Path(script_config.policy_dir) / LYAPUNOV_DIRNAME,
        sweep_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    return script_config, sweep


def main() -> None:
    script_config, sweep = parse_cli_args()
    device = th.device(script_config.device)

    # Load policy and dynamics once (they don't change across runs)
    policy_path = Path(script_config.policy_dir) / POLICY_MODEL_FILENAME
    policy_model = load_policy_model(policy_path, device)
    policy_global_config = policy_model.global_config
    policy_sequence_length = int(getattr(policy_model, "max_seq_len", 1))
    uses_sequence_policy = policy_sequence_length > 1
    
    dyn_model = DoubleIntegratorDynamics(
        dt=policy_global_config.dt,
        method=IntegrationMethod.EXPLICIT_EULER
    ).to(device)
    dyn_model.eval()
    
    rollout_dataset_path = policy_path.parent / POLICY_ROLLOUT_FILENAME
    rollout_dataset = None
    if rollout_dataset_path.exists():
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
        ).to(device)
    else:
        __logger__.info("Using policy directly for Lyapunov learning.")

    state_bounds = np.vstack([policy_global_config.constraints.lbx, policy_global_config.constraints.ubx])
    riccati_p = compute_riccati_value_matrix(float(policy_global_config.dt))

    __logger__.info(f"Starting grid search over {len(sweep)} configurations...")
    for run_idx, run in enumerate(sweep):
        sweep_config = run.config
        __logger__.info("%s", run.progress_message())
        base_path = run.output_dir.resolve()

        # ---------------------------------------------------------------------
        # 1. Initialize fresh Lyapunov Model
        # ---------------------------------------------------------------------
        seed = sweep_config.seed + run_idx if sweep_config.seed is not None else None
        lyap_feature = MLP(
            [policy_global_config.nx] + [script_config.hidden_size] * script_config.layers + [1],
            [script_config.activation] * script_config.layers + ["identity"],
            dropout=sweep_config.dropout,
            normalization='none',
            seed=seed,
        ).to(device)
        lyap_model = NeuralLyapunovCandidate(
            feature_net=lyap_feature,
            state_dim=policy_global_config.nx,
            eps=1e-3,
            riccati_p=riccati_p,
        ).to(device)

        # ---------------------------------------------------------------------
        # 2. Setup Configs with current grid parameters
        # ---------------------------------------------------------------------
        training_config = replace(
            sweep_config,
            state_dim=policy_global_config.nx,
            state_bounds=state_bounds,
            train_policy_model=False,
            seed=seed,
            tb_log_dir=base_path.parent / "tb",
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
                threshold=1.0,
                patience=5,
            ),
            device=device,
        )
        
        train_results = trainer.train()
        trainer.save(base_path)
        if train_results.aborted:
            __logger__.warning(f"Skipping run {run.run_name}: {train_results.abort_reason}")
            continue

        # ---------------------------------------------------------------------
        # 4. Plot & Save
        # ---------------------------------------------------------------------
        if rollout_dataset is not None:
            lyapunov_func = build_lyapunov_func(lyap_model, device)

            lcil_plt.lyapunov(
                lyapunov_func=lyapunov_func,
                dataset=rollout_dataset[:100],
                state_indices=[0, 1],
                state_labels=["$x$", "$v$"],
                plot_3d=False,
                html_path=(base_path / LYAPUNOV_ROLLOUT_FILENAME).with_suffix(".html"),
            )

    __logger__.info(f"\nGrid search complete. All results saved to: {sweep.sweep_base_path}")


if __name__ == "__main__":
    main()