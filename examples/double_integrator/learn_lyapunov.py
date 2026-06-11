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
from lcil.utils import GridSearchHelper, MLP, IntegrationMethod, config_field, ArgumentParserConfig
from mpc_datagen import MPCDataset, mdg_plt

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
    activation: str = config_field(default="relu", help="Activation function for Lyapunov feature net.", display_alias="act")
    hidden_size: int = config_field(default=32, help="Number of neurons in each hidden layer of the Lyapunov feature net.", display_alias="n_hidden")
    layers: int = config_field(default=2, help="Number of hidden layers in the Lyapunov feature net.", display_alias="n_layers")

def _build_script_defaults() -> LyapunovLearningScriptConfig:
    default_policy_dir = discover_latest_policy_dir()
    return LyapunovLearningScriptConfig(
        policy_dir=str(default_policy_dir),
    )

def _build_training_defaults() -> LyapunovTrainingConfig:
    return LyapunovTrainingConfig(
        state_dim=2,
        state_bounds=np.array([[-10.0, -10.0], [10.0, 10.0]], dtype=float),
        initial_sample_size=1000,
        batch_size=1024,
        outer_epochs=2000,
        steps_per_epoch=5,
        policy_epochs=200,
        cex_every=10,
        seed=1674653,
        kappa=0.01,
        condition_margin=0.00001,
        learning_rate=0.0005,
        condition_weight=10.0,
        condition_ibp_weight=1.0,
        l1_weight=0.00001,
        scale_weight=1.0,
        policy_regularization_weight=1.0,
        scale_anchor_num_points=4096,
        origin_exclusion=[0.05, 0.15],
        bins_per_dim=50
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
            "condition_margin",
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
) -> tuple[nn.Module, MPCDataset | None]:
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

    return policy_model, rollout_dataset


def main() -> None:
    sweep = parse_cli_args()

    init_script_config = sweep.configs[0][0]
    device = th.device(init_script_config.device)

    init_policy_dir = init_script_config.policy_dir
    actual_output_root = Path(init_policy_dir) / LYAPUNOV_DIRNAME
    sweep.set_output_root(actual_output_root)

    policy_model, rollout_dataset = load_policy_and_dataset(init_policy_dir, device)
    policy_global_config = policy_model.global_config
    state_bounds = np.vstack([policy_global_config.constraints.lbx, policy_global_config.constraints.ubx])
    riccati_p = compute_riccati_value_matrix(float(policy_global_config.dt))

    __logger__.info(f"Starting grid search over {len(sweep)} configurations...")
    for run_idx, run in enumerate(sweep):
        script_config, train_config = run.config
        __logger__.info("%s", run.progress_message())

        policy_dir = script_config.policy_dir
        if policy_dir != init_policy_dir:
            __logger__.info(f"Loading policy model from {policy_dir} for this run...")
            policy_model, rollout_dataset = load_policy_and_dataset(policy_dir, device)
            policy_global_config = policy_model.global_config
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
            fixed_r_factor=True,
        )

        # ---------------------------------------------------------------------
        # 2. Setup Configs with current grid parameters
        # ---------------------------------------------------------------------
        training_config = replace(
            train_config,
            state_dim=policy_global_config.nx,
            state_bounds=state_bounds,
            seed=seed,
            tb_log_dir=base_path.parent / "tb" / run.run_name,
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

            mdg_plt.lyapunov(
                lyapunov_func=lyapunov_func,
                dataset=rollout_dataset[:100],
                state_indices=[0, 1],
                state_labels=["$x$", "$v$"],
                plot_3d=False,
                html_path=(base_path / LYAPUNOV_ROLLOUT_FILENAME).with_suffix(".html"),
            )

    __logger__.info(f"\nGrid search complete. All results saved to: {sweep._sweep_base_path}")


if __name__ == "__main__":
    main()