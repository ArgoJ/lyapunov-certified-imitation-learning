import argparse
import json
import logging

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch as th

from mpc_datagen import MPCDataset, mdg_plt
from lcil.imitation_learning.models import MLPPolicy
from lcil.lyapunov_learning import (
    LyapunovTrainer,
    LyapunovTrainingConfig,
    NeuralLyapunovCandidate,
    ThresholdMonitor,
)
from lcil.utils import ArgumentParserConfig, GridSearchHelper, MLP, config_field, IntegrationMethod

from . import (
    CartpoleAngleWrapper,
    CartpoleDynamics,
    get_mpc_cfg_from_policy_model,
    compute_riccati_value_matrix,
    discover_latest_policy_dir,
    load_policy_model,
)
from ..constants import LYAPUNOV_DIRNAME, LYAPUNOV_ROLLOUT_FILENAME, POLICY_MODEL_FILENAME, POLICY_ROLLOUT_FILENAME
from ..example_utils import build_lyapunov_func

__logger__ = logging.getLogger("lcil.examples.cartpole.learn_lyapunov")

_DEFAULT_TRAIN_BOUND_FACTORS = (0.15, 0.15, 0.12, 0.15)


@dataclass(frozen=True)
class LyapunovLearningScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(help=f"Policy run directory containing {POLICY_MODEL_FILENAME}.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    activation: str = config_field(default="tanh", help="Activation function for the Lyapunov feature net.", display_alias="act")
    hidden_size: int = config_field(default=32, help="Number of neurons in each hidden layer of the Lyapunov feature net.", display_alias="n_hidden")
    layers: int = config_field(default=2, help="Number of hidden layers in the Lyapunov feature net.", display_alias="n_layers")
    use_angle_wrapper: bool = config_field(default=False, help="Whether to use the CartpoleAngleWrapper around the Lyapunov feature net.")
    fix_r_factor: bool = config_field(default=True, help="Whether to fix the R factor in the Lyapunov candidate to 1.0.")
    last_layer_std: float = config_field(default=0.001, help="Standard deviation for the last layer of the Lyapunov feature net.")
    train_bound_factors: list[float] = config_field(
        default_factory=lambda: list(_DEFAULT_TRAIN_BOUND_FACTORS),
        help="Per-dimension scaling applied to policy state bounds before Lyapunov training.",
    )
    curriculum_scales: list[float] = config_field(
        default_factory=lambda: [0.3, 0.5, 0.8, 1.0],
        help=("Curriculum scales applied to the final training bounds."),
    )


def _build_script_defaults() -> LyapunovLearningScriptConfig:
    default_policy_dir = discover_latest_policy_dir()
    return LyapunovLearningScriptConfig(policy_dir=str(default_policy_dir))


def _build_training_defaults() -> LyapunovTrainingConfig:
    return LyapunovTrainingConfig(
        state_dim=4,
        state_bounds=np.array([[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]], dtype=float),
        initial_sample_size=2048,
        batch_size=2048,
        learning_rate=5e-4,
        outer_epochs=500,
        steps_per_epoch=10,
        policy_epochs=400,
        policy_lr_factor=0.5,
        kappa=0.01,
        seed=1674653,
        scale_anchor_num_points=1024,
        scale_anchor_resample_interval=100,
        origin_exclusion=[0.01, 0.01, 0.001, 0.01],
        bins_per_dim=15,

        condition_weight=10.0,
        roa_weight=0.05,
        condition_ibp_weight=0.1,
        l1_weight=0.00001,
        scale_weight=1.0,
        equilibrium_weight=1.0,
        formal_positivity_weight=1.0,
        policy_regularization_weight=0.01,

        roa_candidate_size=8192,
        rho_estimation_samples=8192,
        rho_growth_gamma=1.2,
        cex_every=20,
        cex_steps=30,
        adversarial_samples=2048,
        adversarial_step_size=0.01,
        condition_margin=0.00001,
    )


def parse_cli_args() -> GridSearchHelper[tuple[LyapunovLearningScriptConfig, LyapunovTrainingConfig]]:
    parser = argparse.ArgumentParser(
        description="Train a Lyapunov candidate for a fixed cartpole policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    script_defaults = _build_script_defaults()
    script_defaults.add_to_argparse(
        parser,
        nargs_fields={"activation", "hidden_size", "layers"},
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
        },
    )

    args = parser.parse_args()
    return GridSearchHelper.from_namespace(args, script_defaults, training_defaults)


def _load_policy_and_rollout_dataset(
    policy_dir: Path,
    device: th.device,
) -> tuple[CartpoleAngleWrapper | MLPPolicy, MPCDataset | None]:
    policy_model = load_policy_model(policy_dir, device)
    rollout_dataset_path = policy_dir / POLICY_ROLLOUT_FILENAME
    rollout_dataset = MPCDataset.load(rollout_dataset_path) if rollout_dataset_path.exists() else None
    return policy_model, rollout_dataset


def _scale_state_bounds(state_bounds: np.ndarray, factors: list[float], *, field_name: str) -> np.ndarray:
    factor_array = np.asarray(factors, dtype=float)
    if factor_array.shape != (state_bounds.shape[1],):
        raise ValueError(
            f"{field_name} must contain exactly {state_bounds.shape[1]} entries, got {factor_array.shape[0]}."
        )
    return np.asarray(state_bounds, dtype=float) * factor_array.reshape(1, -1)


def main() -> None:
    sweep = parse_cli_args()

    init_script_config, _ = sweep.configs[0]
    device = th.device(init_script_config.device)
    init_policy_dir = Path(init_script_config.policy_dir).resolve()
    actual_output_root = init_policy_dir / LYAPUNOV_DIRNAME
    sweep.set_output_root(actual_output_root)

    policy_model, rollout_dataset = _load_policy_and_rollout_dataset(init_policy_dir, device)
    mpc_cfg = get_mpc_cfg_from_policy_model(policy_model)
    state_bounds = np.vstack([
        mpc_cfg.constraints.lbx,
        mpc_cfg.constraints.ubx,
    ])
    riccati_p = compute_riccati_value_matrix(float(mpc_cfg.dt))

    __logger__.info("Starting grid search over %d configurations...", len(sweep))
    for run_idx, run in enumerate(sweep):
        script_config, train_config = run.config
        __logger__.info("%s", run.progress_message())

        current_policy_dir = Path(script_config.policy_dir).resolve()
        if current_policy_dir != init_policy_dir:
            __logger__.info("Loading policy model from %s for this run...", current_policy_dir)
            policy_model, rollout_dataset = _load_policy_and_rollout_dataset(current_policy_dir, device)
            mpc_cfg = get_mpc_cfg_from_policy_model(policy_model)
            state_bounds = np.vstack([
                mpc_cfg.constraints.lbx,
                mpc_cfg.constraints.ubx,
            ])
            riccati_p = compute_riccati_value_matrix(float(mpc_cfg.dt))

        training_bounds = _scale_state_bounds(
            state_bounds,
            script_config.train_bound_factors,
            field_name="train_bound_factors",
        )
        curriculum_scales = [float(scale) for scale in script_config.curriculum_scales]
        __logger__.info("Using training bounds:\n%s", training_bounds)
        __logger__.info("Using curriculum scales: %s", curriculum_scales)

        base_path = run.output_dir.resolve()
        dyn_model = CartpoleDynamics(
            dt=mpc_cfg.dt,
            method=IntegrationMethod.EXPLICIT_EULER,
            abcrown_compatible_ops=True,
        )
        dyn_model.eval()

        seed = train_config.seed + run_idx if train_config.seed is not None else None
        lyap_feature = MLP(
            [5 if script_config.use_angle_wrapper else 4] + [script_config.hidden_size] * script_config.layers + [1],
            [script_config.activation] * script_config.layers + ["identity"],
            dropout=train_config.dropout,
            normalization="none",
            seed=seed,
        )
        lyap_model = NeuralLyapunovCandidate(
            feature_net=(CartpoleAngleWrapper(feature_net=lyap_feature) 
                if script_config.use_angle_wrapper else lyap_feature),
            state_dim=mpc_cfg.nx,
            riccati_p=riccati_p,
            fixed_r_factor=script_config.fix_r_factor,
            feature_last_init_std=script_config.last_layer_std,
        )

        training_config = replace(
            train_config,
            state_dim=mpc_cfg.nx,
            state_bounds=training_bounds,
            seed=seed,
            tb_log_dir=actual_output_root / "tb" / sweep.sweep_id / run.run_name,
        )

        trainer = LyapunovTrainer(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            config=training_config,
            rho_monitor=ThresholdMonitor(threshold=1.0, patience=5),
            device=device,
        )

        curriculum_result = trainer.train_with_scaled_bounds(curriculum_scales)
        train_results = curriculum_result.final_result

        trainer.save(base_path)
        if train_results.aborted:
            __logger__.warning("Skipping run %s: %s", run.run_name, train_results.abort_reason)

        curriculum_summary = {
            "curriculum_scales": curriculum_scales,
            "train_bound_factors": list(script_config.train_bound_factors),
            "stages": [
                {
                    "stage_index": stage.stage_index,
                    "scale": stage.scale.tolist(),
                    "state_bounds": stage.state_bounds.tolist(),
                    "rho_estimate": float(stage.result.rho_estimate),
                    "num_mined_counterexamples": int(stage.result.num_mined_counterexamples),
                    "train_time": float(stage.result.train_time),
                }
                for stage in curriculum_result.stages
            ],
        }
        with (base_path / "curriculum_summary.json").open("w", encoding="utf-8") as summary_file:
            json.dump(curriculum_summary, summary_file, indent=2)

        if rollout_dataset is not None:
            lyapunov_func = build_lyapunov_func(lyap_model, device)
            mdg_plt.lyapunov(
                lyapunov_func=lyapunov_func,
                dataset=rollout_dataset[:100],
                state_indices=[0, 1, 2, 3],
                state_labels=[r"$x$", r"$v$", r"$\theta$", r"$\dot{\theta}$"],
                html_path=(base_path / LYAPUNOV_ROLLOUT_FILENAME).with_suffix(".html"),
            )

    __logger__.info("Grid search complete. All results saved to: %s", sweep.output_root)


if __name__ == "__main__":
    main()