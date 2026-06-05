import argparse
import json
import logging

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch as th

from mpc_datagen import MPCDataset, mdg_plt
from lcil.lyapunov_learning import (
    LyapunovTrainer,
    LyapunovTrainingConfig,
    NeuralLyapunovCandidate,
    ThresholdMonitor,
)
from lcil.utils import ArgumentParserConfig, GridSearchHelper, MLP, config_field

from . import (
    CartpoleAngleWrapper,
    CartpoleDynamics,
    compute_riccati_value_matrix,
    discover_latest_policy_dir,
    load_policy_model,
)
from ..constants import LYAPUNOV_DIRNAME, LYAPUNOV_ROLLOUT_FILENAME, POLICY_MODEL_FILENAME, POLICY_ROLLOUT_FILENAME
from ..example_utils import build_lyapunov_func

__logger__ = logging.getLogger("lcil.examples.cartpole.learn_lyapunov")

_DEFAULT_TRAIN_BOUND_FACTORS = (0.5, 0.4, 0.15, 0.4)


@dataclass(frozen=True)
class LyapunovLearningScriptConfig(ArgumentParserConfig):
    policy_dir: str = config_field(help=f"Policy run directory containing {POLICY_MODEL_FILENAME}.")
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    activation: str = config_field(default="tanh", help="Activation function for the Lyapunov feature net.", display_alias="act")
    hidden_size: int = config_field(default=32, help="Number of neurons in each hidden layer of the Lyapunov feature net.", display_alias="n_hidden")
    layers: int = config_field(default=3, help="Number of hidden layers in the Lyapunov feature net.", display_alias="n_layers")
    lyap_eps: float = config_field(default=0.1, help="Positive-definite baseline epsilon for the Lyapunov candidate.", display_alias="eps")
    train_bound_factors: list[float] = config_field(
        default_factory=lambda: list(_DEFAULT_TRAIN_BOUND_FACTORS),
        help="Per-dimension scaling applied to policy state bounds before Lyapunov training.",
    )
    curriculum_scales: list[float] = config_field(
        default_factory=lambda: [1.0],
        help=(
            "Curriculum scales applied to the final training bounds. "
            "Example: --curriculum-scales 0.3 0.6 1.0 trains first on 30%, then 60%, then 100% of the configured training box."
        ),
    )


def _build_script_defaults() -> LyapunovLearningScriptConfig:
    default_policy_dir = discover_latest_policy_dir()
    return LyapunovLearningScriptConfig(policy_dir=str(default_policy_dir))


def _build_training_defaults() -> LyapunovTrainingConfig:
    return LyapunovTrainingConfig(
        state_dim=4,
        state_bounds=np.array([[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]], dtype=float),
        initial_sample_size=500,
        batch_size=2048,
        outer_epochs=500,
        steps_per_epoch=10,
        train_policy_model=False,
        counterexample_every=10,
        adversarial_samples=1024,
        counterexample_steps=30,
        adversarial_step_size=0.05,
        condition_margin=0.0,
    )


def parse_cli_args() -> GridSearchHelper[tuple[LyapunovLearningScriptConfig, LyapunovTrainingConfig]]:
    parser = argparse.ArgumentParser(
        description="Train a Lyapunov candidate for a fixed cartpole policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    script_defaults = _build_script_defaults()
    script_defaults.add_to_argparse(
        parser,
        nargs_fields={"activation", "hidden_size", "layers", "lyap_eps"},
    )

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
        },
    )

    args = parser.parse_args()
    return GridSearchHelper.from_namespace(args, script_defaults, training_defaults)


def _load_policy_and_rollout_dataset(
    policy_dir: Path,
    device: th.device,
) -> tuple[CartpoleAngleWrapper, MPCDataset | None]:
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
    policy_global_config = policy_model.net.global_config
    state_bounds = np.vstack([
        policy_global_config.constraints.lbx,
        policy_global_config.constraints.ubx,
    ])
    riccati_p = compute_riccati_value_matrix(float(policy_global_config.dt))

    __logger__.info("Starting grid search over %d configurations...", len(sweep))
    for run_idx, run in enumerate(sweep):
        script_config, train_config = run.config
        __logger__.info("%s", run.progress_message())

        current_policy_dir = Path(script_config.policy_dir).resolve()
        if current_policy_dir != init_policy_dir:
            __logger__.info("Loading policy model from %s for this run...", current_policy_dir)
            policy_model, rollout_dataset = _load_policy_and_rollout_dataset(current_policy_dir, device)
            policy_global_config = policy_model.net.global_config
            state_bounds = np.vstack([
                policy_global_config.constraints.lbx,
                policy_global_config.constraints.ubx,
            ])
            riccati_p = compute_riccati_value_matrix(float(policy_global_config.dt))

        training_bounds = _scale_state_bounds(
            state_bounds,
            script_config.train_bound_factors,
            field_name="train_bound_factors",
        )
        curriculum_scales = [float(scale) for scale in script_config.curriculum_scales]
        __logger__.info("Using training bounds:\n%s", training_bounds)
        __logger__.info("Using curriculum scales: %s", curriculum_scales)

        base_path = run.output_dir.resolve()
        dyn_model = CartpoleDynamics(dt=policy_global_config.dt).to(device)
        dyn_model.eval()

        seed = train_config.seed + run_idx if train_config.seed is not None else None
        lyap_feature = MLP(
            [5] + [script_config.hidden_size] * script_config.layers + [1],
            [script_config.activation] * script_config.layers + ["identity"],
            dropout=train_config.dropout,
            normalization="none",
            seed=seed,
        ).to(device)
        lyap_model = NeuralLyapunovCandidate(
            feature_net=CartpoleAngleWrapper(feature_net=lyap_feature),
            state_dim=policy_global_config.nx,
            eps=script_config.lyap_eps,
            riccati_p=riccati_p,
        ).to(device)

        training_config = replace(
            train_config,
            state_dim=policy_global_config.nx,
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
        if train_results is None:
            __logger__.warning("Skipping run %s: curriculum produced no training result", run.run_name)
            continue

        trainer.save(base_path)
        if train_results.aborted:
            __logger__.warning("Skipping run %s: %s", run.run_name, train_results.abort_reason)
            continue

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