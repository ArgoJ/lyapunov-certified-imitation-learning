import argparse
import torch as th
import numpy as np
import logging

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from lcil.lyapunov_learning import (
    LyapunovTrainingConfig,
    NeuralLyapunovCandidate,
    LyapunovTrainer,
    ThresholdMonitor,
    FromRolloutsPolicyWrapper,
)
from lcil.utils import GridSearchHelper, lcil_plt, MLP, IntegrationMethod
from mpc_datagen import MPCDataset

from . import (
    DoubleIntegratorDynamics,
    default_model_path,
    load_policy_model,
    compute_riccati_value_matrix,
)
from ..constants import *

__logger__ = logging.getLogger("lcil.examples.double_integrator.learn_lyapunov")


def parse_cli_args(
    training_defaults: LyapunovTrainingConfig,
) -> argparse.Namespace:
    """Parse command-line arguments for Lyapunov learning with a fixed policy."""
    parser = argparse.ArgumentParser(
        description="Train and certify a Lyapunov candidate for a fixed double-integrator policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=default_model_path(),
        help="Path to the trained fixed policy model checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string.")

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
    
    
    return parser.parse_args()


def main() -> None:
    training_defaults = LyapunovTrainingConfig(
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
    args = parse_cli_args(training_defaults)
    device = th.device(args.device)

    # Load policy and dynamics once (they don't change across runs)
    policy_path = Path(args.policy_path)
    policy_model = load_policy_model(policy_path, device)
    policy_global_config = policy_model.global_config
    policy_sequence_length = int(getattr(policy_model, "max_seq_len", 1))
    uses_sequence_policy = policy_sequence_length > 1

    sweep_output_root = policy_path.parent / LYAPUNOV_DIRNAME
    
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
    __logger__.info("Using Riccati value matrix to seed the Lyapunov R factor:\n%s", riccati_p)

    sweep: GridSearchHelper[LyapunovTrainingConfig] = GridSearchHelper.from_namespace(
        training_defaults,
        args,
        output_root=sweep_output_root,
        sweep_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    __logger__.info(f"Starting grid search over {len(sweep)} configurations...")

    for run_idx, run in enumerate(sweep):
        sweep_config = run.config
        __logger__.info("%s", run.progress_message())
        base_path = run.output_dir.resolve()

        # ---------------------------------------------------------------------
        # 1. Initialize fresh Lyapunov Model
        # ---------------------------------------------------------------------
        lyap_feature = MLP([2, 32, 32, 1], ["relu", "relu", "identity"]).to(device)
        lyap_model = NeuralLyapunovCandidate(
            feature_net=lyap_feature,
            state_dim=2,
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
            seed=sweep_config.seed + run_idx if sweep_config.seed is not None else None,
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
            def lyapunov_func(states: np.ndarray) -> np.ndarray:
                x = th.as_tensor(states, dtype=th.float32, device=device)
                with th.no_grad():
                    v = lyap_model(x)
                return v.detach().cpu().numpy().reshape(-1)

            lcil_plt.lyapunov(
                lyapunov_func=lyapunov_func,
                dataset=rollout_dataset[:100],
                state_indices=[0, 1],
                state_labels=["$x$", "$v$"],
                plot_3d=False,
                html_path=base_path / LYAPUNOV_ROLLOUT_FILENAME.replace(".hdf5", ".html"),
            )

    __logger__.info(f"\nGrid search complete. All results saved to: {sweep.sweep_base_path}")


if __name__ == "__main__":
    main()