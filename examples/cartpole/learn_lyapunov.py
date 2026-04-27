import argparse
import json
import torch as th
import numpy as np

from dataclasses import replace
from pathlib import Path
from numpy.typing import NDArray

from lcil.lyapunov_learning import (
    GridSearchHelper,
    LyapunovTrainingConfig,
    NeuralLyapunovCandidate,
    LyapunovTrainer,
    ThresholdMonitor,
)
from lcil.certification import LyapunovCertificationConfig, ABCrownCertifier, CertificationResultTester
from lcil.utils import lcil_plt, ICNN, MLP
from lcil.imitation_learning_mlp import MLPPolicy
from mpc_datagen import MPCDataset

from cartpole_dyn import CartpoleDynamics
from model import CartpoleAngleWrapper


def parse_cli_args(
    training_defaults: LyapunovTrainingConfig,
    certification_defaults: LyapunovCertificationConfig,
) -> argparse.Namespace:
    """Parse command-line arguments for Lyapunov learning with a fixed policy."""
    parser = argparse.ArgumentParser(
        description="Train and certify a Lyapunov candidate for a fixed inverted pendulum on cart policy."
    )
    parser.add_argument(
        "--policy-path", type=str, default="results/inverted_pendulum_on_cart/20260318_114317/model.pt",
        help="Path to the trained fixed policy model checkpoint.")
    
    # Model Parameters
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string.")
    parser.add_argument("--num_neurons", type=int, default=32, help="Number of neurons per hidden layer.")
    parser.add_argument(
        "--lyap-eps", type=float, default=0.1,
        help="Epsilon used in NeuralLyapunovCandidate for positive-definite baseline term.")
    parser.add_argument(
        "--training-bound-scales",
        nargs='+',
        type=float,
        default=[1.0],
        help=(
            "Curriculum scales applied to the final training bounds. "
            "Example: --training-bound-scales 0.3 0.6 1.0 trains first on 30%%, then 60%%, then 100%% of the configured training box."
        ),
    )

    # Training Parameters
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
    
    # Certification Parameters
    certification_defaults.add_to_argparse(
        parser,
        prefix="cert-",
        nargs_fields={
            "max_recursion_depth",
        },
    )
    parser.add_argument(
        "--test-rollout-steps",
        type=int,
        default=50,
        help="Closed-loop rollout steps for empirical certification-result testing.",
    )
    parser.add_argument(
        "--skip-certification",
        action="store_true",
        help="Skip certification/testing/plotting and run training only.",
    )
    return parser.parse_args()


def main() -> None:
    training_defaults = LyapunovTrainingConfig(
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
    certification_defaults = LyapunovCertificationConfig(
        state_dim=4,
        cert_bounds=np.array([[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]], dtype=float),
        rho_scaling=1.2,
        bisection_tol=1e-2,
        max_scale_steps=15,
        max_bisection_steps=20,
        cert_method="alpha-crown",
    )
    args = parse_cli_args(training_defaults, certification_defaults)
    device = th.device(args.device)

    # Load policy and dynamics once (they don't change across runs)
    policy_path = Path(args.policy_path)
    feature_net = MLPPolicy.load(
        path=policy_path,
        map_location=device,
    )
    policy_model = CartpoleAngleWrapper(feature_net=feature_net).to(device)
    policy_model.eval()

    lyap_path = policy_path.parent / "lyapunov"
    
    dyn_model = CartpoleDynamics(dt=feature_net.global_config.dt).to(device)
    dyn_model.eval()
    
    rollout_dataset_path = policy_path.parent / "policy_rollouts.hdf5"
    rollout_dataset = None
    if rollout_dataset_path.exists():
        rollout_dataset = MPCDataset.load(rollout_dataset_path)

    state_bounds = np.vstack([feature_net.global_config.constraints.lbx, feature_net.global_config.constraints.ubx])

    training_bounds_percentage = np.array([0.5, 0.4, 0.15, 0.4])
    cert_percentage = np.array([0.4, 0.3, 0.1, 0.3])
    cert_percentage = np.array([0.15, 0.15, 0.05, 0.15]) # Temp
    
    training_bounds = state_bounds * training_bounds_percentage[:, None].T
    cert_bounds = state_bounds * cert_percentage[:, None].T
    curriculum_scales = [float(scale) for scale in args.training_bound_scales]
    print(f"State bounds:\n{state_bounds}")
    print(f"Using training bounds:\n{training_bounds}")
    print(f"Using certification bounds:\n{cert_bounds}")
    print(f"Using training bound scales: {curriculum_scales}")

    sweep: GridSearchHelper[LyapunovTrainingConfig] = GridSearchHelper.from_namespace(
        training_defaults,
        args,
        output_root=lyap_path,
        field_aliases={
            "learning_rate": "lr",
            "invariance_weight": "invw",
            "rho_growth_gamma": "rhog",
            "roa_weight": "roaw",
            "l1_weight": "l1w",
            "rho_estimate_quantile": "rhoq",
            "condition_margin": "margin",
            "training_bound_scales": "curr",
            "max_recursion_depth": "recursion",
        },
        extra_name_parts={
            "training_bound_scales": curriculum_scales,
        },
    )
    certification_base_config = certification_defaults.from_namespace(args, prefix="cert-")

    print(f"Starting grid search over {len(sweep)} configurations...")

    for run_idx, run in enumerate(sweep):
        sweep_config = run.config
        print(f"\n{run.progress_message()}, train_policy: {sweep_config.train_policy_model}")
        base_path = run.output_dir

        # ---------------------------------------------------------------------
        # 1. Initialize fresh Lyapunov Model (so it trains from scratch)
        # ---------------------------------------------------------------------
        lyap_feature = MLP([5, 32, 32, 32, 1], ["tanh", "tanh", "tanh", "identity"]).to(device)
        lyap_wrapper = CartpoleAngleWrapper(feature_net=lyap_feature)
        lyap_model = NeuralLyapunovCandidate(
            feature_net=lyap_wrapper,
            state_dim=4,
            eps=args.lyap_eps,
        ).to(device)

        # ---------------------------------------------------------------------
        # 2. Setup Configs with current grid parameters
        # ---------------------------------------------------------------------
        training_config = replace(
            sweep_config,
            state_dim=feature_net.global_config.nx,
            state_bounds=training_bounds,
            seed=sweep_config.seed + run_idx if sweep_config.seed is not None else None,
            tb_log_dir=lyap_path / "tb" / sweep.sweep_id / run.run_name,
        )

        certification_config = LyapunovCertificationConfig.from_training_config(
            training_config,
            bins_per_dim=args.cert_bins_per_dim,
            center_refinement_factor=args.cert_center_refinement_factor,
            origin_exclusion=certification_base_config.origin_exclusion,
            rho_scaling=certification_base_config.rho_scaling,
            bisection_tol=certification_base_config.bisection_tol,
            max_scale_steps=certification_base_config.max_scale_steps,
            max_bisection_steps=certification_base_config.max_bisection_steps,
            cert_method=certification_base_config.cert_method,
            sublevel_tolerance=certification_base_config.sublevel_tolerance,
            condition_margin=certification_base_config.condition_margin,
            suppress_native_output=certification_base_config.suppress_native_output,
            use_ibp_filter=certification_base_config.use_ibp_filter,
            batch_size=certification_base_config.batch_size,
            max_recursion_depth=certification_base_config.max_recursion_depth,
            cert_bounds=cert_bounds,
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
        
        curriculum_result = trainer.train_with_scaled_bounds(curriculum_scales)
        train_results = curriculum_result.final_result
        if train_results is None:
            print(f"Skipping run {run.run_name}: curriculum produced no training result")
            continue
        if train_results.aborted:
            print(f"Skipping run {run.run_name}: {train_results.abort_reason}")
            continue
        trainer.save(base_path)

        curriculum_summary = {
            "training_bound_scales": curriculum_scales,
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

        if args.skip_certification:
            continue

        # ---------------------------------------------------------------------
        # 4. Certify
        # ---------------------------------------------------------------------
        certifier = ABCrownCertifier(
            policy_model,
            lyap_model,
            dyn_model,
            certification_config,
            device,
        )
        cert_results = certifier.certify(max(train_results.rho_estimate, 1e-3))
        certifier.save(base_path)

        cert_tester = CertificationResultTester(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            config=certification_config,
            device=device,
        )
        test_results = cert_tester.test_result(
            cert_result=cert_results,
            rollout_steps=args.test_rollout_steps,
        )
        test_results.save(base_path / "certification_tester_results.json")

        # ---------------------------------------------------------------------
        # 5. Plot & Save
        # ---------------------------------------------------------------------
        state_labels = [r"$x$", r"$v$", r"$\theta$", r"$\dot{\theta}$"]
        if rollout_dataset is not None:
            def lyapunov_func(states: NDArray) -> NDArray:
                x = th.as_tensor(states, dtype=th.float32, device=device)
                with th.no_grad():
                    v = lyap_model(x)
                return v.detach().cpu().numpy().reshape(-1)

            lcil_plt.lyapunov_cert_regions(
                lyapunov_func=lyapunov_func,
                certification_result=cert_results,
                dataset=rollout_dataset[:100],
                state_labels=state_labels,
                html_path=base_path / "lyapunov_plot.html",
            )

        lcil_plt.certified_regions_2d(
            certification_result=cert_results,
            state_labels=state_labels,
            html_path=base_path / "certified_regions.html",
        )

    print(f"\nGrid search complete. All results saved to: {sweep.sweep_base_path}")


if __name__ == "__main__":
    main()