import argparse
import itertools
import torch as th
import numpy as np

from datetime import datetime
from pathlib import Path
from numpy.typing import NDArray

from lcil.lyapunov_learning import LyapunovTrainingConfig, NeuralLyapunovCandidate, LyapunovTrainer
from lcil.certification import LyapunovCertificationConfig, ABCrownCertifier, CertificationResultTester
from lcil.utils import lcil_plt, ICNN, MLP
from lcil.imitation_learning_mlp import MLPPolicy
from mpc_datagen import MPCDataset

from cartpole_dyn import CartpoleDynamics
from model import CartpoleAngleWrapper


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for Lyapunov learning with a fixed policy."""
    parser = argparse.ArgumentParser(
        description="Train and certify a Lyapunov candidate for a fixed inverted pendulum on cart policy."
    )
    parser.add_argument(
        "--policy-path", type=str, default="results/inverted_pendulum_on_cart/20260318_114317/model.pt",
        help="Path to the trained fixed policy model checkpoint.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string.")
    parser.add_argument("--num_neurons", type=int, default=32, help="Number of neurons per hidden layer.")
    parser.add_argument("--initial-sample-size", type=int, default=500, help="Training sample size.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Training batch size.")
    parser.add_argument("--outer-epochs", type=int, default=500, help="Number of outer epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=10, help="Gradient steps per epoch.")
    parser.add_argument(
        "--train-policy-model",
        action="store_true",
        help="Jointly train policy and Lyapunov model (default: only Lyapunov model).",
    )
    parser.add_argument(
        "--counterexample-every", type=int, default=10,
        help="Counterexample search interval in epochs.")
    parser.add_argument(
        "--adversarial-samples", type=int, default=1024,
        help="PGD seed states used in each counterexample mining phase.")
    parser.add_argument(
        "--counterexample-steps", type=int, default=30,
        help="Projected gradient steps used during counterexample mining.")
    parser.add_argument(
        "--adversarial-step-size", type=float, default=0.05,
        help="Relative PGD step size for counterexample mining.")
    parser.add_argument(
        "--condition-margin", type=float, default=0.0,
        help="Training margin for the relaxed Lyapunov condition (>=0 is stricter).")
    parser.add_argument(
        "--lyap-eps", type=float, default=0.1,
        help="Epsilon used in NeuralLyapunovCandidate for positive-definite baseline term.")
    
    # Grid Search Parameters (accept multiple values)
    parser.add_argument("--learning-rate", nargs='+', type=float, default=[5e-3], help="Optimizer learning rate(s).")
    parser.add_argument("--kappa", nargs='+', type=float, default=[0.012], help="Lyapunov decrease margin kappa(s).")
    parser.add_argument("--invariance-weight", nargs='+', type=float, default=[1.0], help="Invariance loss weight(s).")
    parser.add_argument("--rho-growth-gamma", nargs='+', type=float, default=[1.5], help="ROA rho growth factor(s).")
    parser.add_argument("--roa-weight", nargs='+', type=float, default=[2.0], help="ROA objective weight(s).")
    parser.add_argument("--l1-weight", nargs='+', type=float, default=[1e-5], help="L1 regularization weight(s).")
    parser.add_argument(
        "--rho-estimate-quantile",
        nargs='+',
        type=float,
        default=[0.3],
        help="Quantile(s) used for robust boundary aggregation in rho estimation.",
    )

    parser.add_argument("--seed", type=int, default=5912354, help="Random seed.")
    
    # Certification Parameters
    parser.add_argument(
        "--cert-bins-per-dim", nargs='+', type=int, default=[2, 6, 10, 10], 
        help="Initial certification bins per dimension.")
    parser.add_argument(
        "--cert-center-refinement-factor", nargs='+', type=float, default=[1.0, 0.8, 0.4, 0.4], 
        help="Factor for narrowing region around the center.")
    parser.add_argument(
        "--cert-rho-scaling", type=float, default=1.2, 
        help="Certification rho scaling.")
    parser.add_argument(
        "--cert-bisection-tol", type=float, default=1e-2, 
        help="Certification bisection tolerance.")
    parser.add_argument(
        "--cert-max-scale-steps", type=int, default=15, 
        help="Maximum scale expansion steps during certification.",)
    parser.add_argument(
        "--cert-max-bisection-steps", type=int, default=20, 
        help="Maximum bisection steps during certification.")
    parser.add_argument(
        "--cert-method",
        type=str,
        default="alpha-crown",
        choices=["alpha-crown", "crown", "crown-ibp", "ibp"],
        help="AutoLiRPA certification backend method.",
    )
    parser.add_argument(
        "--disable-ibp-filter",
        action="store_true",
        help="Disable IBP pre-filtering and certify all regions directly.",
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
    args = parse_cli_args()
    device = th.device(args.device)

    # Load policy and dynamics once (they don't change across runs)
    policy_path = Path(args.policy_path)
    feature_net = MLPPolicy.load(
        path=policy_path,
        map_location=device,
    )
    policy_model = CartpoleAngleWrapper(feature_net=feature_net).to(device)
    policy_model.eval()

    # Parent directory for this entire grid search sweep
    iso = datetime.now().strftime("%Y%m%d_%H%M%S")
    lyap_path = policy_path.parent / "lyapunov"
    sweep_base_path = lyap_path / iso
    sweep_base_path.mkdir(parents=True, exist_ok=True)
    
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
    print(f"State bounds:\n{state_bounds}")
    print(f"Using training bounds:\n{training_bounds}")
    print(f"Using certification bounds:\n{cert_bounds}")

    # Generate all combinations for the grid search
    grid = list(itertools.product(
        args.learning_rate,
        args.kappa,
        args.invariance_weight,
        args.rho_growth_gamma,
        args.roa_weight,
        args.l1_weight,
        args.rho_estimate_quantile,
    ))

    print(f"Starting grid search over {len(grid)} configurations...")

    for run_idx, (lr, kappa, inv_w, rho_gamma, roa_w, l1_w, rho_q) in enumerate(grid):
        print(f"\n[{run_idx+1}/{len(grid)}] Running config -> "
              f"lr: {lr}, kappa: {kappa}, inv_w: {inv_w}, rho_g: {rho_gamma}, "
              f"roa_w: {roa_w}, l1_w: {l1_w}, rho_q: {rho_q}, "
              f"margin: {args.condition_margin}, lyap_eps: {args.lyap_eps}, "
              f"train_policy: {args.train_policy_model}")
        
        # Create a specific folder for this parameter combination
        run_name = (
            f"lr_{lr}__kappa_{kappa}__invw_{inv_w}__rhog_{rho_gamma}"
            f"__roaw_{roa_w}__l1w_{l1_w}__rhoq_{rho_q}"
            f"__margin_{args.condition_margin}__eps_{args.lyap_eps}"
        )
        base_path = sweep_base_path / run_name
        base_path.mkdir(parents=True, exist_ok=True)

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
        training_config = LyapunovTrainingConfig(
            state_dim=feature_net.global_config.nx,
            state_bounds=training_bounds,
            initial_sample_size=args.initial_sample_size,
            batch_size=args.batch_size,
            outer_epochs=args.outer_epochs,
            steps_per_epoch=args.steps_per_epoch,
            counterexample_every=args.counterexample_every,
            adversarial_samples=args.adversarial_samples,
            counterexample_steps=args.counterexample_steps,
            adversarial_step_size=args.adversarial_step_size,
            train_policy_model=args.train_policy_model,
            seed=args.seed + run_idx,
            learning_rate=lr,
            kappa=kappa,
            invariance_weight=inv_w,
            rho_growth_gamma=rho_gamma,
            rho_estimate_quantile=rho_q,
            roa_weight=roa_w,
            l1_weight=l1_w,
            condition_margin=args.condition_margin,
            tb_log_dir=lyap_path / "tb" / iso / run_name,
        )

        certification_config = LyapunovCertificationConfig.from_training_config(
            training_config,
            bins_per_dim=args.cert_bins_per_dim,
            center_refinement_factor=args.cert_center_refinement_factor,
            origin_exclusion=0.01,
            rho_scaling=args.cert_rho_scaling,
            bisection_tol=args.cert_bisection_tol,
            max_scale_steps=args.cert_max_scale_steps,
            max_bisection_steps=args.cert_max_bisection_steps,
            cert_method=args.cert_method,
            use_ibp_filter=not args.disable_ibp_filter,
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
            device=device,
        )
        train_results = trainer.train()
        trainer.save(base_path)

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

    print(f"\nGrid search complete. All results saved to: {sweep_base_path}")


if __name__ == "__main__":
    main()