import argparse
import itertools
import torch as th
import numpy as np

from datetime import datetime
from pathlib import Path

from lcil.lyapunov_learning import LyapunovTrainingConfig, NeuralLyapunovCandidate, LyapunovTrainer
from lcil.certification import LyapunovCertificationConfig, ABCrownCertifier
from lcil.utils import lcil_plt, ICNN, MLP
from lcil.imitation_learning_mlp import MLPPolicy
from mpc_datagen import MPCDataset

from inv_pend_cart_dyn import InvertedPendulumOnCartDynamics
from model import InvertedPendulumOnCartPolicy


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for Lyapunov learning with a fixed policy."""
    parser = argparse.ArgumentParser(
        description="Train and certify a Lyapunov candidate for a fixed inverted pendulum on cart policy."
    )
    parser.add_argument(
        "--policy-path", type=str, default="results/inverted_pendulum_on_cart/20260318_114317/model.pt",
        help="Path to the trained fixed policy model checkpoint.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string.")
    parser.add_argument("--initial-sample-size", type=int, default=5000, help="Training sample size.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Training batch size.")
    parser.add_argument("--outer-epochs", type=int, default=500, help="Number of outer epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=10, help="Gradient steps per epoch.")
    parser.add_argument(
        "--counterexample-every", type=int, default=10,
        help="Counterexample search interval in epochs.")
    
    # Grid Search Parameters (accept multiple values)
    parser.add_argument("--learning-rate", nargs='+', type=float, default=[5e-3], help="Optimizer learning rate(s).")
    parser.add_argument("--kappa", nargs='+', type=float, default=[0.005, 0.015], help="Lyapunov decrease margin kappa(s).")
    parser.add_argument("--invariance-weight", nargs='+', type=float, default=[1.0], help="Invariance loss weight(s).")
    parser.add_argument("--rho-growth-gamma", nargs='+', type=float, default=[1.5], help="ROA rho growth factor(s).")
    parser.add_argument("--roa-weight", nargs='+', type=float, default=[2.0], help="ROA objective weight(s).")
    parser.add_argument("--l1-weight", nargs='+', type=float, default=[1e-5], help="L1 regularization weight(s).")

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
    policy_model = InvertedPendulumOnCartPolicy(feature_net=feature_net).to(device)
    policy_model.eval()

    # Parent directory for this entire grid search sweep
    sweep_base_path = policy_path.parent / "lyapunov" / datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_base_path.mkdir(parents=True, exist_ok=True)
    
    dyn_model = InvertedPendulumOnCartDynamics(dt=feature_net.global_config.dt).to(device)
    dyn_model.eval()
    
    rollout_dataset_path = policy_path.parent / "policy_rollouts.hdf5"
    rollout_dataset = None
    if rollout_dataset_path.exists():
        rollout_dataset = MPCDataset.load(rollout_dataset_path)

    state_bounds = np.vstack([feature_net.global_config.constraints.lbx, feature_net.global_config.constraints.ubx])
    
    # Pendulum is only stabilizable in a much smaller region around the upward position
    # The full state bounds [-2, 10, -9.4, 10] are too large for the policy to stabilize everywhere.
    # We restrict training and certification bounds to a realistic stabilizable region.
    training_bounds_percentage = np.array([0.5, 0.4, 0.15, 0.4])
    cert_percentage = np.array([0.4, 0.3, 0.1, 0.3])
    
    training_bounds = state_bounds * training_bounds_percentage[:, None].T
    cert_bounds = state_bounds * cert_percentage[:, None].T # Shape (2, state_dim)
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
        args.l1_weight
    ))

    print(f"Starting grid search over {len(grid)} configurations...")

    for run_idx, (lr, kappa, inv_w, rho_gamma, roa_w, l1_w) in enumerate(grid):
        print(f"\n[{run_idx+1}/{len(grid)}] Running config -> "
              f"lr: {lr}, kappa: {kappa}, inv_w: {inv_w}, rho_g: {rho_gamma}, roa_w: {roa_w}, l1_w: {l1_w}")
        
        # Create a specific folder for this parameter combination
        run_name = f"lr_{lr}__kappa_{kappa}__invw_{inv_w}__rhog_{rho_gamma}__roaw_{roa_w}__l1w_{l1_w}"
        base_path = sweep_base_path / run_name
        base_path.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------------
        # 1. Initialize fresh Lyapunov Model (so it trains from scratch)
        # ---------------------------------------------------------------------
        lyap_feature = MLP([4, 128, 128, 128, 1], ["tanh", "tanh", "tanh", "identity"]).to(device)
        lyap_model = NeuralLyapunovCandidate(
            feature_net=lyap_feature,
            state_dim=4,
            eps=0.1,
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
            train_policy_model=False,
            seed=args.seed + run_idx, # Optional: vary seed slightly to avoid perfectly identical samples
            # Variables from grid
            learning_rate=lr,
            kappa=kappa,
            invariance_weight=inv_w,
            rho_growth_gamma=rho_gamma,
            roa_weight=roa_w,
            l1_weight=l1_w,
        )

        certification_config = LyapunovCertificationConfig.from_training_config(
            training_config,
            bins_per_dim=args.cert_bins_per_dim,
            cert_center_refinement_factor=args.cert_center_refinement_factor,
            cert_origin_exclusion=0.01,
            cert_rho_scaling=args.cert_rho_scaling,
            cert_bisection_tol=args.cert_bisection_tol,
            cert_max_scale_steps=args.cert_max_scale_steps,
            cert_max_bisection_steps=args.cert_max_bisection_steps,
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
        _, cert_results = certifier.certify(max(train_results.rho_estimate, 1e-3))

        # ---------------------------------------------------------------------
        # 5. Plot & Save
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
                state_indices=[2, 3],
                state_labels=["theta", "theta_dot"],
                plot_3d=True,
                certified_regions=cert_results.certified_regions,
                uncertified_regions=cert_results.failed_regions,
                html_path=base_path / "lyapunov_plot.html",
            )

        lcil_plt.certified_regions_2d(
            certification_result=cert_results,
            state_labels=["x", "v", "theta", "theta_dot"],
            html_path=base_path / "certified_regions.html",
        )

    print(f"\nGrid search complete. All results saved to: {sweep_base_path}")


if __name__ == "__main__":
    main()