import argparse
import torch as th
import numpy as np

from datetime import datetime
from pathlib import Path

from lcil.lyapunov_learning import LyapunovTrainingConfig, NeuralLyapunovCandidate, LyapunovTrainer
from lcil.certification import LyapunovCertificationConfig, certify_lyapunov
from lcil.utils import lcil_plt, ICNN, MLP
from lcil.imitation_learning_mlp import MLPPolicy
from mpc_datagen import MPCDataset

from double_integrator_dyn import DoubleIntegratorDynamics


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for Lyapunov learning with a fixed policy."""
    parser = argparse.ArgumentParser(
        description="Train and certify a Lyapunov candidate for a fixed double-integrator policy."
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="results/double_integrator/20260222_112847/model.pt",
        help="Path to the trained fixed policy model checkpoint.",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="results/double_integrator_lyap",
        help="Output folder for trained Lyapunov model artifacts.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Training sample size.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--outer-epochs", type=int, default=100, help="Number of outer epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=5, help="Gradient steps per epoch.")
    parser.add_argument(
        "--counterexample-every",
        type=int,
        default=10,
        help="Counterexample search interval in epochs.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=5912354, help="Random seed.")
    parser.add_argument("--kappa", type=float, default=0.12, help="Lyapunov decrease margin kappa.")
    parser.add_argument("--invariance-weight", type=float, default=1.0, help="Invariance loss weight.")
    parser.add_argument("--rho-growth-gamma", type=float, default=1.1, help="ROA rho growth factor.")
    parser.add_argument("--roa-weight", type=float, default=0.1, help="ROA objective weight.")
    parser.add_argument("--l1-weight", type=float, default=1e-6, help="L1 regularization weight.")

    parser.add_argument("--cert-step", type=float, default=0.5, help="Certification grid step.")
    parser.add_argument("--cert-rho-scaling", type=float, default=1.2, help="Certification rho scaling.")
    parser.add_argument("--cert-bisection-tol", type=float, default=1e-3, help="Certification bisection tolerance.")
    parser.add_argument(
        "--cert-max-scale-steps",
        type=int,
        default=15,
        help="Maximum scale expansion steps during certification.",
    )
    parser.add_argument(
        "--cert-max-bisection-steps",
        type=int,
        default=20,
        help="Maximum bisection steps during certification.",
    )
    parser.add_argument(
        "--cert-method",
        type=str,
        default="alpha-crown",
        help="Certification backend/method name.",
    )
    return parser.parse_args()



# TODO: grid search for kappa cert method cert steps e.g.
def main() -> None:
    args = parse_cli_args()
    device = th.device(args.device)
    base_path = Path(args.save_folder) / datetime.now().strftime('%Y%m%d_%H%M%S')

    policy_path = Path(args.policy_path)
    policy_model = MLPPolicy.load(
        path=policy_path,
        map_location=device,
    ).to(device)

    lyap_feature = MLP([2, 32, 1], ["relu", "identity"]).to(device)
    lyap_model = NeuralLyapunovCandidate(
        feature_net=lyap_feature,
        state_dim=2,
        epsilon=1e-3,
    ).to(device)
    dyn_model = DoubleIntegratorDynamics(dt=policy_model.global_config.dt).to(device)

    training_config = LyapunovTrainingConfig(
        state_dim=policy_model.global_config.nx,
        state_bounds=np.vstack([policy_model.global_config.constraints.lbx, policy_model.global_config.constraints.ubx]),
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        outer_epochs=args.outer_epochs,
        steps_per_epoch=args.steps_per_epoch,
        counterexample_every=args.counterexample_every,
        learning_rate=args.learning_rate,
        train_policy_model=False,
        seed=args.seed,
        kappa=args.kappa,
        invariance_weight=args.invariance_weight,
        rho_growth_gamma=args.rho_growth_gamma,
        roa_weight=args.roa_weight,
        l1_weight=args.l1_weight,
    )

    certification_config = LyapunovCertificationConfig.from_training_config(
        training_config,
        cert_step=args.cert_step,
        cert_origin_exclusion=None,
        cert_rho_scaling=args.cert_rho_scaling,
        cert_bisection_tol=args.cert_bisection_tol,
        cert_max_scale_steps=args.cert_max_scale_steps,
        cert_max_bisection_steps=args.cert_max_bisection_steps,
        cert_method=args.cert_method,
        state_bounds=training_config.state_bounds * 1.2,
    )

    trainer = LyapunovTrainer(
        policy_model=policy_model,
        lyap_model=lyap_model,
        dyn_model=dyn_model,
        config=training_config,
        device=device,
    )
    train_results = trainer.train()
    trainer.save(base_path)

    _, cert_results = certify_lyapunov(
        policy_model,
        lyap_model,
        dyn_model,
        certification_config,
        rho_estimate=train_results.rho_estimate,
        device=device,
    )

    rollout_dataset_path = policy_path.parent / "policy_rollouts.hdf5"
    if rollout_dataset_path.exists():
        rollout_dataset = MPCDataset.load(rollout_dataset_path)

        def lyapunov_func(states: np.ndarray) -> np.ndarray:
            x = th.as_tensor(states, dtype=th.float32, device=device)
            with th.no_grad():
                v = lyap_model(x)
            return v.detach().cpu().numpy().reshape(-1)

        lcil_plt.lyapunov(
            dataset=rollout_dataset[:100],
            lyapunov_func=lyapunov_func,
            state_indices=[0, 1],
            state_labels=["x", "v"],
            plot_3d=True,
            certified_regions=cert_results.certified_regions,
            uncertified_regions=cert_results.failed_regions,
            html_path=base_path / "lyapunov_plot.html",
        )

    lcil_plt.certified_regions_2d(
        cert_results.certified_regions,
        cert_results.failed_regions,
        state_labels=["x", "v"],
        html_path=base_path / "certified_regions.html",
    )


if __name__ == "__main__":
    main()