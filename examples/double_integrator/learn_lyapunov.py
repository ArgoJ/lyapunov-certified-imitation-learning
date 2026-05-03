import argparse
import torch as th
import numpy as np

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from lcil.lyapunov_learning import LyapunovTrainingConfig, NeuralLyapunovCandidate, LyapunovTrainer
from lcil.certification import LyapunovCertificationConfig, BisectCertifier
from lcil.utils import lcil_plt, ICNN, MLP
from lcil.imitation_learning_mlp import MLPPolicy
from mpc_datagen import MPCDataset

from double_integrator_dyn import DoubleIntegratorDynamics


def parse_cli_args(
    training_defaults: LyapunovTrainingConfig,
    certification_defaults: LyapunovCertificationConfig,
) -> argparse.Namespace:
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
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string.")
    training_defaults.add_to_argparse(
        parser,
        include_fields={
            "initial_sample_size",
            "batch_size",
            "outer_epochs",
            "steps_per_epoch",
            "counterexample_every",
        },
    )
    
    # Grid Search Parameters (accept multiple values)
    parser.add_argument("--learning-rate", nargs='+', type=float, default=[1e-2], help="Optimizer learning rate(s).")
    parser.add_argument("--kappa", nargs='+', type=float, default=[0.12], help="Lyapunov decrease margin kappa(s).")
    parser.add_argument("--invariance-weight", nargs='+', type=float, default=[1.0], help="Invariance loss weight(s).")
    parser.add_argument("--rho-growth-gamma", nargs='+', type=float, default=[1.1], help="ROA rho growth factor(s).")
    parser.add_argument("--roa-weight", nargs='+', type=float, default=[0.1], help="ROA objective weight(s).")
    parser.add_argument("--l1-weight", nargs='+', type=float, default=[1e-6], help="L1 regularization weight(s).")

    parser.add_argument("--seed", type=int, default=5912354, help="Random seed.")
    
    # Certification Parameters
    certification_defaults.add_to_argparse(
        parser,
        prefix="cert-",
        include_fields={
            "bins_per_dim",
            "rho_scaling",
            "bisection_tol",
            "max_scale_steps",
            "max_bisection_steps",
            "cert_method",
        },
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
    )
    certification_defaults = LyapunovCertificationConfig(
        state_dim=2,
        cert_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=float),
        bins_per_dim=4,
        rho_scaling=1.5,
        bisection_tol=1e-3,
        max_scale_steps=15,
        max_bisection_steps=20,
        cert_method="alpha-crown",
    )
    args = parse_cli_args(training_defaults, certification_defaults)
    device = th.device(args.device)

    # Load policy and dynamics once (they don't change across runs)
    policy_path = Path(args.policy_path)
    policy_model = MLPPolicy.load(
        path=policy_path,
        map_location=device,
    ).to(device)
    policy_model.eval()

    # Parent directory for this entire grid search sweep
    sweep_base_path = policy_path.parent / "lyapunov" / datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_base_path.mkdir(parents=True, exist_ok=True)
    
    dyn_model = DoubleIntegratorDynamics(dt=policy_model.global_config.dt).to(device)
    dyn_model.eval()
    
    rollout_dataset_path = policy_path.parent / "policy_rollouts.hdf5"
    rollout_dataset = None
    if rollout_dataset_path.exists():
        rollout_dataset = MPCDataset.load(rollout_dataset_path)

    state_bounds = np.vstack([policy_model.global_config.constraints.lbx, policy_model.global_config.constraints.ubx])

    training_sweep_configs = training_defaults.iter_from_namespace(args)
    certification_base_config = certification_defaults.from_namespace(args, prefix="cert-")

    print(f"Starting grid search over {len(training_sweep_configs)} configurations...")

    for run_idx, sweep_config in enumerate(training_sweep_configs):
        print(f"\n[{run_idx+1}/{len(training_sweep_configs)}] Running config -> "
              f"lr: {sweep_config.learning_rate}, kappa: {sweep_config.kappa}, "
              f"inv_w: {sweep_config.invariance_weight}, rho_g: {sweep_config.rho_growth_gamma}, "
              f"roa_w: {sweep_config.roa_weight}, l1_w: {sweep_config.l1_weight}")
        
        # Create a specific folder for this parameter combination
        run_name = (
            f"lr_{sweep_config.learning_rate}__kappa_{sweep_config.kappa}"
            f"__invw_{sweep_config.invariance_weight}__rhog_{sweep_config.rho_growth_gamma}"
            f"__roaw_{sweep_config.roa_weight}__l1w_{sweep_config.l1_weight}"
        )
        base_path = sweep_base_path / run_name
        base_path.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------------
        # 1. Initialize fresh Lyapunov Model (so it trains from scratch)
        # ---------------------------------------------------------------------
        lyap_feature = MLP([2, 32, 1], ["tanh", "identity"]).to(device)
        lyap_model = NeuralLyapunovCandidate(
            feature_net=lyap_feature,
            state_dim=2,
            eps=1e-3,
        ).to(device)

        # ---------------------------------------------------------------------
        # 2. Setup Configs with current grid parameters
        # ---------------------------------------------------------------------
        training_config = replace(
            sweep_config,
            state_dim=policy_model.global_config.nx,
            state_bounds=state_bounds,
            train_policy_model=False,
            seed=sweep_config.seed + run_idx if sweep_config.seed is not None else None,
            tb_log_dir=base_path / "tb",
        )

        certification_config = LyapunovCertificationConfig.from_training_config(
            training_config,
            bins_per_dim=certification_base_config.bins_per_dim,
            origin_exclusion=None,
            rho_scaling=certification_base_config.rho_scaling,
            bisection_tol=certification_base_config.bisection_tol,
            max_scale_steps=certification_base_config.max_scale_steps,
            max_bisection_steps=certification_base_config.max_bisection_steps,
            cert_method=certification_base_config.cert_method,
            cert_bounds=training_config.state_bounds * 0.8,
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
        if train_results.aborted:
            print(f"Skipping run {run_name}: {train_results.abort_reason}")
            continue
        trainer.save(base_path)

        # ---------------------------------------------------------------------
        # 4. Certify
        # ---------------------------------------------------------------------
        certifier = BisectCertifier(
            policy_model,
            lyap_model,
            dyn_model,
            certification_config,
            device,
        )
        cert_results = certifier.certify(max(train_results.rho_estimate, 1e-3))
        certifier.save(base_path)

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
                state_indices=[0, 1],
                state_labels=["x", "v"],
                plot_3d=True,
                certified_regions=cert_results.certified_sublevel_regions,
                uncertified_regions=cert_results.uncertified_regions,
                html_path=base_path / "lyapunov_plot.html",
            )

        lcil_plt.certified_regions_2d(
            certification_result=cert_results,
            state_labels=["x", "v"],
            html_path=base_path / "certified_regions.html",
        )

    print(f"\nGrid search complete. All results saved to: {sweep_base_path}")


if __name__ == "__main__":
    main()