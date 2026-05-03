import argparse
import numpy as np
import torch as th
import logging

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from mpc_datagen import MPCDataGenerator

from lcil.lyapunov_learning import LyapunovTrainingConfig, NeuralLyapunovCandidate, LyapunovTrainer
from lcil.certification import LyapunovCertificationConfig, BisectCertifier
from lcil.utils import lcil_plt, ICNN, MLP

from acados_ocp import get_ocp_solver
from diff_mpc import DiffMPCPolicy
from double_integrator_dyn import DoubleIntegratorDynamics


__logger__ = logging.getLogger(__name__)


def parse_cli_args(
    training_defaults: LyapunovTrainingConfig,
    certification_defaults: LyapunovCertificationConfig,
) -> argparse.Namespace:
    """Parse command-line arguments for Lyapunov training with a Diff-MPC policy."""
    parser = argparse.ArgumentParser(
        description="Train a Lyapunov function for a differentiable MPC policy on the double integrator."
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string.")
    parser.add_argument("--dt", type=float, default=0.1, help="Sampling time in seconds.")
    parser.add_argument("--horizon", type=int, default=20, help="MPC horizon length N.")
    parser.add_argument("--mpc-tol", type=float, default=1e-8, help="acados QP tolerance.")
    parser.add_argument(
        "--mpc-batch-threads",
        type=int,
        default=1,
        help="Number of threads used by the batched Diff-MPC solver.",
    )
    parser.add_argument("--terminal-mode", type=str, default="regional", help="Terminal mode for MPC OCP.")
    training_defaults.add_to_argparse(
        parser,
        include_fields={
            "initial_sample_size",
            "batch_size",
            "outer_epochs",
            "steps_per_epoch",
            "counterexample_every",
            "learning_rate",
            "kappa",
            "invariance_weight",
            "rho_growth_gamma",
            "roa_weight",
            "l1_weight",
            "seed",
        },
    )

    parser.add_argument(
        "--save-folder",
        type=str,
        default="results/double_integrator/diff_mpc_lyapunov",
        help="Directory where checkpoints will be stored.",
    )

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
        learning_rate=1e-2,
        kappa=0.12,
        invariance_weight=1.0,
        rho_growth_gamma=1.1,
        roa_weight=0.1,
        l1_weight=1e-6,
        seed=5912354,
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
    save_dir = Path(args.save_folder) / datetime.now().strftime("%Y%m%d_%H%M%S")
    device = th.device(args.device)

    a_c = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    b_c = np.array([[0.0], [1.0]], dtype=np.float64)
    q = np.diag([1.0, 1.0]).astype(np.float64)
    r = np.array([[0.1]], dtype=np.float64)

    solver, _ = get_ocp_solver(
        A_c=a_c,
        B_c=b_c,
        Q=q,
        R=r,
        dt=args.dt,
        N=args.horizon,
        tol=args.mpc_tol,
        terminal_mode=args.terminal_mode,
        dynamics_type="discrete",
    )

    policy_model = DiffMPCPolicy(
        ocp_solver=solver,
        batch_size=args.batch_size,
        num_threads_batch_solver=args.mpc_batch_threads,
    ).to(device)
    policy_model.eval()

    dyn_model = DoubleIntegratorDynamics(dt=args.dt).to(device)
    dyn_model.eval()

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
    ocp = solver.acados_ocp
    state_bounds = np.vstack([ocp.constraints.lbx, ocp.constraints.ubx])
    training_config = replace(
        training_defaults.from_namespace(args),
        state_dim=2,
        state_bounds=state_bounds,
        train_policy_model=False,
        tb_log_dir=save_dir / "tb",
    )
    certification_base_config = certification_defaults.from_namespace(args, prefix="cert-")

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
        __logger__.info("Skipping certification because training aborted: %s", train_results.abort_reason)
        return
    trainer.save(save_dir)

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
    certifier.save(save_dir)

    # ---------------------------------------------------------------------
    # 5. Plot & Save
    # ---------------------------------------------------------------------
    if True:
        generator = MPCDataGenerator(solver, T_sim=40.0, dt=args.dt, reset_solver=True)
        rollout_dataset = generator.generate(100)

        def lyapunov_func(states: np.ndarray) -> np.ndarray:
            x = th.as_tensor(states, dtype=th.float32, device=device)
            with th.no_grad():
                v = lyap_model(x)
            return v.detach().cpu().numpy().reshape(-1)

        lcil_plt.lyapunov(
            lyapunov_func=lyapunov_func,
            dataset=rollout_dataset,
            state_indices=[0, 1],
            state_labels=["x", "v"],
            plot_3d=True,
            certified_regions=cert_results.certified_sublevel_regions,
            uncertified_regions=cert_results.uncertified_regions,
            html_path=save_dir / "lyapunov_plot.html",
        )

    lcil_plt.certified_regions_2d(
        certification_result=cert_results,
        state_labels=["x", "v"],
        html_path=save_dir / "certified_regions.html",
    )


if __name__ == "__main__":
    main()