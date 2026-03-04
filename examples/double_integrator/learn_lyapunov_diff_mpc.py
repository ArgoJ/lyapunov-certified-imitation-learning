import argparse
import numpy as np
import torch as th

from datetime import datetime
from pathlib import Path

from lcil.lyapunov_learning import (
    LyapunovTrainingConfig,
    LyapunovTrainer,
    NeuralLyapunovCandidate,
)
from lcil.utils import MLP
from lcil.utils.package_logger import get_package_logger

from acados_ocp import get_ocp_solver
from diff_mpc import DiffMPCPolicy
from double_integrator_dyn import DoubleIntegratorDynamics


__logger__ = get_package_logger(__name__)


def parse_cli_args() -> argparse.Namespace:
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

    parser.add_argument("--initial-sample-size", type=int, default=1000, help="Initial training sample count.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for Lyapunov training.")
    parser.add_argument("--outer-epochs", type=int, default=100, help="Number of outer CEGIS epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=5, help="Optimizer steps per outer epoch.")
    parser.add_argument(
        "--counterexample-every",
        type=int,
        default=10,
        help="Counterexample mining interval (in outer epochs).",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Optimizer learning rate.")
    parser.add_argument("--kappa", type=float, default=0.12, help="Lyapunov decrease margin.")
    parser.add_argument("--invariance-weight", type=float, default=1.0, help="Invariance loss weight.")
    parser.add_argument("--rho-growth-gamma", type=float, default=1.1, help="ROA rho growth factor.")
    parser.add_argument("--roa-weight", type=float, default=0.1, help="ROA surrogate loss weight.")
    parser.add_argument("--l1-weight", type=float, default=1e-6, help="L1 regularization weight.")
    parser.add_argument("--seed", type=int, default=5912354, help="Random seed.")

    parser.add_argument(
        "--save-folder",
        type=str,
        default="results/double_integrator/diff_mpc_lyapunov",
        help="Directory where checkpoints will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
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

    lyap_feature = MLP([2, 32, 1], ["tanh", "identity"]).to(device)
    lyap_model = NeuralLyapunovCandidate(
        feature_net=lyap_feature,
        state_dim=2,
        eps=1e-3,
    ).to(device)

    ocp = solver.acados_ocp
    state_bounds = np.vstack([ocp.constraints.lbx, ocp.constraints.ubx])
    training_config = LyapunovTrainingConfig(
        state_dim=2,
        state_bounds=state_bounds,
        initial_sample_size=args.initial_sample_size,
        batch_size=args.batch_size,
        outer_epochs=args.outer_epochs,
        steps_per_epoch=args.steps_per_epoch,
        counterexample_every=args.counterexample_every,
        train_policy_model=False,
        seed=args.seed,
        learning_rate=args.learning_rate,
        kappa=args.kappa,
        invariance_weight=args.invariance_weight,
        rho_growth_gamma=args.rho_growth_gamma,
        roa_weight=args.roa_weight,
        l1_weight=args.l1_weight,
    )

    trainer = LyapunovTrainer(
        policy_model=policy_model,
        lyap_model=lyap_model,
        dyn_model=dyn_model,
        config=training_config,
        device=device,
    )
    train_results = trainer.train()

    save_dir = Path(args.save_folder) / datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save(save_dir)

    __logger__.info(
        "Diff-MPC Lyapunov training completed. rho_estimate=%.6f, saved_to=%s",
        train_results.rho_estimate,
        save_dir,
    )


if __name__ == "__main__":
    main()