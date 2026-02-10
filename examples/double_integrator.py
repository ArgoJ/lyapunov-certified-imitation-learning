"""
PVTOL Lyapunov verification example using auto_LiRPA / alpha-beta-CROWN.

This reimplements the CPLEX-based Lyapunov verification from the paper
using neural-network bound-propagation methods:

* **auto_LiRPA**   – CROWN / alpha-CROWN for fast (incomplete) bounds.
* **Domain splitting** – recursive bisection (BaB-style) for complete
  certification.

The PVTOL (Planar Vertical Take-Off and Landing) system has 6-D state::

    [x, y, theta, x_dot, y_dot, theta_dot]

and a pre-computed LQR controller with 2-D output (thrust biases).

Usage::

    python examples/double_integrator.py --seed 42 --device cpu
"""
import argparse
import random
import time

import numpy as np
import torch

from lyapunov_certified_imitation_learning.models.policy import PolicyNet
from lyapunov_certified_imitation_learning.models.lyapunov import LyapunovNet
from lyapunov_certified_imitation_learning.models.dynamics import PVTOLClosedLoop
from lyapunov_certified_imitation_learning.training.lyap_trainer import (
    pre_train,
    train_main,
    build_certify_list,
)
from lyapunov_certified_imitation_learning.verification.abcrown_wrapper import (
    LyapunovVerifier,
)


def main():
    parser = argparse.ArgumentParser(
        description="PVTOL Lyapunov verification with alpha-beta-CROWN"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=0.00625)
    args = parser.parse_args()

    # --- reproducibility ------------------------------------------------
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed: {args.seed}  |  Device: {args.device}")

    # --- models ---------------------------------------------------------
    policy_model = PolicyNet().to(args.device)
    lyap_model = LyapunovNet().to(args.device)
    closed_loop = PVTOLClosedLoop(policy_model, lyap_model).to(args.device)

    # --- pre-train ------------------------------------------------------
    t0 = time.time()
    pre_train(lyap_model, policy_model, args.device)

    # --- certification grid (512 sub-regions, matching the paper) -------
    certify_list = build_certify_list()
    print(f"Certifying over {len(certify_list)} sub-regions")

    # --- main training + certification loop -----------------------------
    certify_counter_examples: list[torch.Tensor] = []
    num_trains = train_main(
        policy_model,
        lyap_model,
        closed_loop,
        certify_list,
        certify_counter_examples,
        args.device,
    )

    total_time = time.time() - t0
    print(f"\nTraining complete!")
    print(f"  Rounds : {num_trains}")
    print(f"  Time   : {total_time:.1f}s")

    # --- save models ----------------------------------------------------
    torch.save(
        {"model_state_dict": lyap_model.state_dict()},
        f"pvtol_lyapunov_seed{args.seed}.pt",
    )
    torch.save(
        {"model_state_dict": policy_model.state_dict()},
        f"pvtol_policy_seed{args.seed}.pt",
    )
    print("Models saved.")

    # --- final demonstration verification -------------------------------
    print("\n--- Final Verification (sample regions) ---")
    verifier = LyapunovVerifier(device=args.device)

    for i, element in enumerate(certify_list[:5]):
        lb = torch.tensor(
            [element[0], element[2], element[4],
             element[6], element[8], element[10]],
            dtype=torch.float32,
        )
        ub = torch.tensor(
            [element[1], element[3], element[5],
             element[7], element[9], element[11]],
            dtype=torch.float32,
        )

        verified, ub_val = verifier.verify_decrease(closed_loop, lb, ub)
        tag = "VERIFIED" if verified else f"FAILED (ub={ub_val:.6f})"
        print(f"  Region {i}: {tag}")


if __name__ == "__main__":
    main()

