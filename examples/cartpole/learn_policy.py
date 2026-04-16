import argparse
import torch as th

from pathlib import Path
from datetime import datetime

from mpc_datagen import MPCDataset
from lcil.utils import EarlyStopping
from lcil.imitation_learning_mlp import *

from cartpole_dyn import CartpoleDynamics
from sys_cfg import PendulumOnCartConfig
from model import CartpoleAngleWrapper


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for policy training."""
    parser = argparse.ArgumentParser(description="Train a inverted pendulum on cart imitation policy.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=Path.cwd() / "results" / "cartpole" / "data" / "cartpole_N40_data.hdf5",
        help="Path to the source MPC dataset (HDF5).",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of policy training epochs.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Optimizer learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument(
        "--near-duplicate-radius",
        type=float,
        default=1e-4,
        help="Optional near-duplicate L2 radius in normalized feature space.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string (e.g. cpu, cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    device = args.device
    dataset_path = args.dataset_path
    base_path = Path("results/cartpole")
    iso = datetime.now().strftime('%Y%m%d_%H%M%S').replace(" ", "_").replace(":", "-")

    source_dataset = MPCDataset.load(Path(dataset_path))
    if len(source_dataset) == 0:
        raise ValueError("MPCDataset is empty; cannot extract configuration.")
    dataset_cfg = source_dataset.global_config

    feature_net = MLPPolicy(
        [5, 24, 24, 24, dataset_cfg.nu],
        ["relu", "relu", "relu", "identity"],
        u_min=dataset_cfg.constraints.lbu,
        u_max=dataset_cfg.constraints.ubu,
    )
    net = CartpoleAngleWrapper(feature_net=feature_net).to(device)
    
    sys_cfg = PendulumOnCartConfig()

    train_loader, val_loader = create_train_and_val_dataloader(
        mpc_dataset=dataset_path,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        dtype=th.float32,
        near_duplicate_radius=args.near_duplicate_radius,
        val_fraction=0.2,
    )

    loss_fn = ReferenceWeightedDynamicsAwareLoss(
        reference_loss=ReferenceWeightedMSELoss(
            reference=[dataset_cfg.cost.yref[-dataset_cfg.nu:]],
            alpha=1.0,
            max_weight=1.0,
            min_weight=0.7),
        dynamics_loss=DynamicsAwareLoss(
            dynamics=CartpoleDynamics(dt=dataset_cfg.dt, sys_cfg=sys_cfg),
            x_min=th.tensor(dataset_cfg.constraints.lbx),
            x_max=th.tensor(dataset_cfg.constraints.ubx)),
        lambda_dyn=2.0,
    )

    training_cfg = ImitationTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        scheduler_type="plateau",
        scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 4},
        tb_log_dir=(base_path / "tb" / iso),
    )

    trainer = PolicyTrainer(
        model=net,
        dataloader=train_loader,
        training_config=training_cfg,
        val_dataloader=val_loader,
        early_stopper=EarlyStopping(patience=10, delta=1e-5),
        loss_fn=loss_fn,
        device=device,
    )
    trainer.train()
    trainer.save(
        save_folder=base_path / iso,
        global_config=source_dataset.global_config
    )

if __name__ == "__main__":
    main()