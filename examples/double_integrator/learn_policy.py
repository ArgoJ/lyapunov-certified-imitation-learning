import argparse
import torch as th

from pathlib import Path
from datetime import datetime

from mpc_datagen import MPCDataset
from lcil.utils import EarlyStopping
from lcil.imitation_learning_mlp import *

from double_integrator_dyn import DoubleIntegratorDynamics


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for policy training."""
    parser = argparse.ArgumentParser(description="Train a double-integrator imitation policy.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/josua/programming_stuff/projects/mpc-datagen/data/double_integrator_regional_N20_data.hdf5",
        help="Path to the source MPC dataset (HDF5).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of policy training epochs.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Optimizer learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument(
        "--save-folder",
        type=str,
        default="results/double_integrator",
        help="Path where the trained model state dict will be saved.",
    )
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

    source_dataset = MPCDataset.load(Path(dataset_path))
    if len(source_dataset) == 0:
        raise ValueError("MPCDataset is empty; cannot extract configuration.")
    dataset_cfg = source_dataset.global_config

    net = MLPPolicy(
        [2, 16, 16, 1],
        ["relu", "relu", "identity"],
        u_min=dataset_cfg.constraints.lbu,
        u_max=dataset_cfg.constraints.ubu,
    )

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
        reference_loss=ReferenceWeightedMSELoss(reference=[dataset_cfg.cost.yref[-dataset_cfg.nu:]], alpha=1.0, max_weight=2.0),
        dynamics_loss=DynamicsAwareLoss(
            dynamics=DoubleIntegratorDynamics(dt=0.1), 
            x_min=th.tensor(dataset_cfg.constraints.lbx), 
            x_max=th.tensor(dataset_cfg.constraints.ubx)
        ),
        lambda_dyn=1.5,
    )

    trainer = Trainer(
        model=net,
        dataloader=train_loader,
        val_dataloader=val_loader,
        early_stopper=EarlyStopping(patience=10, delta=1e-4),
        loss_fn=loss_fn,
        device=device,
    )
    trainer.set_adam_optimizer(learning_rate=args.lr, scheduler_type="cosine")
    trainer.train(num_epochs=args.epochs)
    trainer.save(
        save_folder=Path(args.save_folder) / datetime.now().strftime('%Y%m%d_%H%M%S'), 
        global_config=source_dataset.global_config
    )

if __name__ == "__main__":
    main()