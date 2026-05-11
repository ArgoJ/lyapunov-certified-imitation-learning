import argparse
import torch as th

from pathlib import Path
from datetime import datetime

from mpc_datagen import MPCDataset
from lcil.utils import EarlyStopping
from lcil.imitation_learning_mlp import *

from . import (
    CartpoleDynamics,
    CartpoleAngleWrapper,
    PendulumOnCartConfig,
    default_dataset_path,
    resolve_dataset_path,
)


def parse_cli_args(training_defaults: ImitationTrainingConfig) -> argparse.Namespace:
    """Parse command-line arguments for policy training."""
    parser = argparse.ArgumentParser(
        description="Train a inverted pendulum on cart imitation policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=default_dataset_path(),
        help="Path to the source MPC dataset (HDF5).",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument(
        "--near-duplicate-radius",
        type=float,
        default=1e-4,
        help="Optional near-duplicate L2 radius in normalized feature space.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string (e.g. cpu, cuda).")
    
    training_defaults.add_to_argparse(
        parser,
        exclude_fields={"scheduler_type", "scheduler_kwargs", "tb_log_dir"}
    )
    return parser.parse_args()


def main() -> None:
    base_path = Path("results/cartpole")
    iso = datetime.now().strftime('%Y%m%d_%H%M%S').replace(" ", "_").replace(":", "-")

    training_defaults = ImitationTrainingConfig(
        epochs=200,
        learning_rate=5e-4,
        scheduler_type="plateau",
        scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 4},
        tb_log_dir=(base_path / "tb" / iso),
    )
    args = parse_cli_args(training_defaults)
    device = args.device
    dataset_path = resolve_dataset_path(args.dataset_path)

    source_dataset = MPCDataset.load(dataset_path)
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
        mpc_dataset=str(dataset_path),
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

    training_cfg = training_defaults.from_namespace(args)

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