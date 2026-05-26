import argparse
import torch as th
import sys

from pathlib import Path
from datetime import datetime

from mpc_datagen import MPCDataset
from lcil.utils import EarlyStopping, IntegrationMethod
from lcil.imitation_learning import *

from . import DoubleIntegratorDynamics, default_dataset_path, resolve_dataset_path


def parse_cli_args(training_defaults: ImitationTrainingConfig) -> argparse.Namespace:
    """Parse command-line arguments for policy training."""
    parser = argparse.ArgumentParser(
        description="Train a double-integrator imitation policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string (e.g. cpu, cuda).")
    
    training_defaults.add_to_argparse(
        parser,
        exclude_fields={"scheduler_type", "scheduler_kwargs", "tb_log_dir"},
    )
    return parser.parse_args()


def main() -> None:
    base_path = Path("results/double_integrator")
    iso = datetime.now().strftime('%Y%m%d_%H%M%S').replace(" ", "_").replace(":", "-")

    training_defaults = ImitationTrainingConfig(
        dataset_path=default_dataset_path(),
        val_fraction=0.2,
        split_strategy="random",
        epochs=100,
        learning_rate=5e-4,
        scheduler_type="plateau",
        scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 5},
        tb_log_dir=(base_path / "tb" / iso),
    )
    args = parse_cli_args(training_defaults)
    training_cfg = training_defaults.from_namespace(args)
    device = args.device
    dataset_path = resolve_dataset_path(training_cfg.dataset_path)

    source_dataset = MPCDataset.load(dataset_path)
    if len(source_dataset) == 0:
        raise ValueError("MPCDataset is empty; cannot extract configuration.")
    dataset_cfg = source_dataset.global_config

    net = MLPPolicy(
        [dataset_cfg.nx, 32, 32, dataset_cfg.nu],
        ["relu", "relu", "identity"],
        dropout=training_cfg.dropout,
        normalization="layer_norm",
        u_min=dataset_cfg.constraints.lbu,
        u_max=dataset_cfg.constraints.ubu,
    )

    train_loader, val_loader = create_train_and_val_dataloader(
        training_cfg,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        dtype=th.float32,
    )

    loss_fn = ReferenceWeightedDynamicsAwareLoss(
        reference_loss=ReferenceWeightedMSELoss(
            reference=[dataset_cfg.cost.yref[-dataset_cfg.nu:]],
            alpha=1.0,
            max_weight=5.0,
            min_weight=4.0),
        dynamics_loss=DynamicsAwareLoss(
            dynamics=DoubleIntegratorDynamics(
                dt=dataset_cfg.dt,
                method=IntegrationMethod.CLASSICAL_RK4), 
            x_min=th.tensor(dataset_cfg.constraints.lbx), 
            x_max=th.tensor(dataset_cfg.constraints.ubx)),
        lambda_dyn=5.0,
    )

    trainer = PolicyTrainer(
        model=net,
        dataloader=train_loader,
        training_config=training_cfg,
        val_dataloader=val_loader,
        early_stopper=EarlyStopping(patience=20, delta=1e-4),
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