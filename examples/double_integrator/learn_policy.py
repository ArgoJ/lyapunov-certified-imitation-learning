import argparse
import torch as th
import sys

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, replace

from mpc_datagen import MPCDataset
from lcil.utils import (
    EarlyStopping,
    IntegrationMethod, 
    GridSearchHelper,
    config_field,
    ArgumentParserConfig,
)
from lcil.imitation_learning import *

from . import DoubleIntegratorDynamics, default_dataset_path, resolve_dataset_path, DOUBLE_INTEGRATOR_RESULTS_DIR


@dataclass(frozen=True)
class PolicyScriptConfig(ArgumentParserConfig):
    device: str = config_field(default="cpu", help="Torch device string (e.g. cpu, cuda).")
    activation: str = config_field(default="relu", help="Activation function for policy net hidden layers.")
    hidden_size: int = config_field(default=32, help="Number of neurons in each hidden layer.")
    layers: int = config_field(default=2, help="Number of hidden layers in the policy net.")


def parse_cli_args() -> GridSearchHelper[tuple[PolicyScriptConfig, ImitationTrainingConfig]]:
    """Parse command-line arguments and return a grid search helper for policy training."""
    parser = argparse.ArgumentParser(
        description="Train double-integrator imitation policies using grid search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    script_defaults = PolicyScriptConfig()
    script_defaults.add_to_argparse(
        parser,
        nargs_fields={"activation", "hidden_size", "layers"}
    )
    
    training_defaults = ImitationTrainingConfig(
        dataset_path=default_dataset_path(),
        val_fraction=0.2,
        split_strategy="random",
        epochs=100,
        learning_rate=5e-4,
        scheduler_type="plateau",
        scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 5},
    )
    training_defaults.add_to_argparse(
        parser,
        nargs_fields={"learning_rate", "batch_size", "dropout", "weight_decay"},
        exclude_fields={"scheduler_type", "scheduler_kwargs", "tb_log_dir"},
    )
    
    args = parser.parse_args()
    sweep_id = datetime.now().strftime('%Y%m%d_%H%M%S').replace(" ", "_").replace(":", "-")    
    sweep: GridSearchHelper[tuple[PolicyScriptConfig, ImitationTrainingConfig]] = GridSearchHelper.from_namespace(
        args,
        script_defaults,
        training_defaults,
        output_root=DOUBLE_INTEGRATOR_RESULTS_DIR,
        sweep_id=sweep_id,
    )
    return sweep


def main() -> None:
    sweep = parse_cli_args()
    
    first_script_config, first_train_config = sweep.configs[0]
    device = th.device(first_script_config.device)
    dataset_path = resolve_dataset_path(first_train_config.dataset_path)

    print(f"Starting grid search over {len(sweep)} configurations...")

    # ---------------------------------------------------------------------
    # Load Dataset, Dataloader and create Loss
    # ---------------------------------------------------------------------
    source_dataset = MPCDataset.load(dataset_path)
    if len(source_dataset) == 0:
        raise ValueError("MPCDataset is empty; cannot extract configuration.")
    dataset_cfg = source_dataset.global_config

    train_loader, val_loader = create_train_and_val_dataloader(
        first_train_config,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        dtype=th.float32,
    )
    
    loss_fn = ReferenceWeightedDynamicsAwareLoss(
        reference_loss=ReferenceWeightedMSELoss(
            reference=dataset_cfg.cost.yref[-dataset_cfg.nu:],
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

    # ---------------------------------------------------------------------
    # Grid Search
    # ---------------------------------------------------------------------
    for run in sweep:
        script_config, train_config = run.config
        print(run.progress_message())
        
        base_path = run.output_dir.resolve()
        
        net = MLPPolicy(
            [dataset_cfg.nx] + [script_config.hidden_size] * script_config.layers + [dataset_cfg.nu],
            [script_config.activation] * script_config.layers + ["identity"],
            dropout=train_config.dropout,
            normalization="none",
            u_min=dataset_cfg.constraints.lbu,
            u_max=dataset_cfg.constraints.ubu,
        ).to(device)

        current_train_cfg = replace(
            train_config,
            tb_log_dir=sweep.sweep_base_path.parent.parent / "tb" / sweep.sweep_id / run.run_name
        )

        trainer = PolicyTrainer(
            model=net,
            dataloader=train_loader,
            training_config=current_train_cfg,
            val_dataloader=val_loader,
            early_stopper=EarlyStopping(patience=20, delta=1e-4),
            loss_fn=loss_fn,
            device=device,
        )
        
        trainer.train()
        trainer.save(
            save_folder=base_path, 
            global_config=dataset_cfg
        )

    print(f"\nGrid search complete. Models saved to: {sweep.sweep_base_path}")


if __name__ == "__main__":
    main()