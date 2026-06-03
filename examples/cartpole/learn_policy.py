import argparse
import logging
import torch as th

from dataclasses import dataclass, replace
from datetime import datetime

from mpc_datagen import MPCDataset

from lcil.imitation_learning import *
from lcil.utils import ArgumentParserConfig, EarlyStopping, GridSearchHelper, config_field

from . import (
    CARTPOLE_RESULTS_DIR,
    CartpoleDynamics,
    CartpoleAngleWrapper,
    PendulumOnCartConfig,
    default_dataset_path,
    resolve_dataset_path,
)


__logger__ = logging.getLogger("lcil.examples.cartpole.learn_policy")


@dataclass(frozen=True)
class PolicyScriptConfig(ArgumentParserConfig):
    device: str = config_field(default="cpu", help="Torch device string (e.g. cpu, cuda).")
    activation: str = config_field(default="relu", help="Activation function for policy net hidden layers.")
    hidden_size: int = config_field(default=24, help="Number of neurons in each hidden layer.")
    layers: int = config_field(default=3, help="Number of hidden layers in the policy net.")


def parse_cli_args() -> GridSearchHelper[tuple[PolicyScriptConfig, ImitationTrainingConfig]]:
    parser = argparse.ArgumentParser(
        description="Train an inverted pendulum on cart imitation policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    script_defaults = PolicyScriptConfig()
    script_defaults.add_to_argparse(
        parser,
        nargs_fields={"activation", "hidden_size", "layers"},
    )

    training_defaults = ImitationTrainingConfig(
        dataset_path=default_dataset_path(),
        val_fraction=0.2,
        split_strategy="random",
        epochs=200,
        learning_rate=5e-4,
        scheduler_type="plateau",
        scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 4},
    )
    training_defaults.add_to_argparse(
        parser,
        nargs_fields={"learning_rate", "batch_size", "dropout", "weight_decay"},
        exclude_fields={"scheduler_type", "scheduler_kwargs", "tb_log_dir"},
    )

    args = parser.parse_args()
    return GridSearchHelper.from_namespace(
        args,
        script_defaults,
        training_defaults,
        output_root=CARTPOLE_RESULTS_DIR,
        sweep_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )


def main() -> None:
    sweep = parse_cli_args()

    _, first_train_config = sweep.configs[0]
    dataset_path = resolve_dataset_path(first_train_config.dataset_path)

    source_dataset = MPCDataset.load(dataset_path)
    if len(source_dataset) == 0:
        raise ValueError("MPCDataset is empty; cannot extract configuration.")
    dataset_cfg = source_dataset.global_config
    sys_cfg = PendulumOnCartConfig()

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
            reference=[dataset_cfg.cost.yref[-dataset_cfg.nu:]],
            alpha=1.0,
            max_weight=1.0,
            min_weight=0.7,
        ),
        dynamics_loss=DynamicsAwareLoss(
            dynamics=CartpoleDynamics(dt=dataset_cfg.dt, sys_cfg=sys_cfg),
            x_min=th.tensor(dataset_cfg.constraints.lbx),
            x_max=th.tensor(dataset_cfg.constraints.ubx),
        ),
        lambda_dyn=2.0,
    )

    __logger__.info("Starting grid search over %d configurations...", len(sweep))
    for run in sweep:
        script_config, train_config = run.config
        __logger__.info("%s", run.progress_message())
        device = th.device(script_config.device)

        feature_net = MLPPolicy(
            [5] + [script_config.hidden_size] * script_config.layers + [dataset_cfg.nu],
            [script_config.activation] * script_config.layers + ["identity"],
            dropout=train_config.dropout,
            u_min=dataset_cfg.constraints.lbu,
            u_max=dataset_cfg.constraints.ubu,
        )
        net = CartpoleAngleWrapper(feature_net=feature_net).to(device)

        current_train_config = replace(
            train_config,
            tb_log_dir=CARTPOLE_RESULTS_DIR / "tb" / sweep.sweep_id / run.run_name,
        )

        trainer = PolicyTrainer(
            model=net,
            dataloader=train_loader,
            training_config=current_train_config,
            val_dataloader=val_loader,
            early_stopper=EarlyStopping(patience=10, delta=1e-5),
            loss_fn=loss_fn,
            device=device,
        )
        trainer.train()
        trainer.save(
            save_folder=run.output_dir.resolve(),
            global_config=source_dataset.global_config,
        )


if __name__ == "__main__":
    main()