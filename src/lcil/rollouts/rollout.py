import torch as th
import logging

from torch import nn
from numpy.typing import NDArray
from mpc_datagen import MPCDataset
from mpc_datagen.mpc_data import MPCConfig

from .policy_rollout import PolicyRolloutConfig, PolicyRolloutGenerator, StateSampler
from .lyapunov_rollout import LyapunovRollout

__logger__ = logging.getLogger(__name__)


def build_policy_rollout_dataset(
    policy_model: nn.Module,
    mpc_config: MPCConfig,
    dyn_model: nn.Module,
    rollout_steps: int,
    device: th.device,
    initial_states: NDArray | None = None,
    sampler: StateSampler | None = None,
    n_samples: int | None = None,
) -> MPCDataset | None: 
    rollout_config = PolicyRolloutConfig.from_mpc_config(
        mpc_config,
        t_sim=int(rollout_steps),
    )

    policy_rollout_generator = PolicyRolloutGenerator(
            policy=policy_model,
            simulator=dyn_model,
            cfg=rollout_config,
            sampler=sampler,
            device=device,
        )

    if initial_states is not None:
        return policy_rollout_generator.generate_from_states(initial_states)

    if sampler is not None and n_samples is not None:
        return policy_rollout_generator.generate(n_samples)

    __logger__.warning("No initial states or sampler and number of samples provided for rollout dataset generation.")
    return None


def build_rollout_dataset(
    policy_model: nn.Module,
    mpc_config: MPCConfig,
    dyn_model: nn.Module,
    lyap_model: nn.Module,
    rollout_steps: int,
    device: th.device,
    initial_states: NDArray | None = None,
    sampler: StateSampler | None = None,
    n_samples: int | None = None,
) -> MPCDataset | None:
    """Builds a rollout dataset using the provided policy, dynamics, and Lyapunov models. 
    The dataset can be generated either from a set of initial states or by sampling states, 
    using a provided sampler.

    Parameters
    ----------
    policy_model : nn.Module
        The policy model to be used for generating rollouts.
    dyn_model : nn.Module
        The dynamics model to be used for simulating the environment.
    lyap_model : nn.Module
        The Lyapunov model to be used for evaluating the rollouts.
    rollout_steps : int
        The number of steps to rollout for each trajectory.
    device : th.device
        The device on which to perform the rollouts.
    initial_states : NDArray | None, optional
        The initial states from which to start the rollouts, by default None
    sampler : StateSampler | None, optional
        The state sampler to use for generating initial states, by default None
    n_samples : int | None, optional
        The number of rollouts to generate when using a sampler, by default None

    Returns
    -------
    MPCDataset | None
        The generated rollout dataset, or None if no initial states or sampler and number of samples are provided.
    """
    rollout_dataset = build_policy_rollout_dataset(
        policy_model=policy_model,
        mpc_config=mpc_config,
        dyn_model=dyn_model,
        rollout_steps=rollout_steps,
        device=device,
        initial_states=initial_states,
        sampler=sampler,
        n_samples=n_samples,
    )

    if rollout_dataset is None:
        return None

    LyapunovRollout(
        mpc_dataset=rollout_dataset,
        lyap_model=lyap_model,
        device=device,
    ).rollout()
    return rollout_dataset