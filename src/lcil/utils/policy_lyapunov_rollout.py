import torch as th
import logging

from torch import nn
from numpy.typing import NDArray
from mpc_datagen import MPCDataset

from ..imitation_learning.policy_rollout import PolicyRolloutConfig, PolicyRolloutGenerator, StateSampler
from ..lyapunov_learning.rollout import LyapunovRollout

__logger__ = logging.getLogger(__name__)


def build_rollout_dataset(
    policy_model: nn.Module,
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
    rollout_config = PolicyRolloutConfig.from_mpc_config(
        policy_model.global_config,
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
        rollout_dataset = policy_rollout_generator.generate_from_states(initial_states)
    elif sampler is not None and n_samples is not None:
        rollout_dataset = policy_rollout_generator.generate(n_samples)
    else:
        __logger__.warning("No initial states or sampler and number of samples provided for rollout dataset generation.")
        return None

    LyapunovRollout(
        mpc_dataset=rollout_dataset,
        lyap_model=lyap_model,
        device=device,
    ).rollout()
    return rollout_dataset