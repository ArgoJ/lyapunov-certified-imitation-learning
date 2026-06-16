from __future__ import annotations

import logging
import torch as th
from collections.abc import Callable

from ..utils import timeit

__logger__ = logging.getLogger(__name__)


def get_spatial_diversity_indices(
    states: th.Tensor,
    values: th.Tensor,
    filter_eps: float,
    descending: bool = True,
    max_elements: int | None = None,
) -> th.Tensor:
    """
    Computes indices of states that satisfy spatial diversity constraints.
    Returns the indices relative to the ORIGINAL input tensors.

    Parameters
    ----------
    states : Tensor
        The states to consider for spatial diversity.
    values : Tensor
        The values associated with each state.
    filter_eps : float
        The minimum distance between selected states.
    descending : bool, optional
        Whether to sort values in descending order. Default is True.
    max_elements : int or None, optional
        The maximum number of elements to keep. Default is None.

    Returns
    -------
    Tensor
        The indices of the selected states relative to the original input tensors.
    """
    if states.shape[0] <= 1:
        return th.arange(states.shape[0], device=states.device)

    # Sort after values (descending or ascending) 
    sorted_indices = th.argsort(values, descending=descending)
    sorted_states = states[sorted_indices]

    keep_idx_relative = []
    remaining_idx = th.arange(sorted_states.shape[0], device=states.device)
    eps_sq = filter_eps ** 2

    # Greedy NMS Loop
    while remaining_idx.numel() > 0:
        first_idx = remaining_idx[0].item()
        keep_idx_relative.append(first_idx)

        # Early Exit
        if max_elements is not None and len(keep_idx_relative) >= max_elements:
            break

        if remaining_idx.numel() == 1:
            break

        current_state = sorted_states[first_idx]
        other_states = sorted_states[remaining_idx[1:]]
        
        dists_sq = th.sum((other_states - current_state) ** 2, dim=1)
        keep_mask = dists_sq >= eps_sq

        remaining_idx = remaining_idx[1:][keep_mask]

    final_keep_relative = th.tensor(keep_idx_relative, dtype=th.long, device=states.device)
    original_indices = sorted_indices[final_keep_relative]

    return original_indices


class AgedTensorPool:
    """Efficiently wraps a state tensor, tracking and managing the FIFO age of each state."""

    def __init__(
        self,
        state_dim: int,
        max_age: int,
        device: th.device | str = "cpu",
        dtype: th.dtype = th.float32,
    ) -> None:
        self.max_age = int(max_age)
        self.device = th.device(device)
        self.dtype = dtype

        self.states = th.empty((0, state_dim), dtype=self.dtype, device=self.device)
        self.ages = th.empty((0,), dtype=th.long, device=self.device)

    def step_time_and_clean(self) -> None:
        """Increments the age of all states and strictly drops expired ones."""
        if self.states.shape[0] == 0:
            return
            
        self.ages += 1
        valid_mask = self.ages <= self.max_age
        
        self.states = self.states[valid_mask]
        self.ages = self.ages[valid_mask]

    def add_fresh(self, new_states: th.Tensor) -> None:
        """Appends new states, initializing their age to 0."""
        if new_states.numel() == 0:
            return
            
        new_states = new_states.to(self.device)
        new_ages = th.zeros(new_states.shape[0], dtype=th.long, device=self.device)

        self.states = th.cat((self.states, new_states), dim=0)
        self.ages = th.cat((self.ages, new_ages), dim=0)

    def filter_by_indices(self, keep_indices: th.Tensor) -> None:
        """Synchronously slices both states and ages based on external logic (e.g., NMS)."""
        self.states = self.states[keep_indices]
        self.ages = self.ages[keep_indices]

    def __len__(self) -> int:
        return self.states.shape[0]


class BoundaryStateBuffer:
    """Cache boundary states with the smallest Lyapunov values seen so far."""

    def __init__(
        self,
        state_dim: int,
        max_size: int,
        max_age: int = 15,
        filter_eps: float = 0.01,
        device: th.device | str = "cpu",
        dtype: th.dtype = th.float32,
    ) -> None:
        self.max_size = int(max_size)
        self.filter_eps = float(filter_eps)

        self._pool = AgedTensorPool(state_dim, max_age, device, dtype)

    @property
    def states(self) -> th.Tensor:
        """Access to the pool's states."""
        return self._pool.states

    def __len__(self) -> int:
        return len(self._pool)

    @timeit(__logger__)
    def update(self, new_states: th.Tensor, value_fn: Callable[[th.Tensor], th.Tensor]) -> None:
        self._pool.step_time_and_clean()
        self._pool.add_fresh(new_states)

        if len(self) == 0: 
            return

        with th.no_grad():
            values = value_fn(self.states).flatten()

        keep_indices = get_spatial_diversity_indices(
            states=self.states,
            values=values,
            filter_eps=self.filter_eps,
            descending=False,
            max_elements=self.max_size,
        )

        self._pool.filter_by_indices(keep_indices)


class DynamicStateBuffer:
    """A dynamic replay buffer for CEGIS counterexamples and initial states, strictly kept on the specified device."""

    def __init__(
        self,
        initial_states: th.Tensor,
        state_buffer_limit: int,
        cex_buffer_limit: int,
        device: th.device,
        min_cex_fraction: float = 0.0,
        max_cex_fraction: float = 1.0,
        generator: th.Generator | None = None,
        filter_eps: float = 0.05,
    ):
        """Initialize the dynamic state buffer.

        Parameters
        ----------
        initial_states : th.Tensor
            Tensor of shape (N, state_dim) containing the initial states.
        state_buffer_limit : int
            Maximum number of states to retain in the buffer.
        cex_buffer_limit : int
            Maximum number of counterexample states to retain in the buffer.
        device : th.device
            The device on which to store the buffer tensors.
        min_cex_fraction : float, optional
            Minimum fraction of counterexample states in sampled batches, by default 0.0
        max_cex_fraction : float, optional
            Maximum fraction of counterexample states in sampled batches, by default 1.0
        generator : th.Generator | None, optional
            Random number generator for sampling, by default None
        filter_eps : float, optional
            Minimum distance between retained states for spatial diversity, by default 0.05

        Raises
        ------
        ValueError
            If initial_states is empty.
        ValueError
            If min_cex_fraction or max_cex_fraction are out of bounds.
        """
        if initial_states.numel() == 0:
            raise ValueError("initial_states cannot be empty.")
        if min_cex_fraction < 0.0 or max_cex_fraction > 1.0 or min_cex_fraction > max_cex_fraction:
            raise ValueError(
                "Invalid CEX fraction bounds. Require 0.0 <= min_cex_fraction <= max_cex_fraction <= 1.0. " 
                f"Got min_cex_fraction={min_cex_fraction}, max_cex_fraction={max_cex_fraction}."
            )
        
        self.states = initial_states.to(device)
        self.state_buffer_limit = state_buffer_limit
        self.cex_buffer_limit = cex_buffer_limit
        self.device = device
        self.min_cex_fraction = min_cex_fraction
        self.max_cex_fraction = max_cex_fraction
        self.generator = generator
        self.filter_eps = filter_eps

        self._cex_pool = AgedTensorPool(initial_states.shape[1], max_age=10, device=device, dtype=initial_states.dtype)

    @property
    def cexs(self) -> th.Tensor:
        """Access to the CEX pool's states."""
        return self._cex_pool.states
    
    @property
    def state_count(self) -> int:
        return self.states.shape[0]

    @property
    def cex_count(self) -> int:
        return len(self._cex_pool)

    def __len__(self) -> int:
        return self.state_count + self.cex_count

    def register_cex(
        self,
        new_cexs: th.Tensor,
        objective: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        """Registers new counterexamples and retains the strongest filtered violations."""
        if new_cexs.numel() == 0:
            return

        self._cex_pool.step_time_and_clean()
        self._cex_pool.add_fresh(new_cexs)

        if len(self._cex_pool) == 0:
            return

        with th.no_grad():
            violation_scores = -objective(self.cexs).flatten()

        keep_indices = get_spatial_diversity_indices(
            states=self.cexs,
            values=violation_scores,
            filter_eps=self.filter_eps,
            descending=True,
            max_elements=self.cex_buffer_limit,
        )
        self._cex_pool.filter_by_indices(keep_indices)

    def sample(self, batch_size: int, cex_fraction: float = 0.25) -> th.Tensor:
        """
        Uniformly samples a batch of states from the buffer, injecting a 
        portion of recent counterexamples if available.

        Parameters
        ----------
        batch_size : int
            Total number of states to return.
        cex_fraction: float, optional
            Fraction of the batch reserved for recent CEXs. default is 0.25
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not (self.min_cex_fraction <= cex_fraction <= self.max_cex_fraction):
            cex_fraction = max(self.min_cex_fraction, min(cex_fraction, self.max_cex_fraction))

        state_count = self.state_count
        cex_count = self.cex_count

        if cex_count == 0:
            batch_idx = th.randint(
                low=0,
                high=state_count,
                size=(batch_size,),
                device=self.device,
                generator=self.generator
            )
            return self.states[batch_idx]

        max_inject = int(batch_size * cex_fraction)
        n_inject = min(cex_count, max_inject)

        cex_idx = th.randint(
            low=0,
            high=cex_count,
            size=(n_inject,),
            device=self.device,
            generator=self.generator
        )
        injected_cexs = self.cexs[cex_idx]

        n_regular = batch_size - n_inject
        reg_idx = th.randint(
            low=0,
            high=state_count,
            size=(n_regular,),
            device=self.device,
            generator=self.generator
        )
        regular_states = self.states[reg_idx]

        return th.cat((injected_cexs, regular_states), dim=0)
