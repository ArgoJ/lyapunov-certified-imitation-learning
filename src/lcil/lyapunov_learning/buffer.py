from __future__ import annotations

import logging
import torch as th
import numpy as np

from collections.abc import Callable
from scipy.spatial import cKDTree

from ..utils import timeit

__logger__ = logging.getLogger(__name__)

@th.no_grad()
def get_spatial_diversity_indices(
    states: th.Tensor,
    values: th.Tensor,
    filter_eps: float,
    descending: bool = True,
    max_elements: int | None = None,
) -> th.Tensor:
    n = states.shape[0]
    if n <= 1:
        return th.arange(n, device=states.device)

    sorted_indices = th.argsort(values, descending=descending)

    if filter_eps <= 0:
        return sorted_indices if max_elements is None else sorted_indices[:max_elements]

    pts = states[sorted_indices].detach().cpu().numpy()
    tree = cKDTree(pts)

    neighbors = tree.query_ball_tree(tree, r=filter_eps)

    suppressed = np.zeros(n, dtype=bool)
    keep_rel = []

    limit = n if max_elements is None else min(n, max_elements)

    for i in range(n):
        if suppressed[i]:
            continue

        keep_rel.append(i)
        if len(keep_rel) >= limit:
            break

        suppressed[np.asarray(neighbors[i], dtype=np.int64)] = True
        suppressed[i] = False

    keep_rel = th.as_tensor(keep_rel, dtype=th.long, device=sorted_indices.device)
    return sorted_indices[keep_rel]

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
