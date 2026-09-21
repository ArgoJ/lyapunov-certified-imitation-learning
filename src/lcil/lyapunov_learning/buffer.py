from __future__ import annotations

import logging
import torch as th
from collections.abc import Callable

from .utils import get_bounded_fraction
from .sampling import sample_mixed_batch, sample_box_rejection_states

__logger__ = logging.getLogger(__name__)

@th.no_grad()
def normalize_states(states: th.Tensor, lb: th.Tensor, ub: th.Tensor) -> th.Tensor:
    """Normalizes the states by bounds"""
    return ((states - lb) / (ub - lb).clamp_min(1e-6))
    

@th.no_grad()
def get_spatial_diversity_indices(
    states: th.Tensor,
    values: th.Tensor,
    filter_eps: float,
    descending: bool = True,
    max_elements: int | None = None,
    lb: th.Tensor | None = None,
    ub: th.Tensor | None = None,
) -> th.Tensor:
    """Select diverse states by suppressing close spatial neighbors.

    Uses GPU-vectorized spatial voxel hashing with normalized coordinates/epsilon
    to ensure consistent diversity across dimensions with different scales and avoid
    CPU-GPU synchronization overhead.
    """
    if states.ndim == 1:
        states = states.unsqueeze(-1)

    n, d = states.shape
    if n <= 1:
        return th.arange(n, device=states.device)

    sorted_indices = th.argsort(values, descending=descending)

    if filter_eps <= 0.0:
        return sorted_indices if max_elements is None else sorted_indices[:max_elements]

    sorted_states = states[sorted_indices]

    if lb is not None and ub is not None:
        lb_t = lb.to(device=states.device, dtype=states.dtype).view(1, d)
        ub_t = ub.to(device=states.device, dtype=states.dtype).view(1, d)
        widths = (ub_t - lb_t).clamp_min(1e-6)
        min_coords = lb_t
    else:
        min_coords = sorted_states.min(dim=0).values.view(1, d)
        max_coords = sorted_states.max(dim=0).values.view(1, d)
        widths = (max_coords - min_coords).clamp_min(1e-6)

    # Normalize step size per dimension so filter_eps is relative to the domain bounds
    step = (widths * filter_eps).clamp_min(1e-6)
    coords = th.floor((sorted_states - min_coords) / step).to(th.int64)

    unique_coords, inverse_indices = th.unique(coords, dim=0, return_inverse=True)
    perm = th.arange(n, device=states.device)
    first_occurrences = th.zeros(
        len(unique_coords), dtype=th.long, device=states.device
    ).scatter_reduce(
        0, inverse_indices, perm, reduce="amin", include_self=False
    )

    keep_rel = first_occurrences.sort().values
    if max_elements is not None:
        keep_rel = keep_rel[:max_elements]

    return sorted_indices[keep_rel]

class AgedTensorPool:
    """Efficiently wraps a state tensor, tracking and managing the FIFO age of each state."""

    def __init__(
        self,
        state_dim: int,
        max_age: int | None,
        device: th.device | str = "cpu",
        dtype: th.dtype = th.float32,
    ) -> None:
        self.max_age = int(max_age) if (max_age is not None and max_age >= 0) else None
        self.device = th.device(device)
        self.dtype = dtype

        self.states = th.empty((0, state_dim), dtype=self.dtype, device=self.device)
        self.ages = th.empty((0,), dtype=th.long, device=self.device)

    def states_of_age(self, age: int) -> th.Tensor:
        """Return all currently stored states with the given age."""
        if len(self) == 0:
            return self.states
        return self.states[self.ages == age]

    def step_time_and_clean(self) -> None:
        """Increments the age of all states and strictly drops expired ones."""
        if self.states.shape[0] == 0:
            return
            
        self.ages += 1
        if self.max_age is not None:
            valid_mask = self.ages <= self.max_age
            self.states = self.states[valid_mask]
            self.ages = self.ages[valid_mask]

    def add_fresh(self, new_states: th.Tensor) -> None:
        """Appends new states, initializing their age to 0."""
        if new_states.numel() == 0:
            return
            
        new_states = new_states.to(device=self.device, dtype=self.dtype)
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
        max_age: int | None = None,
        filter_eps: float = 0.01,
        device: th.device | str = "cpu",
        dtype: th.dtype = th.float32,
        lb: th.Tensor | None = None,
        ub: th.Tensor | None = None,
    ) -> None:
        self.max_size = int(max_size)
        self.filter_eps = float(filter_eps)
        self.lb = lb.to(device) if lb is not None else None
        self.ub = ub.to(device) if ub is not None else None

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
            lb=self.lb,
            ub=self.ub,
        )

        self._pool.filter_by_indices(keep_indices)


class CEGISBuffer:
    """A dynamic replay buffer for CEGIS counterexamples and initial states, strictly kept on the specified device."""

    def __init__(
        self,
        initial_states: th.Tensor,
        state_buffer_limit: int,
        cex_buffer_limit: int,
        lb: th.Tensor,
        ub: th.Tensor,
        min_cex_fraction: float = 0.0,
        max_cex_fraction: float = 1.0,
        generator: th.Generator | None = None,
        filter_eps: float = 0.05,
        max_cex_age: int = 5,
        device: th.device | str = "cpu",
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
        lb : th.Tensor
            Lower bounds of shape (state_dim,) for uniform resampling.
        ub : th.Tensor
            Upper bounds of shape (state_dim,) for uniform resampling.
        min_cex_fraction : float, optional
            Minimum fraction of counterexample states in sampled batches, by default 0.0
        max_cex_fraction : float, optional
            Maximum fraction of counterexample states in sampled batches, by default 1.0
        generator : th.Generator | None, optional
            Random number generator for sampling, by default None
        filter_eps : float, optional
            Minimum distance between retained states for spatial diversity, by default 0.05
        max_cex_age : int, optional
            Maximum age for counterexamples in the buffer before they are automatically removed, by default 5
        device : th.device | str, optional
            The device on which to store the buffer tensors, by default "cpu".

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
        self.lb = lb.to(device)
        self.ub = ub.to(device)
        self.device = device
        self.min_cex_fraction = min_cex_fraction
        self.max_cex_fraction = max_cex_fraction
        self.generator = generator
        self.filter_eps = filter_eps

        self._cex_pool = AgedTensorPool(
            initial_states.shape[1], 
            max_age=max_cex_age,
            device=device,
            dtype=initial_states.dtype,
        )

    @property
    def cexs(self) -> th.Tensor:
        """Access to the CEX pool's states."""
        return self._cex_pool.states

    @property
    def newest_cexs(self) -> th.Tensor:
        return self._cex_pool.states_of_age(0)
    
    @property
    def state_count(self) -> int:
        return self.states.shape[0]

    @property
    def cex_count(self) -> int:
        return len(self._cex_pool)

    def __len__(self) -> int:
        return self.state_count + self.cex_count

    @th.no_grad()
    def resample_states(
        self,
        value_fn: Callable[[th.Tensor], th.Tensor],
        rho_estimate: float,
        rho_margin: float = 2.0,
    ) -> None:
        """Replace the regular state pool with rejection-sampled states focused on the ρ-sublevel region.

        Samples uniformly from the training bounds, evaluates ``V(x)`` via
        *value_fn*, and keeps states with ``V(x) ≤ rho_margin * ρ``.  If fewer
        than 50% of the target count pass the acceptance test, the remaining
        slots are filled with the lowest-V rejected candidates to ensure the
        buffer never starves.

        Parameters
        ----------
        value_fn : Callable[[th.Tensor], th.Tensor]
            Maps ``(N, state_dim) → (N, 1)`` Lyapunov values.
        rho_estimate : float
            Current ρ estimate from boundary analysis.
        rho_margin : float, optional
            Multiplicative factor over ρ for the acceptance threshold, by default 2.0.
        """
        rho_target = rho_margin * rho_estimate
        rho_scale = max(rho_estimate, 1e-9)
        
        def score_fn(x: th.Tensor) -> th.Tensor:
            return (rho_target - value_fn(x).flatten()) / rho_scale

        self.states = sample_box_rejection_states(
            lb=self.lb,
            ub=self.ub,
            target_count=self.state_buffer_limit,
            score_fn=score_fn,
            oversample_factor=4,
            device=self.device,
            generator=self.generator,
        )

    def register_cex(
        self,
        new_cexs: th.Tensor,
        objective: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        """Registers new counterexamples and retains the strongest violations.

        Only new counterexamples are filtered for spatial diversity before being
        added to the pool. Existing counterexamples in the pool are not spatially
        filtered against each other or against new counterexamples.
        """
        self._cex_pool.step_time_and_clean()

        if new_cexs.numel() > 0:
            with th.no_grad():
                new_violation_scores = -objective(new_cexs).flatten()

            keep_indices = get_spatial_diversity_indices(
                states=new_cexs,
                values=new_violation_scores,
                filter_eps=self.filter_eps,
                descending=True,
                lb=self.lb,
                ub=self.ub,
            )
            self._cex_pool.add_fresh(new_cexs[keep_indices])

        if len(self._cex_pool) == 0:
            return

        if len(self._cex_pool) > self.cex_buffer_limit:
            with th.no_grad():
                violation_scores = -objective(self.cexs).flatten()
            top_indices = th.topk(
                violation_scores,
                k=self.cex_buffer_limit,
                largest=True,
            ).indices
            self._cex_pool.filter_by_indices(top_indices)

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
        cex_fraction = get_bounded_fraction(
            base=cex_fraction,
            min=self.min_cex_fraction,
            max=self.max_cex_fraction,
        )
        return sample_mixed_batch(
            regular_states=self.states,
            cexs=self.cexs,
            batch_size=batch_size,
            cex_fraction=cex_fraction,
            device=self.device,
            generator=self.generator,
        )