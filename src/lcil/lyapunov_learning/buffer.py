from __future__ import annotations

from collections.abc import Callable

import torch as th


class BoundaryStateBuffer:
    """Cache boundary states with the smallest Lyapunov values seen so far."""

    def __init__(
        self,
        state_dim: int,
        max_size: int,
        device: th.device | str,
        dtype: th.dtype = th.float32,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive.")

        self.max_size = int(max_size)
        self.device = th.device(device)
        self.dtype = dtype
        self.states = th.empty((0, state_dim), dtype=dtype, device=self.device)

    def update(
        self,
        new_states: th.Tensor,
        value_fn: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        """Merge new boundary points and retain the states with the smallest values."""
        if new_states.numel() == 0:
            return

        combined = th.cat((self.states, new_states.to(self.device)), dim=0)
        if combined.shape[0] <= self.max_size:
            self.states = combined
            return

        with th.no_grad():
            values = value_fn(combined).flatten()
            _, keep_idx = th.sort(values, descending=False)
        self.states = combined[keep_idx[: self.max_size]]

    def __len__(self) -> int:
        return self.states.shape[0]


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
        if initial_states.numel() == 0:
            raise ValueError("initial_states cannot be empty.")
        if min_cex_fraction < 0.0 or max_cex_fraction > 1.0 or min_cex_fraction > max_cex_fraction:
            raise ValueError(
                "Invalid CEX fraction bounds. Require 0.0 <= min_cex_fraction <= max_cex_fraction <= 1.0. " 
                f"Got min_cex_fraction={min_cex_fraction}, max_cex_fraction={max_cex_fraction}."
            )
        
        self.states = initial_states.to(device)
        self.cexs = th.empty((0, initial_states.shape[1]), dtype=initial_states.dtype, device=device)
        self.state_buffer_limit = state_buffer_limit
        self.cex_buffer_limit = cex_buffer_limit
        self.device = device
        self.min_cex_fraction = min_cex_fraction
        self.max_cex_fraction = max_cex_fraction
        self.generator = generator
        self.filter_eps = filter_eps

    def _apply_spatial_diversity_filter(
        self,
        states: th.Tensor,
        violations: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:

        if states.shape[0] <= 1:
            return states, violations

        sorted_indices = th.argsort(violations, descending=True)
        sorted_states = states[sorted_indices]
        sorted_violations = violations[sorted_indices]

        distances = th.cdist(sorted_states, sorted_states)

        close_mask = distances < self.filter_eps
        close_mask = th.triu(close_mask, diagonal=1)

        suppress_mask = close_mask.any(dim=0)
        keep_mask = ~suppress_mask

        return (
            sorted_states[keep_mask],
            sorted_violations[keep_mask]
        )

    def add(self, new_states: th.Tensor) -> None:
        """Adds new states to the buffer and strictly enforces the maximum size."""
        if new_states.numel() == 0:
            return

        self.states = th.cat((self.states, new_states), dim=0).to(self.device)

        if self.states.shape[0] > self.state_buffer_limit:
            # Randomly sub-sample to respect the max_buffer limit
            keep_idx = th.randperm(
                self.states.shape[0],
                device=self.device,
                generator=self.generator,
            )[:self.state_buffer_limit]
            self.states = self.states[keep_idx]

    def register_cex(
        self,
        new_cexs: th.Tensor,
        objective: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        """Registers new counterexamples and retains the strongest filtered violations."""
        if new_cexs.numel() == 0:
            return

        combined_cexs = th.cat((new_cexs.to(self.device), self.cexs), dim=0)

        with th.no_grad():
            violation_scores = -objective(combined_cexs).flatten()

        filtered_cexs, filtered_scores = self._apply_spatial_diversity_filter(
            states=combined_cexs,
            violations=violation_scores,
        )
        self.cexs = filtered_cexs[: self.cex_buffer_limit]

    def sample(self, batch_size: int, cex_fraction: float = 0.25) -> th.Tensor:
        """
        Uniformly samples a batch of states from the buffer, injecting a 
        portion of recent counterexamples if available.

        Parameters
        ----------
        batch_size : int
            Total number of states to return.
        cex_fraction: float
            Fraction of the batch reserved for recent CEXs.
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

    @property
    def state_count(self) -> int:
        return self.states.shape[0]

    @property
    def cex_count(self) -> int:
        return self.cexs.shape[0]

    def __len__(self) -> int:
        return self.state_count + self.cex_count