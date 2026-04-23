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
        max_size: int,
        device: th.device,
        cex_buffer_size: int | None = None,
    ):
        self.states = initial_states.to(device)
        self.cexs = th.empty((0, initial_states.shape[1]), dtype=initial_states.dtype, device=device)
        self.max_size = max_size
        self.device = device
        self.cex_buffer_size = max_size if cex_buffer_size is None else int(cex_buffer_size)
        if self.cex_buffer_size <= 0:
            raise ValueError("cex_buffer_size must be positive.")

    def add(self, new_states: th.Tensor) -> None:
        """Adds new states to the buffer and strictly enforces the maximum size."""
        if new_states.numel() == 0:
            return
            
        self.states = th.cat((self.states, new_states), dim=0).to(self.device)
        
        if self.states.shape[0] > self.max_size:
            # Randomly sub-sample to respect the max_buffer limit
            keep_idx = th.randperm(self.states.shape[0], device=self.device)[:self.max_size]
            self.states = self.states[keep_idx]

    def register_cex(
        self,
        new_cexs: th.Tensor,
        objective: Callable[[th.Tensor], th.Tensor] | None = None,
    ) -> None:
        """
        Registers new counterexamples and retains the strongest violations.

        This mirrors the external training buffer semantics where adversarial
        states are ranked by violation severity instead of being kept by age.
        """
        if new_cexs.numel() == 0:
            return

        combined_cexs = th.cat((new_cexs.to(self.device), self.cexs), dim=0)
        if combined_cexs.shape[0] > self.cex_buffer_size:
            if objective is not None:
                with th.no_grad():
                    violation = -objective(combined_cexs).flatten()
                    _, keep_idx = th.sort(violation, descending=True)
                combined_cexs = combined_cexs[keep_idx[: self.cex_buffer_size]]
            else:
                combined_cexs = combined_cexs[: self.cex_buffer_size]

        self.cexs = combined_cexs

    def sample(self, batch_size: int, cex_fraction: float = 0.25) -> th.Tensor:
        """
        Uniformly samples a batch of states from the buffer, injecting a 
        portion of recent counterexamples if available.
        
        Parameters
        ----------
        batch_size : int
            Total number of states to return.
        cex_fraction: float
            Maximum fraction of the batch reserved for recent CEXs.
        """
        current_size = self.states.shape[0]
        actual_batch_size = min(batch_size, current_size)
        
        num_cexs = self.cexs.shape[0]
        
        if num_cexs > 0:
            max_inject = int(actual_batch_size * cex_fraction)
            n_inject = min(num_cexs, max_inject)
            
            # Pick random CEXs to inject into the batch
            cex_idx = th.randint(low=0, high=num_cexs, size=(n_inject,), device=self.device)
            injected_cexs = self.cexs[cex_idx]
            
            # Fill remaining batch with random samples from the main buffer
            n_regular = actual_batch_size - n_inject
            reg_idx = th.randint(low=0, high=current_size, size=(n_regular,), device=self.device)
            regular_states = self.states[reg_idx]
            
            return th.cat((injected_cexs, regular_states), dim=0)
            
        else:
            batch_idx = th.randint(low=0, high=current_size, size=(actual_batch_size,), device=self.device)
            return self.states[batch_idx]

    def __len__(self) -> int:
        return self.states.shape[0]