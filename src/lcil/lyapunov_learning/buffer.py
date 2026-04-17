import torch as th

class DynamicStateBuffer:
    """A dynamic replay buffer for CEGIS counterexamples and initial states, strictly kept on the specified device."""
    def __init__(self, initial_states: th.Tensor, max_size: int, device: th.device):
        self.states = initial_states.to(device)
        self.cexs = th.empty((0, initial_states.shape[1]), dtype=initial_states.dtype, device=device)
        self.max_size = max_size
        self.device = device

    def add(self, new_states: th.Tensor) -> None:
        """Adds new states to the buffer and strictly enforces the maximum size."""
        if new_states.numel() == 0:
            return
            
        self.states = th.cat((self.states, new_states), dim=0).to(self.device)
        
        if self.states.shape[0] > self.max_size:
            # Randomly sub-sample to respect the max_buffer limit
            keep_idx = th.randperm(self.states.shape[0], device=self.device)[:self.max_size]
            self.states = self.states[keep_idx]

    def register_cex(self, new_cexs: th.Tensor) -> None:
        """
        Registers new counterexamples. They are added to the long-term buffer
        and kept separately for priority injection during sampling.
        """
        self.add(self.cexs)  # Add existing CEXs to the main buffer before updating
        self.cexs = new_cexs.to(self.device) if new_cexs.numel() > 0 else th.empty((0, self.states.shape[1]), dtype=self.states.dtype, device=self.device)

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