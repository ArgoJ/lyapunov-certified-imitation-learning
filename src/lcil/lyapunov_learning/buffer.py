import torch as th

class DynamicStateBuffer:
    """A dynamic replay buffer for CEGIS counterexamples and initial states, strictly kept on the specified device."""
    def __init__(self, initial_states: th.Tensor, max_size: int):
        self.states = initial_states
        self.max_size = max_size
        self.device = initial_states.device

    def add(self, new_states: th.Tensor) -> None:
        """Adds new states to the buffer and strictly enforces the maximum size."""
        if new_states.numel() == 0:
            return
            
        self.states = th.cat((self.states, new_states), dim=0)
        
        if self.states.shape[0] > self.max_size:
            # Randomly sub-sample to respect the max_buffer limit
            keep_idx = th.randperm(self.states.shape[0], device=self.device)[:self.max_size]
            self.states = self.states[keep_idx]

    def sample(self, batch_size: int) -> th.Tensor:
        """Uniformly samples a batch of states from the buffer."""
        current_size = self.states.shape[0]
        actual_batch_size = min(batch_size, current_size)
        
        # Fast GPU-bound random sampling
        batch_idx = th.randint(
            low=0, 
            high=current_size, 
            size=(actual_batch_size,), 
            device=self.device
        )
        return self.states[batch_idx]

    def __len__(self) -> int:
        return self.states.shape[0]