import torch as th
import torch.nn as nn
from pathlib import Path


class InvertedPendulumOnCartPolicy(nn.Module):
    """MLP policy for the inverted pendulum on a cart system."""

    def __init__(
        self,
        feature_net: nn.Module,
    ):
        super().__init__()
        self.net = feature_net
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        cart_pos_vel = x[:, :2]
        theta = x[:, 2:3]
        theta_dot = x[:, 3:]
        
        features = th.cat([
            cart_pos_vel,
            th.sin(theta),
            th.cos(theta),
            theta_dot
        ], dim=-1)
        
        return self.net(features)

    def save(self, *args, **kwargs) -> None:
        """Propagate the save call to the underlying feature network."""
        if hasattr(self.net, "save") and callable(self.net.save):
            self.net.save(*args, **kwargs)
        else:
            raise AttributeError(f"The underlying network {type(self.net).__name__} does not implement a 'save' method.")