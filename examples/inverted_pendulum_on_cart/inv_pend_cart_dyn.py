import torch as th
import torch.nn as nn

from lcil.utils import RK4Integrator

class InvertedPendulumOnCartDynamics(nn.Module):
    """Inverted pendulum on a cart dynamics with RK4-discretized continuous dynamics."""

    def __init__(self, dt: float = 0.1):
        super().__init__()
        self.dt = dt
        self.sys = ...
        self.integrator = RK4Integrator(self.sys, dt=dt)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return self.integrator(x, u)