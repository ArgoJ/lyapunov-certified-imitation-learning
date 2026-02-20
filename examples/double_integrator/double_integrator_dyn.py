import torch as th
import torch.nn as nn

from lcil.utils import RK4Integrator, LinearDynamics

class DoubleIntegratorDynamics(nn.Module):
    """Double integrator with RK4-discretized continuous dynamics."""

    def __init__(self, dt: float = 0.1):
        super().__init__()
        self.dt = dt
        self.lin_sys = LinearDynamics(
            A=th.tensor([[0.0, 1.0], [0.0, 0.0]]),
            B=th.tensor([[0.0], [1.0]]))
        self.integrator = RK4Integrator(self.lin_sys, dt=dt)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return self.integrator(x, u)