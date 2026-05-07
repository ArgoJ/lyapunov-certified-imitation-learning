import torch as th
import torch.nn as nn

from lcil.utils import ERKIntegrator, IntegrationMethod, LinearDynamics

class DoubleIntegratorDynamics(nn.Module):
    """Double integrator with ERK-discretized continuous dynamics."""

    def __init__(
        self, 
        dt: float = 0.1,
        method: IntegrationMethod = IntegrationMethod.CLASSICAL_RK4,
    ):
        super().__init__()
        self.dt = dt
        self.lin_sys = LinearDynamics(
            A=th.tensor([[0.0, 1.0], [0.0, 0.0]]),
            B=th.tensor([[0.0], [1.0]]))
        self.integrator = ERKIntegrator.build_compiled(self.lin_sys, dt=dt, method=method)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return self.integrator(x, u)