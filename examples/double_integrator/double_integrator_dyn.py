import torch as th
import torch.nn as nn

from lcil.utils import RK4Integrator

class DoubleIntegratorDynamics(nn.Module):
	"""Double integrator with RK4-discretized continuous dynamics."""

	def __init__(self, dt: float = 0.1):
		super().__init__()
		self.dt = dt
		self.integrator = RK4Integrator(self.continuous_dynamics, dt=dt)

	def continuous_dynamics(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
		if u.ndim == 1:
			u = u.unsqueeze(1)
		x_pos = x[:, 0:1]
		x_vel = x[:, 1:2]
		x_dot_pos = x_vel
		x_dot_vel = u
		return th.cat([x_dot_pos, x_dot_vel], dim=1)

	def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
		return self.integrator(x, u)