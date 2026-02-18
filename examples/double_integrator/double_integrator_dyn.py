import torch as th
import torch.nn as nn

class DoubleIntegratorDynamics(nn.Module):
	"""Discrete-time double integrator dynamics."""

	def __init__(self, dt: float = 0.1):
		super().__init__()
		self.dt = dt

	def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
		if u.ndim == 1:
			u = u.unsqueeze(1)
		x_pos = x[:, 0:1]
		x_vel = x[:, 1:2]
		x_next_pos = x_pos + self.dt * x_vel
		x_next_vel = x_vel + self.dt * u
		return th.cat([x_next_pos, x_next_vel], dim=1)