import torch as th
import torch.nn as nn

from lcil.utils import RK4Integrator

from sys_cfg import PendulumOnCartConfig


class InvertedPendulumOnCartContinuousDynamics(nn.Module):
    """Continuous-time inverted pendulum on a cart dynamics.

    State is ``[cart_pos, cart_vel, pole_angle, pole_ang_vel]`` and input is
    scalar force applied to the cart. Includes viscous damping on the cart.
    """

    def __init__(
        self,
        sys_cfg: PendulumOnCartConfig,
    ):
        super().__init__()
        self.m_cart = float(sys_cfg.m_cart)
        self.m_pole = float(sys_cfg.m_pole)
        self.length = float(sys_cfg.length)
        self.gravity = float(sys_cfg.gravity)
        self.damping = float(sys_cfg.damping)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        cart_vel = x[:, 1]
        theta = x[:, 2]
        theta_dot = x[:, 3]
        force = u[:, 0]

        sin_theta = th.sin(theta)
        cos_theta = th.cos(theta)

        effective_force = force - self.damping * cart_vel
        total_mass = self.m_cart + self.m_pole
        denom = total_mass - self.m_pole * cos_theta * cos_theta

        x_ddot = (
            effective_force
            + self.m_pole * self.length * theta_dot * theta_dot * sin_theta
            - self.m_pole * self.gravity * sin_theta * cos_theta
        ) / denom
        theta_ddot = (
            effective_force * cos_theta
            + self.m_pole * self.length * theta_dot * theta_dot * sin_theta * cos_theta
            + total_mass * self.gravity * sin_theta
        ) / (self.length * denom)

        return th.stack((cart_vel, x_ddot, theta_dot, theta_ddot), dim=1)


class InvertedPendulumOnCartDynamics(nn.Module):
    """Inverted pendulum on a cart dynamics with RK4-discretized continuous dynamics."""

    def __init__(
        self,
        dt: float = 0.1,
        sys_cfg: PendulumOnCartConfig = PendulumOnCartConfig(),
    ):
        super().__init__()
        self.dt = float(dt)
        self.sys = InvertedPendulumOnCartContinuousDynamics(sys_cfg=sys_cfg)
        self.integrator = RK4Integrator(self.sys, dt=dt)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return self.integrator(x, u)
