import torch as th
import torch.nn as nn

from lcil.utils import RK4Integrator


class InvertedPendulumOnCartContinuousDynamics(nn.Module):
    """Continuous-time inverted pendulum on a cart dynamics.

    State is ``[cart_pos, cart_vel, pole_angle, pole_ang_vel]`` and input is
    scalar force applied to the cart. Includes viscous damping on the cart.
    """

    def __init__(
        self,
        m_cart: float = 1.0,
        m_pole: float = 0.1,
        length: float = 0.5,
        gravity: float = 9.81,
        damping: float = 1.0,
    ):
        super().__init__()
        self.m_cart = float(m_cart)
        self.m_pole = float(m_pole)
        self.length = float(length)
        self.gravity = float(gravity)
        self.damping = float(damping)

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
        m_cart: float = 1.0,
        m_pole: float = 0.1,
        length: float = 0.5,
        gravity: float = 9.81,
        damping: float = 1.0,
    ):
        super().__init__()
        self.dt = float(dt)
        self.sys = InvertedPendulumOnCartContinuousDynamics(
            m_cart=m_cart,
            m_pole=m_pole,
            length=length,
            gravity=gravity,
            damping=damping,
        )
        self.integrator = RK4Integrator(self.sys, dt=dt)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return self.integrator(x, u)
