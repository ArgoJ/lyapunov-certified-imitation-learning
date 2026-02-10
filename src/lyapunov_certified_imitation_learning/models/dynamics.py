"""
PVTOL (Planar Vertical Take-Off and Landing) dynamics and closed-loop models.

Matches the dynamics from the paper exactly.

State vector (6D):
    [x, y, theta, x_dot, y_dot, theta_dot]

Controller output (2D):
    [u1_bias, u2_bias]

Actual thrust forces:
    F_i = clamp(u_i + m*g/2, [0, F_max])

The closed-loop model ``PVTOLClosedLoop`` is designed to be fed directly
to ``auto_LiRPA.BoundedModule`` for Lyapunov verification.
Its scalar output is V(x_next) - V(x), which should be <= 0.
"""
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Environment parameters (from the paper)
# ---------------------------------------------------------------------------
ENV_PARAMS = {
    "mass": 4.0,
    "inertia": 0.0475,
    "dist": 0.25,
    "gravity": 9.8,
    "dt": 0.05,              # Euler integration step
    "max_force_ub": 39.2,
    "max_force_lb": 0.0,
}

STATE_DIM = 6


# ---------------------------------------------------------------------------
# Standalone dynamics function (for counterexample search / training)
# ---------------------------------------------------------------------------
def pvtol_dynamics(state: torch.Tensor, action: torch.Tensor,
                   env_params: dict | None = None) -> torch.Tensor:
    """
    Discrete-time PVTOL dynamics (forward Euler).

    Parameters
    ----------
    state : Tensor, shape (batch, 6)
    action : Tensor, shape (batch, 2)
    env_params : dict, optional

    Returns
    -------
    state_next : Tensor, shape (batch, 6)
    """
    p = env_params or ENV_PARAMS
    mass = p["mass"]
    inertia = p["inertia"]
    dist = p["dist"]
    gravity = p["gravity"]
    dt = p["dt"]
    fmax = p["max_force_ub"]
    fmin = p["max_force_lb"]

    u_1 = torch.clamp(action[:, 0:1] + mass * gravity / 2.0, min=fmin, max=fmax)
    u_2 = torch.clamp(action[:, 1:2] + mass * gravity / 2.0, min=fmin, max=fmax)

    x_pos   = state[:, 0:1]
    y_pos   = state[:, 1:2]
    theta   = state[:, 2:3]
    x_d     = state[:, 3:4]
    y_d     = state[:, 4:5]
    theta_d = state[:, 5:6]

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    x_change     = x_d * cos_theta - y_d * sin_theta
    y_change     = x_d * sin_theta + y_d * cos_theta
    theta_change = theta_d
    xd_change    = y_d * theta_d - gravity * sin_theta
    yd_change    = -x_d * theta_d - gravity * cos_theta + (u_1 + u_2) / mass
    thetad_change = (u_1 - u_2) * dist / inertia

    return torch.cat([
        x_pos   + dt * x_change,
        y_pos   + dt * y_change,
        theta   + dt * theta_change,
        x_d     + dt * xd_change,
        y_d     + dt * yd_change,
        theta_d + dt * thetad_change,
    ], dim=1)


# ---------------------------------------------------------------------------
# Closed-loop nn.Module for auto_LiRPA verification
# ---------------------------------------------------------------------------
class PVTOLClosedLoop(nn.Module):
    """
    Closed-loop PVTOL model for Lyapunov decrease verification.

    Forward pass:  x  ->  V(f(x, pi(x))) - V(x)

    If the output is <= 0 everywhere in a region, the Lyapunov decrease
    condition holds there.  This module is passed directly to
    ``auto_LiRPA.BoundedModule`` for bound propagation.
    """

    def __init__(self, controller: nn.Module, lyapunov: nn.Module,
                 env_params: dict | None = None):
        super().__init__()
        self.controller = controller
        self.lyapunov = lyapunov

        p = env_params or ENV_PARAMS
        self.mass: float      = p["mass"]
        self.inertia: float   = p["inertia"]
        self.dist: float      = p["dist"]
        self.gravity: float   = p["gravity"]
        self.dt: float        = p["dt"]
        self.max_force_ub: float = p["max_force_ub"]
        self.max_force_lb: float = p["max_force_lb"]

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, 6)

        Returns
        -------
        delta_V : Tensor, shape (batch, 1)
            V(x_next) - V(x)  (want <= 0)
        """
        V_x = self.lyapunov(x)
        action = self.controller(x)
        x_next = self._dynamics(x, action)
        V_x_next = self.lyapunov(x_next)
        return V_x_next - V_x

    # -- internal dynamics ------------------------------------------------
    def _dynamics(self, state: torch.Tensor,
                  action: torch.Tensor) -> torch.Tensor:
        u_1 = torch.clamp(
            action[:, 0:1] + self.mass * self.gravity / 2.0,
            min=self.max_force_lb, max=self.max_force_ub,
        )
        u_2 = torch.clamp(
            action[:, 1:2] + self.mass * self.gravity / 2.0,
            min=self.max_force_lb, max=self.max_force_ub,
        )

        x_pos   = state[:, 0:1]
        y_pos   = state[:, 1:2]
        theta   = state[:, 2:3]
        x_d     = state[:, 3:4]
        y_d     = state[:, 4:5]
        theta_d = state[:, 5:6]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        x_change      = x_d * cos_theta - y_d * sin_theta
        y_change      = x_d * sin_theta + y_d * cos_theta
        theta_change  = theta_d
        xd_change     = y_d * theta_d - self.gravity * sin_theta
        yd_change     = (-x_d * theta_d
                         - self.gravity * cos_theta
                         + (u_1 + u_2) / self.mass)
        thetad_change = (u_1 - u_2) * self.dist / self.inertia

        return torch.cat([
            x_pos   + self.dt * x_change,
            y_pos   + self.dt * y_change,
            theta   + self.dt * theta_change,
            x_d     + self.dt * xd_change,
            y_d     + self.dt * yd_change,
            theta_d + self.dt * thetad_change,
        ], dim=1)

