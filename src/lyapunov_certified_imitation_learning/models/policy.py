"""
PVTOL (Planar Vertical Take-Off and Landing) LQR controller.

State:  [x, y, theta, x_dot, y_dot, theta_dot]  (6D)
Action: [u1_bias, u2_bias]  (2D)

Actual thrust forces are computed as:
    F_i = clamp(u_i + m*g/2, [0, F_max])

The LQR gains are pre-computed for the linearized PVTOL dynamics
and match the paper exactly.
"""
import torch
import torch.nn as nn

STATE_DIM = 6
ACTION_DIM = 2

# Pre-computed LQR gains from the paper (linearized PVTOL)
_LQR_GAINS = torch.tensor([
    [ 0.70710678, -0.70710678, -5.03954871,
      1.10781077, -1.82439774, -1.20727555],
    [-0.70710678, -0.70710678,  5.03954871,
     -1.10781077, -1.82439774,  1.20727555],
], dtype=torch.float32)


class PolicyNet(nn.Module):
    """Linear LQR controller for PVTOL.  u = K @ x  (no bias)."""

    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(STATE_DIM, ACTION_DIM, bias=False)
        self.policy.weight = nn.Parameter(_LQR_GAINS.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, 6)
            State vector.

        Returns
        -------
        u : Tensor, shape (batch, 2)
            Control bias (before gravity compensation and clamping).
        """
        return self.policy(x)
