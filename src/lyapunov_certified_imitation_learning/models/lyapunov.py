"""
Lyapunov network for PVTOL Lyapunov verification.

Architecture from the paper:
    Linear(6, 32, bias=False) -> ReLU
    -> Linear(32, 32, bias=False) -> ReLU
    -> Linear(32, 1, bias=False)

Properties:
    - V(0) = 0 by construction (no bias + ReLU).
    - V(x) >= 0 for all x (enforced by training).
    - auto_LiRPA / alpha-beta-CROWN compatible (only standard layers).
"""
import torch
import torch.nn as nn

STATE_DIM = 6
HIDDEN_DIM = 32


class LyapunovNet(nn.Module):
    """
    Lyapunov candidate V(x).

    Three-layer ReLU network with no biases, so V(0) = 0 automatically.
    Positivity (V(x) > 0 for x != 0) is enforced during training
    via auto_LiRPA lower-bound penalties.
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.lyapunov = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, state_dim)

        Returns
        -------
        V : Tensor, shape (batch, 1)
        """
        return self.lyapunov(x)
