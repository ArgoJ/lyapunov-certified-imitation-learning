import torch
import torch.nn as nn

from .base_models import ICNN, MLP


class LyapunovNet(nn.Module):
    """Lyapunov function approximator using ICNN or MLP.

    Parameters
    ----------
    layer_dims : list[int]
        Layer sizes including input and output dimensions.
    activations : list[str]
        Activation names for each layer transition.
    use_icnn : bool
        Whether to use an ICNN (convex) or a plain MLP.
    """

    def __init__(
        self,
        layer_dims: list[int],
        activations: list[str],
        use_icnn: bool = True,
    ):
        super().__init__()
        if use_icnn:
            self.net = ICNN(layer_dims, activations)
        else:
            self.net = MLP(layer_dims, activations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClosedLoopLyapunovVerifier(nn.Module):
    """
    Dieses Modul repräsentiert den Graphen, der verifiziert werden soll.
    Output: V(f(x, pi(x))) - V(x)
    Ziel: Beweisen, dass der Output < 0 ist (Upper Bound < 0).
    """
    def __init__(
        self, 
        policy_model: nn.Module, 
        lyap_model: nn.Module, 
        dyn_model: nn.Module
    ):
        super(ClosedLoopLyapunovVerifier, self).__init__()
        self.policy = policy_model
        self.lyap = lyap_model
        self.dyn = dyn_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_curr = self.lyap(x)
        u = self.policy(x)
        x_next = self.dyn(x, u)
        v_next = self.lyap(x_next)
        return v_next - v_curr