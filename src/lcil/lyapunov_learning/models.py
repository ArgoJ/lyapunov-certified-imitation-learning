import torch as th
import torch.nn as nn

from ..utils.base_models import ICNN, MLP


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

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)

# TODO: Change names to be control confirm
class NeuralLyapunovCandidate(nn.Module):
    """Lyapunov candidate from Eq. (9) in the paper.

    V(x) = |phi(x) - phi(x*)| + ||(eps I + R^T R)(x - x*)||_1
    """

    def __init__(
        self,
        feature_net: nn.Module,
        state_dim: int,
        eps: float = 1e-3,
        x_star: th.Tensor | None = None,
    ):
        super().__init__()
        self.feature_net = feature_net
        self.state_dim = state_dim
        self.eps = float(eps)
        self.r_factor = nn.Parameter(th.eye(state_dim))
        if x_star is None:
            x_star = th.zeros(state_dim, dtype=th.float32)
        self.register_buffer("x_star", x_star.reshape(1, state_dim))

    def _pd_matrix(self) -> th.Tensor:
        eye = th.eye(
            self.state_dim,
            dtype=self.r_factor.dtype,
            device=self.r_factor.device,
        )
        return self.eps * eye + self.r_factor.transpose(0, 1) @ self.r_factor

    def set_x_star(self, x_star: th.Tensor) -> None:
        self.x_star.copy_(x_star.reshape(1, -1))

    def forward(self, x: th.Tensor) -> th.Tensor:
        x_star = self.x_star.to(dtype=x.dtype, device=x.device)
        x_star_batch = x_star.expand(x.shape[0], -1)
        phi_x = self.feature_net(x)
        phi_x_star = self.feature_net(x_star_batch)
        feature_term = th.abs(phi_x - phi_x_star).sum(dim=1, keepdim=True)

        delta = x - x_star_batch
        pd_matrix = self._pd_matrix()
        linear_term = th.abs(delta @ pd_matrix.transpose(0, 1)).sum(
            dim=1,
            keepdim=True,
        )
        return feature_term + linear_term


class QuadraticLyapunovCandidate(nn.Module):
    """Quadratic Lyapunov candidate from Eq. (10) in the paper."""

    def __init__(
        self,
        state_dim: int,
        eps: float = 1e-3,
        x_star: th.Tensor | None = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.eps = float(eps)
        self.r_factor = nn.Parameter(th.eye(state_dim))
        if x_star is None:
            x_star = th.zeros(state_dim, dtype=th.float32)
        self.register_buffer("x_star", x_star.reshape(1, state_dim))

    def _pd_matrix(self) -> th.Tensor:
        eye = th.eye(
            self.state_dim,
            dtype=self.r_factor.dtype,
            device=self.r_factor.device,
        )
        return self.eps * eye + self.r_factor.transpose(0, 1) @ self.r_factor

    def forward(self, x: th.Tensor) -> th.Tensor:
        goal = self.x_star.to(dtype=x.dtype, device=x.device)
        delta = x - goal.expand(x.shape[0], -1)
        pd_matrix = self._pd_matrix()
        value = (delta @ pd_matrix) * delta
        return value.sum(dim=1, keepdim=True)