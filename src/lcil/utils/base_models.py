import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable
from pathlib import Path


def _get_activation(name: str) -> nn.Module:
    name = name.strip().lower()
    match name:
        case "identity" | "linear":
            return nn.Identity()
        case "relu":
            return nn.ReLU()
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case "softplus":
            return nn.Softplus()
        case "elu":
            return nn.ELU()
        case "leaky_relu":
            return nn.LeakyReLU()
        case "gelu":
            return nn.GELU()
        case _:
            raise ValueError(f"Unknown activation '{name}'.")

def _check_layer_dims(layer_dims: list[int]) -> None:
    if len(layer_dims) < 2:
        raise ValueError("layer_dims must contain at least input and output dimensions.")

def _check_activations(activations: list[str], layer_dims: list[int]) -> None:
    if len(activations) != len(layer_dims) - 1:
        raise ValueError(
            "activations must have the same length as layer_dims minus one."
        )

def save_model_checkpoint(model: nn.Module, save_path: str | os.PathLike) -> None:
    """Save model checkpoint using custom ``save`` when available, else save state dict."""
    save_path = Path(save_path)
    if hasattr(model, "save") and callable(model.save):
        model.save(save_path)
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        th.save(model.state_dict(), save_path)


class LinearDynamics(nn.Module):
    """Simple linear dynamics model :math:`\dot{x} = A x + B u`."""

    def __init__(self, A: th.Tensor, B: th.Tensor):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("B", B)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return F.linear(x, self.A) + F.linear(u, self.B)


class RK4Integrator(nn.Module):
    """Generic fourth-order Runge-Kutta integrator for control systems.

    Parameters
    ----------
    dynamics : Callable[[th.Tensor, th.Tensor], th.Tensor]
        Continuous-time dynamics function :math:`\dot{x} = f(x, u)`.
    dt : float
        Integration step size.
    """

    def __init__(
        self,
        dynamics: Callable[[th.Tensor, th.Tensor], th.Tensor],
        dt: float,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.dt = dt

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        """Integrate one step with RK4."""
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 0.5 * self.dt * k1, u)
        k3 = self.dynamics(x + 0.5 * self.dt * k2, u)
        k4 = self.dynamics(x + self.dt * k3, u)
        return x + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class Linearize(nn.Module):
    """Linearize nonlinear dynamics around an operating point.

    Given ``f(x, u)``, returns Jacobians ``A = df/dx`` and ``B = df/du`` at
    ``(x0, u0)`` and the nominal vector field value ``f0``.
    """

    def __init__(self, dynamics: Callable[[th.Tensor, th.Tensor], th.Tensor]):
        super().__init__()
        self.dynamics = dynamics

    def forward(
        self,
        x0: th.Tensor,
        u0: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Compute local linearization matrices for a batch.

        Parameters
        ----------
        x0 : th.Tensor
            State operating points of shape ``(B, nx)``.
        u0 : th.Tensor
            Input operating points of shape ``(B, nu)``.

        Returns
        -------
        tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]
            ``(A, B, f0, c0)`` where ``A`` has shape ``(B, nx, nx)``,
            ``B`` has shape ``(B, nx, nu)``, ``f0`` has shape ``(B, nx)``,
            and ``c0`` has shape ``(B, nx)``.
        """
        x0 = x0.detach().clone()
        u0 = u0.detach().clone()

        single = (x0.ndim == 1)

        if single:
            x0 = x0.unsqueeze(0)
            u0 = u0.unsqueeze(0)

        def f_single(x_var: th.Tensor, u_var: th.Tensor) -> th.Tensor:
            return self.dynamics(x_var.unsqueeze(0), u_var.unsqueeze(0)).squeeze(0)

        a_fun = th.func.jacrev(f_single, argnums=0)
        b_fun = th.func.jacrev(f_single, argnums=1)

        a_mat = th.func.vmap(a_fun)(x0, u0)   # (B, nx, nx)
        b_mat = th.func.vmap(b_fun)(x0, u0)   # (B, nx, nu)
        f0 = th.func.vmap(f_single)(x0, u0)   # (B, nx)

        ax = th.matmul(a_mat, x0.unsqueeze(-1)).squeeze(-1)  # (B, nx)
        bu = th.matmul(b_mat, u0.unsqueeze(-1)).squeeze(-1)  # (B, nx)
        c0 = f0 - ax - bu

        if single:
            return a_mat[0], b_mat[0], f0[0], c0[0]
        return a_mat.detach(), b_mat.detach(), f0.detach(), c0.detach()


class ResidualBlock(nn.Module):
    """Residual MLP block with two linear layers.

    Parameters
    ----------
    dim : int
        Input/output feature dimension.
    activation : str
        Activation name applied between the two linear layers.
    """

    def __init__(self, dim: int, activation: str):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = _get_activation(activation)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x + self.linear2(self.activation(self.linear1(x)))


class PositiveLinear(nn.Module):
    """Linear layer with non-negative weights via softplus."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(th.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(th.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: th.Tensor) -> th.Tensor:
        weight = F.softplus(self.weight)
        return F.linear(x, weight, self.bias)


class MLP(nn.Module):
    """Simple feed-forward multilayer perceptron.

    Parameters
    ----------
    layer_dims : list[int]
        Layer sizes including input and output dimensions.
    activations : list[str]
        Activation names for each layer transition.
    """

    def __init__(
        self, 
        layer_dims: list[int],
        activations: list[str]
    ):
        super(MLP, self).__init__()

        _check_layer_dims(layer_dims)
        _check_activations(activations, layer_dims)

        layers: list[nn.Module] = []
        for in_dim, out_dim, act_name in zip(
            layer_dims[:-1], layer_dims[1:], activations
        ):
            layers.append(nn.Linear(in_dim, out_dim))
            activation = _get_activation(act_name)
            if not isinstance(activation, nn.Identity):
                layers.append(activation)

        self.net = nn.Sequential(*layers)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)



class ResNet(nn.Module):
    """Residual MLP with skip connections on matching dimensions.

    Parameters
    ----------
    layer_dims : list[int]
        Layer sizes including input and output dimensions.
    activations : list[str]
        Activation names for each layer transition.
    """

    def __init__(
        self, 
        layer_dims: list[int],
        activations: list[str]
    ):
        super(ResNet, self).__init__()

        _check_layer_dims(layer_dims)
        _check_activations(activations, layer_dims)

        layers: list[nn.Module] = []
        for in_dim, out_dim, act_name in zip(
            layer_dims[:-1], layer_dims[1:], activations
        ):
            if in_dim == out_dim:
                layers.append(ResidualBlock(in_dim, act_name))
            else:
                layers.append(nn.Linear(in_dim, out_dim))
                activation = _get_activation(act_name)
                if not isinstance(activation, nn.Identity):
                    layers.append(activation)

        self.net = nn.Sequential(*layers)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)



class ICNN(nn.Module):
    """Input convex neural network (ICNN).

    Parameters
    ----------
    layer_dims : list[int]
        Layer sizes including input and output dimensions.
    activations : list[str]
        Activation names for each layer transition.
    """

    def __init__(
        self, 
        layer_dims: list[int],
        activations: list[str]
    ):
        super(ICNN, self).__init__()

        _check_layer_dims(layer_dims)
        _check_activations(activations, layer_dims)

        self.input_dim = layer_dims[0]
        self.layer_dims = layer_dims
        self.activations = activations

        self.W_x = nn.ModuleList()
        self.W_z = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            in_dim = layer_dims[0]
            out_dim = layer_dims[idx + 1]
            self.W_x.append(nn.Linear(in_dim, out_dim, bias=True))
            if idx > 0:
                prev_dim = layer_dims[idx]
                self.W_z.append(PositiveLinear(prev_dim, out_dim, bias=False))
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        z = None
        for idx, act_name in enumerate(self.activations):
            activation = _get_activation(act_name)
            if idx == 0:
                z = activation(self.W_x[idx](x))
            else:
                z = activation(self.W_x[idx](x) + self.W_z[idx - 1](z))
        return z