import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(name: str) -> nn.Module:
    name = name.strip().lower()
    if name in {"identity", "linear", "none"}:
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "softplus":
        return nn.Softplus()
    if name == "elu":
        return nn.ELU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation '{name}'.")

def _check_layer_dims(layer_dims: list[int]) -> None:
    if len(layer_dims) < 2:
        raise ValueError("layer_dims must contain at least input and output dimensions.")

def _check_activations(activations: list[str], layer_dims: list[int]) -> None:
    if len(activations) != len(layer_dims) - 1:
        raise ValueError(
            "activations must have the same length as layer_dims minus one."
        )


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear2(self.activation(self.linear1(x)))


class PositiveLinear(nn.Module):
    """Linear layer with non-negative weights via softplus."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
    def forward(self, x):
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
        
    def forward(self, x):
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = None
        for idx, act_name in enumerate(self.activations):
            activation = _get_activation(act_name)
            if idx == 0:
                z = activation(self.W_x[idx](x))
            else:
                z = activation(self.W_x[idx](x) + self.W_z[idx - 1](z))
        return z