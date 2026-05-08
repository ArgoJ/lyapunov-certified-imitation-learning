import importlib
import importlib.util
import inspect
import os
from enum import Enum

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable
from pathlib import Path
from typing import Any


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


def _build_feature_net_payload(
    feature_net: nn.Module,
    save_path: Path,
    suffix: str = "",
    save_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "module": type(feature_net).__module__,
        "class_name": type(feature_net).__name__,
        "module_file": inspect.getsourcefile(type(feature_net)),
    }
    save_kwargs = {} if save_kwargs is None else dict(save_kwargs)

    can_roundtrip_feature_net = (
        hasattr(feature_net, "save")
        and callable(getattr(feature_net, "save"))
        and hasattr(type(feature_net), "load")
        and callable(getattr(type(feature_net), "load"))
    )
    if can_roundtrip_feature_net:
        feature_net_path = save_path.with_name(save_path.stem + f"{suffix}_feature.pt")
        feature_net.save(feature_net_path, **save_kwargs)
        payload["path"] = str(feature_net_path)
        return payload

    if hasattr(feature_net, "layer_dims") and hasattr(feature_net, "activations"):
        payload["init_kwargs"] = {
            "layer_dims": list(feature_net.layer_dims),
            "activations": list(feature_net.activations),
        }
        payload["state_dict"] = feature_net.state_dict()
        return payload

    child_modules = list(feature_net.named_children())
    init_parameters = [
        parameter.name
        for parameter in inspect.signature(type(feature_net).__init__).parameters.values()
        if parameter.name != "self"
    ]
    if len(child_modules) == 1 and len(init_parameters) == 1:
        _, child_module = child_modules[0]
        payload["inner_arg_name"] = init_parameters[0]
        payload["inner"] = _build_feature_net_payload(
            child_module,
            save_path,
            suffix=f"{suffix}_inner",
        )
        payload["state_dict"] = feature_net.state_dict()
        return payload

    raise ValueError(
        f"Cannot serialize feature net of type '{type(feature_net).__name__}'."
    )


def save_feature_net(
    feature_net: nn.Module,
    save_path: str | os.PathLike,
    **save_kwargs: Any,
) -> None:
    """Save a feature net with enough metadata to reconstruct common wrappers and MLP/ICNN-style nets."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_feature_net_payload(
        feature_net,
        save_path,
        save_kwargs=save_kwargs,
    )
    th.save(payload, save_path)


def _load_feature_net_from_payload(
    payload: dict[str, Any],
    map_location: th.device | str,
    strict: bool,
    feature_net_cls: type[nn.Module] | None = None,
    feature_net_args: tuple[Any, ...] | None = None,
    feature_net_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    if feature_net_cls is None:
        module_name = payload.get("module")
        class_name = payload.get("class_name")
        module_file = payload.get("module_file")
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            if module_file is None:
                raise
            module_path = Path(module_file)
            module_spec = importlib.util.spec_from_file_location(
                f"_lcil_feature_net_{module_path.stem}",
                module_path,
            )
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"Could not load module from '{module_path}'.")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        feature_net_cls = getattr(module, class_name)

    feature_net_args = () if feature_net_args is None else feature_net_args
    feature_net_kwargs = {} if feature_net_kwargs is None else dict(feature_net_kwargs)

    if "path" in payload:
        if not hasattr(feature_net_cls, "load") or not callable(getattr(feature_net_cls, "load")):
            raise ValueError(
                "Feature net checkpoint requires a loadable feature_net_cls."
            )
        return feature_net_cls.load(payload["path"], map_location=map_location)

    if "inner" in payload:
        inner_feature_net = _load_feature_net_from_payload(
            payload["inner"],
            map_location=map_location,
            strict=strict,
        )
        feature_net = feature_net_cls(**{payload["inner_arg_name"]: inner_feature_net})
    else:
        init_kwargs = dict(payload.get("init_kwargs", {}))
        init_kwargs.update(feature_net_kwargs)
        feature_net = feature_net_cls(*feature_net_args, **init_kwargs)

    feature_net.load_state_dict(payload["state_dict"], strict=strict)
    return feature_net


def load_feature_net(
    path: str | os.PathLike,
    map_location: th.device | str = "cpu",
    strict: bool = True,
    feature_net_cls: type[nn.Module] | None = None,
    feature_net_args: tuple[Any, ...] | None = None,
    feature_net_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Load a feature net saved with :func:`save_feature_net`."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No feature net checkpoint found at '{checkpoint_path}'.")

    payload = th.load(checkpoint_path, map_location=map_location, weights_only=True)
    feature_net = _load_feature_net_from_payload(
        payload,
        map_location=map_location,
        strict=strict,
        feature_net_cls=feature_net_cls,
        feature_net_args=feature_net_args,
        feature_net_kwargs=feature_net_kwargs,
    )
    feature_net.eval()
    return feature_net


class LinearDynamics(nn.Module):
    r"""Simple linear dynamics model :math:`\dot{x} = A x + B u`."""

    def __init__(self, A: th.Tensor, B: th.Tensor):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("B", B)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return F.linear(x, self.A) + F.linear(u, self.B)


class AffineDynamics(nn.Module):
    r"""Simple affine dynamics model :math:`\dot{x} = A x + B u + c`."""

    def __init__(self, A: th.Tensor, B: th.Tensor, c: th.Tensor):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("c", c)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return F.linear(x, self.A) + F.linear(u, self.B) + self.c


class IntegrationMethod(str, Enum):
    """Supported explicit one-step integration methods."""

    EXPLICIT_EULER = "explicit_euler"
    HEUN2 = "heun2"
    MIDPOINT_RK2 = "midpoint_rk2"
    KUTTA3 = "kutta3"
    HEUN3 = "heun3"
    CLASSICAL_RK4 = "classical_rk4"
    KUTTA_38_RK4 = "kutta_38_rk4"


class ERKIntegrator(nn.Module):
    """Explicit Runge-Kutta one-step integrator for control systems.

    Parameters
    ----------
    dynamics : Callable[[th.Tensor, th.Tensor], th.Tensor]
        Continuous-time dynamics function :math:`\dot{x} = f(x, u)`.
    dt : float
        Integration step size.
    method : IntegrationMethod | str
        Explicit integration scheme to apply.
    """

    def __init__(
        self,
        dynamics: Callable[[th.Tensor, th.Tensor], th.Tensor],
        dt: float,
        method: IntegrationMethod | str = IntegrationMethod.CLASSICAL_RK4,
        abcrown_compatible_ops: bool = False,
        device: th.device | str = "cpu",
        dtype: th.dtype = th.float32,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.dt = float(dt)
        self.abcrown_compatible_ops = bool(abcrown_compatible_ops)
        self.device = th.device(device)
        self.dtype = dtype

        try:
            self.method = method if isinstance(method, IntegrationMethod) else IntegrationMethod(method)
        except ValueError as exc:
            supported = ", ".join(integration_method.value for integration_method in IntegrationMethod)
            raise ValueError(
                f"Unsupported integration method '{method}'. Supported values are: {supported}."
            ) from exc

        a_mat, b_vec = self._get_butcher_tableau(self.method)
        self.register_buffer("a", a_mat)
        self.register_buffer("b", b_vec)
        self.register_buffer("c", a_mat.sum(dim=1))

    @classmethod
    def build_compiled(cls, *args, **kwargs):
        module = cls(*args, **kwargs)
        return th.compile(module)

    def _get_butcher_tableau(
        self,
        method: IntegrationMethod,
    ) -> tuple[th.Tensor, th.Tensor]:
        dtype = self.dtype
        device = self.device

        match method:
            case IntegrationMethod.EXPLICIT_EULER:
                a_mat = th.tensor([[0.0]], dtype=dtype, device=device)
                b_vec = th.tensor([1.0], dtype=dtype, device=device)
            case IntegrationMethod.HEUN2:
                a_mat = th.tensor(
                [[0.0, 0.0], [1.0, 0.0]],
                dtype=dtype,
                device=device,
                )
                b_vec = th.tensor([0.5, 0.5], dtype=dtype, device=device)
            case IntegrationMethod.MIDPOINT_RK2:
                a_mat = th.tensor(
                [[0.0, 0.0], [0.5, 0.0]],
                dtype=dtype,
                device=device,
                )
                b_vec = th.tensor([0.0, 1.0], dtype=dtype, device=device)
            case IntegrationMethod.KUTTA3:
                a_mat = th.tensor(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-1.0, 2.0, 0.0]],
                dtype=dtype,
                device=device,
                )
                b_vec = th.tensor([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=dtype, device=device)
            case IntegrationMethod.HEUN3:
                a_mat = th.tensor(
                [[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0], [0.0, 2.0 / 3.0, 0.0]],
                dtype=dtype,
                device=device,
                )
                b_vec = th.tensor([1.0 / 4.0, 0.0, 3.0 / 4.0], dtype=dtype, device=device)
            case IntegrationMethod.CLASSICAL_RK4:
                a_mat = th.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                    dtype=dtype,
                    device=device,
                )
                b_vec = th.tensor([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=dtype, device=device)
            case IntegrationMethod.KUTTA_38_RK4:
                a_mat = th.tensor(
                    [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0 / 3.0, 0.0, 0.0, 0.0],
                    [-1.0 / 3.0, 1.0, 0.0, 0.0],
                    [1.0, -1.0, 1.0, 0.0],
                ],
                dtype=dtype,
                device=device,
                )
                b_vec = th.tensor([1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0], dtype=dtype, device=device)
            case _:
                raise ValueError(f"Unsupported integration method '{method}'.")

        return a_mat, b_vec

    def _compute_explicit_stages_abcrown_compatible(
        self,
        x: th.Tensor,
        u: th.Tensor,
        a_mat: th.Tensor,
    ) -> list[th.Tensor]:
        stages: list[th.Tensor] = []
        for i in range(a_mat.shape[0]):
            if i == 0:
                stage_state = x
            else:
                stage_delta = th.zeros_like(x)
                for j in range(i):
                    stage_delta = stage_delta + a_mat[i, j] * stages[j]
                stage_state = x + self.dt * stage_delta
            stages.append(self.dynamics(stage_state, u))
        return stages

    def _compute_explicit_stages_vectorized(
        self,
        x: th.Tensor,
        u: th.Tensor,
        a_mat: th.Tensor,
    ) -> th.Tensor:
        stages: list[th.Tensor] = []
        for i in range(a_mat.shape[0]):
            if i == 0:
                stage_state = x
            else:
                prev_stages = th.stack(stages, dim=0)
                stage_state = x + self.dt * th.tensordot(
                    a_mat[i, :i],
                    prev_stages,
                    dims=([0], [0]),
                )
            stages.append(self.dynamics(stage_state, u))
        return th.stack(stages, dim=0)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        """Integrate one step with the configured ERK scheme."""
        a_mat = self.a.to(device=x.device, dtype=x.dtype)
        b_vec = self.b.to(device=x.device, dtype=x.dtype)

        if not self.abcrown_compatible_ops:
            k_tensor = self._compute_explicit_stages_vectorized(x, u, a_mat)
            return x + self.dt * th.tensordot(b_vec, k_tensor, dims=([0], [0]))

        stages = self._compute_explicit_stages_abcrown_compatible(x, u, a_mat)

        update = th.zeros_like(x)
        for idx, stage in enumerate(stages):
            update = update + b_vec[idx] * stage

        return x + self.dt * update


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

        self.layer_dims = list(layer_dims)
        self.activations = list(activations)

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

        self.layer_dims = list(layer_dims)
        self.activations = list(activations)

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