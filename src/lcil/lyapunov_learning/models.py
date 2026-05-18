import torch as th
import torch.nn as nn
from pathlib import Path
from typing import Any

from ..utils.base_models import ICNN, MLP, load_feature_net, save_feature_net


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
        self.r_factor = nn.Parameter(th.eye(state_dim)) # TODO: maybe preset with lqr
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

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        feature_net_path = checkpoint_path.with_name(
            checkpoint_path.stem + "_feature_net.pt"
        )
        save_feature_net(self.feature_net, feature_net_path)

        model_payload = {
            "state_dict": self.state_dict(),
            "feature_net_path": feature_net_path.name,
            "state_dim": self.state_dim,
            "eps": self.eps,
        }
        th.save(model_payload, checkpoint_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: th.device | str = "cpu",
        strict: bool = True,
        feature_net_cls: type[nn.Module] | None = None,
        feature_net_args: tuple[Any, ...] | None = None,
        feature_net_kwargs: dict[str, Any] | None = None,
    ) -> "NeuralLyapunovCandidate":
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'.")
        
        payload = th.load(checkpoint_path, map_location=map_location, weights_only=True)
        state_dim = payload["state_dim"]
        eps = payload["eps"]

        feature_net_path = checkpoint_path.with_name(payload["feature_net_path"])
        feature_net = load_feature_net(
            feature_net_path,
            map_location=map_location,
            strict=strict,
            feature_net_cls=feature_net_cls,
            feature_net_args=feature_net_args,
            feature_net_kwargs=feature_net_kwargs,
        )
        
        model = cls(
            feature_net=feature_net,
            state_dim=state_dim,
            eps=eps,
            x_star=payload["state_dict"].get("x_star", None),
        ).to(map_location)
        model.load_state_dict(payload["state_dict"], strict=strict)
        model.eval()
        return model



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