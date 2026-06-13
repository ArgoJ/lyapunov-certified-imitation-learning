import torch as th
import torch.nn as nn
import logging

from pathlib import Path
from typing import Any

from ..utils.base_models import load_feature_net, save_feature_net

__logger__ = logging.getLogger(__name__)

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
        riccati_p: th.Tensor | None = None,
        fixed_r_factor: bool = False,
        feature_last_init_std: float | None = None,
    ):
        super().__init__()
        self.feature_net = feature_net
        self.state_dim = state_dim
        self.eps = float(eps)
        if x_star is None:
            x_star = th.zeros(state_dim, dtype=th.float32)
        self.register_buffer("x_star", x_star.reshape(1, state_dim))

        if fixed_r_factor:
            self.register_buffer("r_factor", th.eye(state_dim))
        else:
            self.r_factor = nn.Parameter(th.eye(state_dim))

        with th.no_grad():
            initial_scale = th.linalg.norm(self.r_factor, ord="fro").clamp_min(1.0)
        self.register_buffer("feature_scale", initial_scale)

        if riccati_p is not None:
            self.set_riccati_p(riccati_p)
            __logger__.info(
                "Using Riccati value matrix to seed the Lyapunov R factor: \n%s",
                riccati_p,
            )
            __logger__.info(
                "Initialized NeuralLyapunovCandidate with feature scale %.3e from Riccati seed.",
                self.feature_scale.item(),
            )
        
        if feature_last_init_std is not None:
            self._set_last_feature_layer(feature_last_init_std)

    def _set_last_feature_layer(self, std: float) -> None:
        if std <= 0.0:
            return

        linear_layers = [
            module for module in self.feature_net.modules() 
            if isinstance(module, nn.Linear)
        ]

        if not linear_layers:
            __logger__.warning(
                "No Linear layers found in feature_net. Cannot apply last layer initialization."
            )
            return

        last_layer = linear_layers[-1]
        with th.no_grad():
            last_layer.weight.normal_(mean=0.0, std=std)
            if last_layer.bias is not None:
                last_layer.bias.zero_()

    def _pd_matrix(self) -> th.Tensor:
        eye = th.eye(
            self.state_dim,
            dtype=self.r_factor.dtype,
            device=self.r_factor.device,
        )
        return self.eps * eye + self.r_factor.transpose(0, 1) @ self.r_factor

    def set_x_star(self, x_star: th.Tensor) -> None:
        self.x_star.copy_(x_star.reshape(1, -1))

    def set_riccati_p(self, riccati_p: th.Tensor) -> None:
        p_matrix = th.as_tensor(
            riccati_p,
            dtype=self.r_factor.dtype,
            device=self.r_factor.device,
        )
        if p_matrix.shape != (self.state_dim, self.state_dim):
            raise ValueError(
                "riccati_p must have shape "
                f"({self.state_dim}, {self.state_dim}), got {tuple(p_matrix.shape)}."
            )
        if not bool(th.isfinite(p_matrix).all()):
            raise ValueError("riccati_p must contain only finite values.")

        p_sym = 0.5 * (p_matrix + p_matrix.transpose(0, 1))
        eye = th.eye(self.state_dim, dtype=p_sym.dtype, device=p_sym.device)
        shifted = p_sym - self.eps * eye
        eigvals, eigvecs = th.linalg.eigh(shifted)
        scale = max(1.0, float(th.linalg.norm(p_sym, ord=2).item()))
        tol = 1e-6 * scale
        min_eig = float(eigvals.min().item())
        if min_eig < -tol:
            raise ValueError(
                "riccati_p must satisfy P - eps I >= 0 so it can seed R^T R. "
                f"Minimum eigenvalue after subtracting eps is {min_eig:.6e}."
            )

        with th.no_grad():
            factor = th.diag(th.sqrt(eigvals.clamp_min(0.0))) @ eigvecs.transpose(0, 1)
            scale = th.linalg.norm(factor, ord="fro").clamp_min(1.0)
            self.r_factor.copy_(factor)
            self.feature_scale.copy_(scale)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x_star = self.x_star.to(dtype=x.dtype, device=x.device)
        x_star_batch = x_star.expand(x.shape[0], -1)
        phi_x = self.feature_net(x) * self.feature_scale
        phi_x_star = self.feature_net(x_star_batch) * self.feature_scale
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
            "fixed_r_factor": not isinstance(self.r_factor, nn.Parameter),
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
        fixed_r_factor = payload.get("fixed_r_factor", False)
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
            fixed_r_factor=fixed_r_factor,
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