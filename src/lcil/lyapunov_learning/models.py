import numpy as np
import torch as th
import torch.nn as nn
import logging

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..utils.base_models import load_feature_net, save_feature_net

__logger__ = logging.getLogger(__name__)

_COND_WARN_THRESHOLD: float = 1e4


def has_learnable_r_factor(module: nn.Module) -> bool:
    """Check if the Lyapunov model has a learnable R factor attribute."""
    return hasattr(module, "r_factor") and isinstance(module.r_factor, nn.Parameter)


def check_r_factor_conditioning(eigs: th.Tensor, threshold: float = _COND_WARN_THRESHOLD) -> float:
    """Warn if εI + RᵀR becomes ill-conditioned based on its eigenvalues."""
    lo, hi = eigs[0].item(), eigs[-1].item()
    cond = hi / lo if lo > 0.0 else float("inf")
    if cond > threshold:
        __logger__.warning(
            "PD matrix (εI + RᵀR) ill-conditioned: cond=%.2e "
            "(λ_min=%.2e, λ_max=%.2e). Consider fixing R or "
            "adding regularization.",
            cond, lo, hi,
        )
    return cond


def check_r_factor_kappa(eigs: th.Tensor, kappa: float) -> None:
    """Warn if configured decay rate kappa is incompatible with eigenvalues of εI + RᵀR."""
    lo, hi = eigs[0].item(), eigs[-1].item()
    spectral_ratio = lo / hi if hi > 0.0 else 0.0
    if kappa > spectral_ratio:
        __logger__.warning(
            "Configured decay rate kappa=%.4e exceeds the spectral lower bound "
            "λ_min/λ_max = %.4e (λ_min=%.2e, λ_max=%.2e). The Lyapunov decrease "
            "condition may be impossible or difficult to satisfy.",
            kappa, spectral_ratio, lo, hi,
        )


@runtime_checkable
class LyapunovCandidate(Protocol):
    """Protocol defining the interface for Lyapunov candidate functions.

    A Lyapunov candidate evaluates a scalar value V(x) for given states x.
    Implementations may also decompose V(x) into feature and linear terms.
    """

    def __call__(self, x: th.Tensor) -> th.Tensor:
        """Evaluate the Lyapunov candidate function V(x).

        Parameters
        ----------
        x : th.Tensor
            State tensor of shape (..., state_dim).

        Returns
        -------
        th.Tensor
            Lyapunov candidate values of shape (..., 1).
        """
        ...

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Compute the forward pass of the Lyapunov candidate V(x).

        Parameters
        ----------
        x : th.Tensor
            State tensor of shape (..., state_dim).

        Returns
        -------
        th.Tensor
            Lyapunov candidate values of shape (..., 1).
        """
        ...

    def get_feature_term(self, x: th.Tensor) -> th.Tensor:
        """Compute the feature term of the Lyapunov candidate.

        Parameters
        ----------
        x : th.Tensor
            State tensor of shape (..., state_dim).

        Returns
        -------
        th.Tensor
            Feature term tensor of shape (..., 1).
        """
        ...

    def get_linear_term(self, x: th.Tensor) -> th.Tensor:
        """Compute the linear term of the Lyapunov candidate.

        Parameters
        ----------
        x : th.Tensor
            State tensor of shape (..., state_dim).

        Returns
        -------
        th.Tensor
            Linear term tensor of shape (..., 1).
        """
        ...


class NeuralLyapunovCandidate(nn.Module, LyapunovCandidate):
    """Lyapunov candidate from Eq. (9) in the paper.

    V(x) = |phi(x) - phi(x*)| + ||(eps I + R^T R)(x - x*)||_1
    """

    def __init__(
        self,
        feature_net: nn.Module,
        state_dim: int,
        eps: float = 1e-3,
        x_star: th.Tensor | None = None,
        r_factor: th.Tensor | np.ndarray | None = None,
        fixed_r_factor: bool = False,
        kappa: float | None = None,
    ):
        """Initialize the NeuralLyapunovCandidate.

        Parameters
        ----------
        feature_net : nn.Module
            The feature network phi(x).
        state_dim : int
            The dimension of the state space.
        eps : float, optional
            A small positive constant, by default 1e-3
        x_star : th.Tensor | None, optional
            The equilibrium point, by default None
        r_factor : th.Tensor | np.ndarray | None, optional
            The initial R factor matrix, by default None
        fixed_r_factor : bool, optional
            Whether the R factor is fixed, by default False
        kappa : float | None, optional
            Exponential decay rate for the Lyapunov decrease condition, by default None
        """
        super().__init__()
        self.feature_net = feature_net
        self.state_dim = state_dim
        self.eps = 0.0 if fixed_r_factor else float(eps)
        self.kappa: float | None = float(kappa) if kappa is not None else None
        if x_star is None:
            x_star = th.zeros(state_dim, dtype=th.float32)

        self.register_buffer("x_star", x_star.reshape(1, state_dim))
        self.register_buffer("eye", th.eye(state_dim, dtype=th.float32), persistent=False)
        self.register_buffer("_cached_phi_x_star", None, persistent=False)
        self.register_buffer("_cached_pd_matrix", None, persistent=False)
        self._is_fixed: bool = False
        self._set_last_feature_layer(0.5)
        self._setup_r_factor(r_factor, fixed_r_factor)

    def _set_last_feature_layer(self, std: float) -> None:
        """Set the last linear layer of the feature network to have weights initialized
        with a normal distribution of mean 0 and standard deviation `std`, and biases to zero.
        """
        if std <= 0.0:
            return

        linear_layers = [
            module for module in self.feature_net.modules() 
            if isinstance(module, nn.Linear)
        ]

        if not linear_layers:
            __logger__.warning("No Linear layers found in feature_net. Cannot apply last layer initialization.")
            return

        last_layer = linear_layers[-1]
        with th.no_grad():
            last_layer.weight.normal_(mean=0.0, std=std)
            if last_layer.bias is not None:
                last_layer.bias.zero_()

    def _setup_r_factor(
        self,
        r_factor: th.Tensor | np.ndarray | None,
        fixed: bool,
    ) -> None:
        """Set up the R factor for the Lyapunov candidate."""
        self._cached_pd_matrix = None

        if fixed:
            self.eps = 0.0
            self.register_buffer("r_factor", th.eye(self.state_dim))
        else:
            self.r_factor = nn.Parameter(th.eye(self.state_dim))

        if r_factor is not None:
            self.set_r_factor(r_factor)
    
    def _pd_matrix(self) -> th.Tensor:
        """Return the positive definite matrix εI + RᵀR with caching."""
        if (self._is_fixed or not th.is_grad_enabled()) and self._cached_pd_matrix is not None:
            return self._cached_pd_matrix

        if self.eps > 0.0:
            pd = th.addmm(self.eye, self.r_factor.t(), self.r_factor, beta=self.eps)
        else:
            pd = self.r_factor.t() @ self.r_factor

        if not th.is_grad_enabled():
            self._cached_pd_matrix = pd.detach()

        return pd

    def prepare_fixed(self) -> None:
        """Precompute and cache fixed components for verification and inference."""
        with th.no_grad():
            self._cached_phi_x_star = self.feature_net(self.x_star).squeeze(0).detach()
            self._cached_pd_matrix = self._pd_matrix().detach()
        self._is_fixed = True

    def reset_fixed(self) -> None:
        """Reset fixed caching, allowing dynamic recomputation."""
        self._is_fixed = False
        self._cached_phi_x_star = None
        self._cached_pd_matrix = None

    def train(self, mode: bool = True) -> "NeuralLyapunovCandidate":
        """Set training mode and invalidate fixed caches if switching to training."""
        if mode and self._is_fixed:
            self.reset_fixed()
        return super().train(mode)

    def set_x_star(self, x_star: th.Tensor) -> None:
        """Set the equilibrium point x* for the Lyapunov candidate."""
        self.x_star.copy_(x_star.reshape(1, -1))
        self._cached_phi_x_star = None
        if self._is_fixed:
            with th.no_grad():
                self._cached_phi_x_star = self.feature_net(self.x_star).squeeze(0).detach()

    def set_r_factor(self, r_factor: th.Tensor | np.ndarray) -> None:
        """Set the R factor matrix for the Lyapunov candidate.

        Parameters
        ----------
        r_factor : th.Tensor | np.ndarray
            The R factor matrix of shape (state_dim, state_dim).
        """
        r_mat = th.as_tensor(
            r_factor,
            dtype=self.r_factor.dtype,
            device=self.r_factor.device,
        )
        if r_mat.shape != (self.state_dim, self.state_dim):
            raise ValueError(
                f"r_factor must have shape ({self.state_dim}, {self.state_dim}), got {tuple(r_mat.shape)}."
            )
        if not bool(th.isfinite(r_mat).all()):
            raise ValueError("r_factor must contain only finite values.")

        with th.no_grad():
            self.r_factor.copy_(r_mat)
        self._cached_pd_matrix = None

        pd = self._pd_matrix()
        p_scale = max(1.0, float(th.linalg.matrix_norm(pd, ord=2).item()))
        self._set_last_feature_layer(0.5 * p_scale)
        __logger__.info("Set Lyapunov R factor: \n%s", r_mat)
        __logger__.info("Setting the last layer of the feature net to std: %.3f", 0.5 * p_scale)
        if self._is_fixed:
            self.prepare_fixed()

    def get_feature_term(self, x: th.Tensor) -> th.Tensor:
        """Compute the feature term |phi(x) - phi(x*)| for the Lyapunov candidate."""
        phi_x = self.feature_net(x)
        if (self._is_fixed or not th.is_grad_enabled()) and self._cached_phi_x_star is not None:
            phi_x_star = self._cached_phi_x_star
        else:
            phi_x_star = self.feature_net(self.x_star).squeeze(0)
            if not th.is_grad_enabled():
                self._cached_phi_x_star = phi_x_star.detach()
            else:
                self._cached_phi_x_star = None

        feature_term = th.abs(phi_x - phi_x_star).sum(dim=1, keepdim=True)
        return feature_term

    def get_linear_term(self, x: th.Tensor) -> th.Tensor:
        """Compute the linear term |x - x*| for the Lyapunov candidate."""
        delta = x - self.x_star
        pd_matrix = self._pd_matrix()
        linear_term = th.abs(delta @ pd_matrix).sum(dim=1, keepdim=True)
        return linear_term

    def forward(self, x: th.Tensor) -> th.Tensor:
        feature_term = self.get_feature_term(x)
        linear_term = self.get_linear_term(x)
        return feature_term + linear_term

    def save(self, path: str | Path) -> None:
        """Save the model and its feature network to a checkpoint file."""
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
            "kappa": self.kappa,
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
        kappa = payload.get("kappa", None)
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
            kappa=kappa,
        ).to(map_location)
        model.load_state_dict(payload["state_dict"], strict=strict)
        model.eval()
        return model