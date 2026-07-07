import torch as th
import torch.nn as nn
import logging

from pathlib import Path
from typing import Any

from ..utils.base_models import load_feature_net, save_feature_net

__logger__ = logging.getLogger(__name__)


_COND_WARN_THRESHOLD: float = 1e4


def _symmetrize_matrix(matrix: th.Tensor) -> th.Tensor:
    """Return the symmetric part of a matrix."""
    return 0.5 * (matrix + matrix.transpose(0, 1))

def _scale_riccati(riccati_p: th.Tensor, scale_mode: str | float) -> th.Tensor:
    if scale_mode != "none":
        if scale_mode == "spectral":
            scale_factor = th.linalg.norm(riccati_p, ord=2).item()
        elif scale_mode == "frobenius":
            scale_factor = th.linalg.norm(riccati_p, ord="fro").item()
        else:
            try:
                scale_factor = float(scale_mode)
            except ValueError:
                raise ValueError(
                    f"Invalid riccati_scale mode: {scale_mode}. "
                    "Must be 'none', 'spectral', 'frobenius', or a numeric value."
                )
        riccati_p = riccati_p / scale_factor
    return riccati_p

def _check_riccati_shape(p_matrix: th.Tensor, state_dim: int) -> None:
    if p_matrix.shape != (state_dim, state_dim):
        raise ValueError(
            "riccati_p must have shape "
            f"({state_dim}, {state_dim}), got {tuple(p_matrix.shape)}."
        )
    if not bool(th.isfinite(p_matrix).all()):
        raise ValueError("riccati_p must contain only finite values.")


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
        riccati_scale: str | float = "none",
        fixed_r_factor: bool = False,
        feature_last_init_std: float | None = None,
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
        riccati_p : th.Tensor | None, optional
            The Riccati value matrix, by default None
        riccati_scale : str | float, optional
            The scaling mode for the Riccati matrix, by default "none"
        fixed_r_factor : bool, optional
            Whether the R factor is fixed, by default False
        feature_last_init_std : float | None, optional
            The standard deviation for initializing the last feature layer, by default None
        """
        super().__init__()
        self.feature_net = feature_net
        self.state_dim = state_dim
        self.eps = float(eps)
        if x_star is None:
            x_star = th.zeros(state_dim, dtype=th.float32)

        self.register_buffer("x_star", x_star.reshape(1, state_dim))
        self._setup_r_factor(riccati_p, riccati_scale, fixed_r_factor)
        
        if feature_last_init_std is not None:
            self._set_last_feature_layer(feature_last_init_std)

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

    def _setup_r_factor(self, riccati_p: th.Tensor | None, riccati_scale: str | float, fixed: bool) -> None:
        """Set up the R factor for the Lyapunov candidate."""
        self._cached_pd_matrix: th.Tensor | None = None
        self._cached_pd_version: int = -1
        self._cached_eps: float | None = None

        if fixed:
            self.register_buffer("r_factor", th.eye(self.state_dim))
        else:
            self.r_factor = nn.Parameter(th.eye(self.state_dim))

        if riccati_p is not None:
            self.set_riccati_p(riccati_p, scale_mode=riccati_scale)
        if isinstance(self.r_factor, nn.Parameter):
            self.r_factor.register_post_accumulate_grad_hook(
                self._check_r_factor_conditioning
            )

    def _check_r_factor_conditioning(self, _param: th.Tensor) -> None:
        """Warn if εI + RᵀR becomes ill-conditioned."""
        with th.no_grad():
            pd = self._pd_matrix()
            eigs = th.linalg.eigvalsh(pd)
            lo, hi = eigs[0].item(), eigs[-1].item()
            cond = hi / lo if lo > 0.0 else float("inf")
        if cond > _COND_WARN_THRESHOLD:
            __logger__.warning(
                "PD matrix (εI + RᵀR) ill-conditioned: cond=%.2e "
                "(λ_min=%.2e, λ_max=%.2e). Consider fixing R or "
                "adding regularization.",
                cond, lo, hi,
            )

    @th.no_grad()
    def _calculate_r_factor_from_riccati(self, riccati_p: th.Tensor) -> th.Tensor:
        """Calculate the R factor from the Riccati matrix."""
        p_sym = _symmetrize_matrix(riccati_p)
        eigvals, eigvecs = th.linalg.eigh(p_sym)
        scale = max(1.0, float(th.linalg.norm(p_sym, ord=2).item()))
        tol = 1e-6 * scale
        min_eig = float(eigvals.min().item())
        if min_eig < -tol:
            raise ValueError(
                "riccati_p must satisfy P >= 0 so it can seed R^T R. "
                f"Minimum eigenvalue is {min_eig:.6e}."
            )
        
        factor = th.diag(th.sqrt(eigvals.clamp_min(0.0))) @ eigvecs.transpose(0, 1)
        return factor
    
    def _pd_matrix(self) -> th.Tensor:
        """Return the positive definite matrix εI + RᵀR with lazy caching."""
        needs_grad = (
            isinstance(self.r_factor, nn.Parameter) 
            and self.r_factor.requires_grad 
            and th.is_grad_enabled()
        )
        current_version = self.r_factor._version

        # Cache Hit Check
        if not needs_grad and self._cached_pd_matrix is not None:
            if (
                self._cached_pd_version == current_version 
                and self._cached_eps == self.eps
                and self._cached_pd_matrix.device == self.r_factor.device
                and self._cached_pd_matrix.dtype == self.r_factor.dtype
            ):
                return self._cached_pd_matrix

        # Compute Matrix
        eye = th.eye(
            self.state_dim,
            dtype=self.r_factor.dtype,
            device=self.r_factor.device,
        )
        pd = self.eps * eye + self.r_factor.transpose(0, 1) @ self.r_factor

        # Update Cache
        if not needs_grad:
            self._cached_pd_matrix = pd.detach()
            self._cached_pd_version = current_version
            self._cached_eps = self.eps

        return pd

    def set_x_star(self, x_star: th.Tensor) -> None:
        """Set the equilibrium point x* for the Lyapunov candidate."""
        self.x_star.copy_(x_star.reshape(1, -1))

    def set_riccati_p(self, riccati_p: th.Tensor, scale_mode: str | float = "none") -> None:
        """Set the Riccati value matrix to seed the Lyapunov R factor.

        Parameters
        ----------
        riccati_p : th.Tensor
            The Riccati value matrix.
        scale_mode : str | float, optional
            The scaling mode for the Riccati matrix, by default "none"

        Raises
        ------
        ValueError
            If the Riccati matrix is not positive semi-definite.
        """
        p_matrix = th.as_tensor(
            riccati_p,
            dtype=self.r_factor.dtype,
            device=self.r_factor.device,
        )
        p_matrix = _scale_riccati(p_matrix, scale_mode)
        _check_riccati_shape(p_matrix, self.state_dim)
        __logger__.info("Using Riccati value matrix to seed the Lyapunov R factor: \n%s", p_matrix)
        factor = self._calculate_r_factor_from_riccati(p_matrix)
        with th.no_grad():
            self.r_factor.copy_(factor)
    
    def get_feature_term(self, x: th.Tensor) -> th.Tensor:
        """Compute the feature term |phi(x) - phi(x*)| for the Lyapunov candidate."""
        x_star = self.x_star.to(dtype=x.dtype, device=x.device)
        x_star_batch = x_star.expand(x.shape[0], -1)
        phi_x = self.feature_net(x)
        phi_x_star = self.feature_net(x_star_batch)
        feature_term = th.abs(phi_x - phi_x_star).sum(dim=1, keepdim=True)
        return feature_term

    def get_linear_term(self, x: th.Tensor) -> th.Tensor:
        """Compute the linear term |x - x*| for the Lyapunov candidate."""
        x_star_batch = self.x_star.expand(x.shape[0], -1)
        delta = x - x_star_batch
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