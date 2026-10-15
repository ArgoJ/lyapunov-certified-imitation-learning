import torch as th
import torch.nn as nn
import logging
import numpy as np
from scipy.linalg import solve_discrete_are

from pathlib import Path
from typing import Any, Literal

from ..utils.base_models import (
    CertifiableTransformerEncoder,
    CertifiableTransformerEncoderLayer,
    build_generator,
    load_feature_net,
    save_feature_net
)

__logger__ = logging.getLogger(__name__)


def _resolve_bound_shape(
    bound: float | list[float] | th.Tensor | None,
    output_dim: int | None,
    name: str,
) -> th.Tensor | None:
    if bound is None:
        return None

    bound_tensor = th.as_tensor(bound, dtype=th.float32)
    if bound_tensor.ndim == 0:
        return bound_tensor

    flat_bound = bound_tensor.flatten()
    if output_dim is not None and flat_bound.numel() != output_dim:
        raise ValueError(
            f"{name} must be a scalar or have {output_dim} elements, got {flat_bound.numel()}."
        )
    return flat_bound


def _resolve_feature_net_dim(feature_net: nn.Module, attr_name: str, layer_index: int) -> int:
    value = getattr(feature_net, attr_name, None)
    if value is not None:
        return int(value)

    layer_dims = getattr(feature_net, "layer_dims", None)
    if layer_dims is None:
        raise AttributeError(
            f"Feature net of type '{type(feature_net).__name__}' must define '{attr_name}' "
            "or 'layer_dims'."
        )
    return int(layer_dims[layer_index])


def _load_feature_net_and_payload(
    path: str | Path,
    map_location: th.device | str = "cpu",
    strict: bool = True,
    feature_net_cls: type[nn.Module] | None = None,
    feature_net_args: tuple[Any, ...] | None = None,
    feature_net_kwargs: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found at '{checkpoint_path}'.")

    payload = th.load(checkpoint_path, map_location=map_location, weights_only=True)
    
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise TypeError("Unsupported checkpoint format. Expected a dict with 'state_dict'.")

    feature_net_path = checkpoint_path.with_name(payload["feature_net_path"])
    feature_net = load_feature_net(
        feature_net_path,
        map_location=map_location,
        strict=strict,
        feature_net_cls=feature_net_cls,
        feature_net_args=feature_net_args,
        feature_net_kwargs=feature_net_kwargs,
    )
    return feature_net, payload



class BaseBoundedPolicy(nn.Module):
    """Base class providing actuator control bounds [u_min, u_max] and clamping."""

    def __init__(
        self, 
        u_min: float | list[float] | th.Tensor | None = None,
        u_max: float | list[float] | th.Tensor | None = None,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if output_dim is None:
            b = u_min if u_min is not None else u_max
            if b is not None and not isinstance(b, (int, float)):
                t = th.as_tensor(b)
                if t.ndim > 0:
                    output_dim = t.numel()
        self.output_dim = output_dim
        self.register_buffer("_u_min", _resolve_bound_shape(u_min, self.output_dim, "u_min"))
        self.register_buffer("_u_max", _resolve_bound_shape(u_max, self.output_dim, "u_max"))
        
    def clamp_u(self, u: th.Tensor) -> th.Tensor:
        """Clamp control action to bounds [u_min, u_max]."""
        if self._u_min is None and self._u_max is None:
            return u
        return th.clamp(u, min=self._u_min, max=self._u_max)



class BoundedPolicy(BaseBoundedPolicy):
    """MLP policy model for imitation learning with control bounds."""

    def __init__(
        self,
        feature_net: nn.Module,
        u_min: float | list[float] | th.Tensor | None = None,
        u_max: float | list[float] | th.Tensor | None = None,
        u_ref: float | list[float] | th.Tensor | None = None,
        x_ref: float | list[float] | th.Tensor | None = None,
    ) -> None:
        """
        Initialize the MLP policy.
        
        Parameters
        ----------
        feature_net : nn.Module
            Feature extraction network that maps raw inputs to features.
        u_min : float or list[float] or torch.Tensor or None, optional
            Lower control bound(s). If provided, policy outputs are clamped from below.
            Can be a scalar or per-control-dimension bounds.
        u_max : float or list[float] or torch.Tensor or None, optional
            Upper control bound(s). If provided, policy outputs are clamped from above.
            Can be a scalar or per-control-dimension bounds.
        u_ref : float or list[float] or torch.Tensor or None, optional
            Reference control(s). If provided, policy outputs are shifted by this reference.
            Can be a scalar or per-control-dimension references.
        x_ref : float or list[float] or torch.Tensor or None, optional
            Reference state(s). If provided, policy inputs are shifted by this reference.
            Can be a scalar or per-state-dimension references.
        """
        output_dim = _resolve_feature_net_dim(feature_net, "output_dim", -1)
        super().__init__(u_min, u_max, output_dim)

        self.feature_net = feature_net
        input_dim = _resolve_feature_net_dim(feature_net, "input_dim", 0) if x_ref is not None else None
        self.register_buffer("_u_ref", _resolve_bound_shape(bound=u_ref, output_dim=output_dim, name="u_ref"))
        self.register_buffer("_x_ref", _resolve_bound_shape(bound=x_ref, output_dim=input_dim, name="x_ref"))
        self.register_buffer("_cached_u_offset", None, persistent=False)
        self._is_fixed: bool = False

    def _reference_offset(self) -> th.Tensor | None:
        """Return the reference offset u_ref - feature_net(x_ref) with caching."""
        if self._x_ref is None and self._u_ref is None:
            return None
        if (self._is_fixed or not th.is_grad_enabled()) and self._cached_u_offset is not None:
            return self._cached_u_offset

        offset = self._u_ref
        if self._x_ref is not None:
            pi = self.feature_net(self._x_ref)
            offset = -pi if offset is None else offset - pi

        if not th.is_grad_enabled() and offset is not None:
            self.register_buffer("_cached_u_offset", offset.detach(), persistent=False)
        return offset

    def prepare_fixed(self) -> None:
        """Precompute and cache fixed reference offset for verification and inference."""
        with th.no_grad():
            self._reference_offset()
        self._is_fixed = True

    def reset_fixed(self) -> None:
        """Reset fixed caching, allowing dynamic computation."""
        self._is_fixed = False
        self._cached_u_offset = None

    def train(self, mode: bool = True) -> "BoundedPolicy":
        """Set training mode and invalidate fixed caches if switching to training."""
        if mode and self._is_fixed:
            self.reset_fixed()
        return super().train(mode)

    def forward_raw(self, x: th.Tensor) -> th.Tensor:
        """Return the unconstrained (unclamped) policy output."""
        u = self.feature_net(x)
        offset = self._reference_offset()
        return u if offset is None else u + offset
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.clamp_u(self.forward_raw(x))

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        feature_net_path = checkpoint_path.with_name(
            checkpoint_path.stem + "_feature_net.pt"
        )
        save_feature_net(self.feature_net, feature_net_path)

        model_payload = {
            "policy_type": self.__class__.__name__,
            "state_dict": self.state_dict(),
            "feature_net_path": feature_net_path.name,
        }
        th.save(model_payload, checkpoint_path)
        __logger__.info(f"Saved policy weights and config to {checkpoint_path.parent}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: th.device | str = "cpu",
        strict: bool = True,
        feature_net_cls: type[nn.Module] | None = None,
        feature_net_args: tuple[Any, ...] | None = None,
        feature_net_kwargs: dict[str, Any] | None = None,
    ) -> "BoundedPolicy":
        feature_net, payload = _load_feature_net_and_payload(
            path,
            map_location=map_location,
            strict=strict,
            feature_net_cls=feature_net_cls,
            feature_net_args=feature_net_args,
            feature_net_kwargs=feature_net_kwargs,
        )
        
        state_dict = payload["state_dict"]
        u_min = state_dict.get("_u_min", None)
        u_max = state_dict.get("_u_max", None)
        u_ref = state_dict.get("_u_ref", None)
        x_ref = state_dict.get("_x_ref", None)
        model = cls(feature_net=feature_net, u_min=u_min, u_max=u_max, u_ref=u_ref, x_ref=x_ref)
        model.load_state_dict(state_dict, strict=strict)
        __logger__.debug(f"Loaded {cls.__name__} from {path}")
        return model



class LQRController(BaseBoundedPolicy):
    """Discrete-time linear quadratic regulator (LQR) policy: u = clamp(-K x).

    Parameters
    ----------
    A : th.Tensor
        Discrete-time state transition matrix of shape (nx, nx).
    B : th.Tensor
        Discrete-time control input matrix of shape (nx, nu).
    R : th.Tensor
        Control cost weight matrix of shape (nu, nu).
    P : th.Tensor | None, optional
        Riccati value matrix of shape (nx, nx). If not provided, `Q` must be given
        to solve the discrete algebraic Riccati equation (DARE) via `scipy.linalg.solve_discrete_are`.
    Q : th.Tensor | None, optional
        State cost weight matrix of shape (nx, nx). Used to compute `P` via DARE
        when `P` is not supplied.
    u_min : float or list[float] or torch.Tensor or None, optional
        Lower control bound(s). If provided, control actions are clamped from below.
    u_max : float or list[float] or torch.Tensor or None, optional
        Upper control bound(s). If provided, control actions are clamped from above.
    """

    def __init__(
        self,
        A: th.Tensor,
        B: th.Tensor,
        R: th.Tensor,
        P: th.Tensor | None = None,
        Q: th.Tensor | None = None,
        u_min: float | list[float] | th.Tensor | None = None,
        u_max: float | list[float] | th.Tensor | None = None,
    ) -> None:
        nu = B.shape[1] if B.ndim == 2 else 1
        super().__init__(u_min=u_min, u_max=u_max, output_dim=nu)

        if P is None:
            if Q is None:
                raise ValueError("Either `P` or `Q` must be provided to LQRController.")
            P = self.solve_dare(A, B, Q, R)

        P_64 = th.as_tensor(P, dtype=th.float64)
        P_sym_64 = 0.5 * (P_64 + P_64.T)
        L_64 = th.linalg.cholesky(P_sym_64)

        self.register_buffer("A", th.as_tensor(A, dtype=th.float32))
        self.register_buffer("B", th.as_tensor(B, dtype=th.float32))
        self.register_buffer("R", th.as_tensor(R, dtype=th.float32))
        self.register_buffer("P", P_sym_64.to(dtype=th.float32))
        self.register_buffer("L", L_64.to(dtype=th.float32))
        if Q is not None:
            self.register_buffer("Q", th.as_tensor(Q, dtype=th.float32))
        self.register_buffer("K", self.compute_gain())

    @staticmethod
    def solve_dare(A: th.Tensor, B: th.Tensor, Q: th.Tensor, R: th.Tensor) -> th.Tensor:
        """Solve Discrete Algebraic Riccati Equation (DARE) using scipy.linalg."""
        a_np = A.detach().cpu().numpy().astype(np.float64)
        b_np = B.detach().cpu().numpy().astype(np.float64)
        q_np = Q.detach().cpu().numpy().astype(np.float64)
        r_np = R.detach().cpu().numpy().astype(np.float64)
        p_np = solve_discrete_are(a_np, b_np, q_np, r_np)
        return th.as_tensor(p_np, dtype=th.float32, device=A.device)

    def compute_gain(self) -> th.Tensor:
        """Compute discrete-time LQR gain K = (R + B^T P B)^(-1) B^T P A using float64."""
        A_64 = self.A.to(dtype=th.float64)
        B_64 = self.B.to(dtype=th.float64)
        R_64 = self.R.to(dtype=th.float64)
        P_64 = self.P.to(dtype=th.float64)

        M = R_64 + B_64.T @ P_64 @ B_64
        rhs = B_64.T @ P_64 @ A_64
        K_64 = th.linalg.solve(M, rhs)
        return K_64.to(dtype=th.float32)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.clamp_u(-(x @ self.K.T))



class BlendedLQRPolicy(BaseBoundedPolicy):
    """Blends local LQR near the origin with a neural policy outside.

    Parameters
    ----------
    policy : BoundedPolicy
        Global learned neural policy.
    lqr : LQRController
        Local LQR controller active near the origin.
    c0 : float
        Upper Lyapunov sublevel bound for pure LQR (V <= c0 -> alpha = 0).
    c1 : float
        Lower Lyapunov sublevel bound for pure policy (V >= c1 -> alpha = 1).
        Must satisfy 0 < c0 < c1.
    """

    def __init__(
        self,
        policy: BoundedPolicy,
        lqr: LQRController,
        c0: float,
        c1: float,
    ) -> None:
        if not (0.0 < c0 < c1):
            raise ValueError(f"Expected 0 < c0 < c1, got c0={c0}, c1={c1}.")

        super().__init__(
            u_min=getattr(policy, "_u_min", None),
            u_max=getattr(policy, "_u_max", None),
            output_dim=policy.output_dim,
        )

        self.policy = policy
        self.lqr = lqr
        self.register_buffer("c0", th.as_tensor(c0, dtype=th.float32))
        self.register_buffer("c1", th.as_tensor(c1, dtype=th.float32))
        self.register_buffer("inv_diff", th.as_tensor(1.0 / (c1 - c0), dtype=th.float32))

    def compute_alpha(self, x: th.Tensor) -> th.Tensor:
        """Compute blending weight alpha in [0, 1].

        alpha = 0 for V <= c0 (pure LQR)
        alpha = 1 for V >= c1 (pure NN policy)
        alpha in (0, 1) linearly interpolated for c0 < V < c1
        Evaluated as sum of squares via Cholesky factor L for tight CROWN bounds.
        """
        v_lqr = (x @ self.lqr.L).pow(2).sum(dim=-1, keepdim=True)
        return th.clamp((v_lqr - self.c0) * self.inv_diff, 0.0, 1.0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        alpha = self.compute_alpha(x)
        return self.clamp_u((1.0 - alpha) * self.lqr(x) + alpha * self.policy(x))



class TransformerPolicy(BaseBoundedPolicy):
    """Transformer-based policy model for imitation learning."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        activation: str = "relu",
        max_seq_len: int = 1,
        causal: bool = True,
        output_mode: Literal["last", "per_step"] = "last",
        u_min: float | list[float] | th.Tensor | None = None,
        u_max: float | list[float] | th.Tensor | None = None,
        seed: int | None = None,
        device: th.device | str = "cpu",
    ) -> None:
        """
        Initialize the transformer policy.

        Parameters
        ----------
        input_dim : int
            State feature dimension per time step.
        output_dim : int
            Action dimension per time step.
        d_model : int, optional
            Internal transformer embedding width.
        nhead : int, optional
            Number of attention heads. Must divide ``d_model``.
        num_encoder_layers : int, optional
            Number of stacked transformer encoder layers.
        dim_feedforward : int, optional
            Hidden width in the encoder feedforward blocks.
        dropout : float, optional
            Dropout probability used inside the transformer.
        activation : str, optional
            Feedforward activation name.
        max_seq_len : int, optional
            Maximum supported sequence length.
        causal : bool, optional
            If ``True``, attention is masked to only access the current and past tokens.
        output_mode : {"last", "per_step"}, optional
            Output reduction mode for sequence inputs. ``"last"`` returns only the
            final token prediction with shape ``(batch, output_dim)``. ``"per_step"``
            returns one prediction per token with shape ``(batch, seq_len, output_dim)``.
        u_min : float or list[float] or torch.Tensor or None, optional
            Lower control bound(s). If provided, policy outputs are clamped from below.
        u_max : float or list[float] or torch.Tensor or None, optional
            Upper control bound(s). If provided, policy outputs are clamped from above.
        """
        super().__init__(u_min=u_min, u_max=u_max, output_dim=output_dim)

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if nhead <= 0:
            raise ValueError("nhead must be positive.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if num_encoder_layers <= 0:
            raise ValueError("num_encoder_layers must be positive.")
        if dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")
        if output_mode not in {"last", "per_step"}:
            raise ValueError("output_mode must be either 'last' or 'per_step'.")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_encoder_layers = int(num_encoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.activation = activation
        self.max_seq_len = int(max_seq_len)
        self.causal = bool(causal)
        self.output_mode = output_mode
        self.device = th.device(device)
        self.seed: int | None = seed

        encoder_layer = CertifiableTransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=False,
        )
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = nn.Parameter(th.zeros(self.max_seq_len, self.d_model))
        self.embedding_dropout = nn.Identity() if self.dropout == 0.0 else nn.Dropout(self.dropout)
        self.encoder = CertifiableTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_encoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )
        self.output_projection = nn.Linear(self.d_model, self.output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply lightweight initialization for learnable projections and norms."""
        gen = build_generator(self.seed, self.device)
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02, generator=gen)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, generator=gen)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _prepare_inputs(self, x: th.Tensor) -> tuple[th.Tensor, bool]:
        if x.ndim == 2:
            return x.unsqueeze(1), True
        if x.ndim == 3:
            return x, False
        raise ValueError(
            "TransformerPolicy expects inputs with shape (batch, features) or "
            "(batch, seq_len, features)."
        )

    def _build_attention_mask(self, seq_len: int, device: th.device) -> th.Tensor | None:
        if not self.causal or seq_len <= 1:
            return None
        return th.triu(
            th.ones(seq_len, seq_len, device=device, dtype=th.bool),
            diagonal=1,
        )

    def _encode_sequence(self, sequence: th.Tensor) -> th.Tensor:
        seq_len = sequence.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        embedded = self.input_projection(sequence)
        embedded = embedded + self.positional_encoding[:seq_len].unsqueeze(0)
        embedded = self.embedding_dropout(embedded)
        encoded = self.encoder(
            embedded,
            mask=self._build_attention_mask(seq_len, embedded.device),
        )
        return self.output_projection(encoded)

    def forward_raw(self, x: th.Tensor) -> th.Tensor:
        """Return the unconstrained (unclamped) policy output."""
        sequence, squeeze_sequence = self._prepare_inputs(x)
        if sequence.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {sequence.size(-1)}."
            )

        raw_u = self._encode_sequence(sequence)
        if squeeze_sequence:
            return raw_u[:, 0, :]
        if self.output_mode == "last":
            return raw_u[:, -1, :]
        return raw_u

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Evaluate the policy and clamp to control bounds [u_min, u_max]."""
        return self.clamp_u(self.forward_raw(x))

    def prepare_fixed(self) -> None:
        """No-op for transformer policy interface compatibility."""
        pass

    def reset_fixed(self) -> None:
        """No-op for transformer policy interface compatibility."""
        pass

    def save(
        self,
        path: str | Path,
    ) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        model_payload = {
            "policy_type": self.__class__.__name__,
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
            "max_seq_len": self.max_seq_len,
            "causal": self.causal,
            "output_mode": self.output_mode,
        }
        th.save(model_payload, checkpoint_path)

        __logger__.info(f"Saved {self.__class__.__name__} weights and config to {checkpoint_path.parent}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: th.device | str = "cpu",
        strict: bool = True,
    ) -> "TransformerPolicy":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found at '{checkpoint_path}'.")

        checkpoint = th.load(checkpoint_path, map_location=map_location, weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise TypeError("Unsupported checkpoint format. Expected a dict with 'state_dict'.")

        state_dict = checkpoint["state_dict"]

        required_fields = (
            "input_dim",
            "output_dim",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "dim_feedforward",
            "dropout",
            "activation",
            "max_seq_len",
            "causal",
            "output_mode",
        )
        missing_fields = [field for field in required_fields if field not in checkpoint]
        if missing_fields:
            raise ValueError(
                "Missing architecture metadata in transformer checkpoint: "
                + ", ".join(missing_fields)
                + "."
            )

        model = cls(
            input_dim=int(checkpoint["input_dim"]),
            output_dim=int(checkpoint["output_dim"]),
            d_model=int(checkpoint["d_model"]),
            nhead=int(checkpoint["nhead"]),
            num_encoder_layers=int(checkpoint["num_encoder_layers"]),
            dim_feedforward=int(checkpoint["dim_feedforward"]),
            dropout=float(checkpoint["dropout"]),
            activation=str(checkpoint["activation"]),
            max_seq_len=int(checkpoint["max_seq_len"]),
            causal=bool(checkpoint["causal"]),
            output_mode=str(checkpoint["output_mode"]),
            u_min=state_dict.get("_u_min", None),
            u_max=state_dict.get("_u_max", None),
        )
        model.load_state_dict(state_dict, strict=strict)

        __logger__.info(f"Loaded {cls.__name__} from {checkpoint_path}")
        return model
