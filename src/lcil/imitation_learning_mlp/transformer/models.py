import logging

from pathlib import Path
from typing import Any, Literal

import torch as th
import torch.nn as nn

from mpc_datagen import MPCConfig


__logger__ = logging.getLogger(__name__)


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


class TransformerPolicy(nn.Module):
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
        activation: str = "gelu",
        max_seq_len: int = 1,
        causal: bool = True,
        output_mode: Literal["last", "per_step"] = "last",
        u_min: float | list[float] | th.Tensor | None = None,
        u_max: float | list[float] | th.Tensor | None = None,
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
        super().__init__()

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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=_get_activation(self.activation),
            batch_first=True,
            norm_first=False,
        )
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = nn.Parameter(th.zeros(self.max_seq_len, self.d_model))
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_encoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )
        self.output_projection = nn.Linear(self.d_model, self.output_dim)

        self._init_weights()

        u_min_tensor = self._validate_bound_shape(bound=u_min, output_dim=self.output_dim, name="u_min")
        u_max_tensor = self._validate_bound_shape(bound=u_max, output_dim=self.output_dim, name="u_max")
        self.register_buffer("_u_min", u_min_tensor)
        self.register_buffer("_u_max", u_max_tensor)

        self.global_config: MPCConfig | dict[str, Any] | None = None

    def _init_weights(self) -> None:
        """Apply lightweight initialization for learnable projections and norms."""
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _validate_bound_shape(
        bound: float | list[float] | th.Tensor | None,
        output_dim: int,
        name: str,
    ) -> th.Tensor | None:
        if bound is None:
            return None

        bound_tensor = th.as_tensor(bound, dtype=th.float32)
        if bound_tensor.ndim == 0:
            return bound_tensor

        flat_bound = bound_tensor.flatten()
        if flat_bound.numel() != output_dim:
            raise ValueError(
                f"{name} must be a scalar or have {output_dim} elements, got {flat_bound.numel()}."
            )
        return flat_bound

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

    def _reduce_sequence_output(self, raw_u: th.Tensor, squeeze_sequence: bool) -> th.Tensor:
        if squeeze_sequence:
            return raw_u.squeeze(1)
        if self.output_mode == "last":
            return raw_u[:, -1, :]
        return raw_u

    def forward_raw(self, x: th.Tensor) -> th.Tensor:
        """Return the unconstrained policy output used during imitation fitting."""
        sequence, squeeze_sequence = self._prepare_inputs(x)
        if sequence.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {sequence.size(-1)}."
            )

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
        raw_u = self.output_projection(encoded)
        return self._reduce_sequence_output(raw_u, squeeze_sequence)

    def forward(self, x: th.Tensor) -> th.Tensor:
        raw_u = self.forward_raw(x)
        if self._u_min is None and self._u_max is None:
            return raw_u
        if self._u_min is None:
            return th.clamp(raw_u, max=self._u_max)
        if self._u_max is None:
            return th.clamp(raw_u, min=self._u_min)
        return th.clamp(raw_u, min=self._u_min, max=self._u_max)

    def save(
        self,
        path: str | Path,
        global_config: MPCConfig | dict[str, Any] | None = None,
    ) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if global_config is not None:
            self.global_config = global_config

        if isinstance(self.global_config, MPCConfig):
            resolved_global_cfg = self.global_config.to_dict()
        else:
            resolved_global_cfg = self.global_config

        model_payload = {
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
            "train_data_config": resolved_global_cfg,
        }
        th.save(model_payload, checkpoint_path)

        __logger__.info(f"Saved transformer policy weights and config to {checkpoint_path.parent}")

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
        raw_global_cfg = checkpoint.get("train_data_config", None)
        global_cfg = None if raw_global_cfg is None else MPCConfig.from_dict(raw_global_cfg)

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
        model.global_config = global_cfg

        __logger__.info(f"Loaded TransformerPolicy from {checkpoint_path}")
        return model
