import torch as th
import torch.nn as nn
import json
import logging

from pathlib import Path
from typing import Any
from dataclasses import is_dataclass, asdict
from numpy.typing import NDArray

from mpc_datagen import MPCConfig

from ..utils.base_models import MLP

__logger__ = logging.getLogger(__name__)


class ConfigEncoder(json.JSONEncoder):
    """Custom JSON Encoder that converts Numpy arrays and Tensors to lists."""
    def default(self, obj):
        if isinstance(obj, NDArray):
            return obj.tolist()
        if isinstance(obj, th.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super().default(obj)


class MLPPolicy(nn.Module):
    """MLP policy model for imitation learning."""

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[str],
        u_min: float | list[float] | th.Tensor | None = None,
        u_max: float | list[float] | th.Tensor | None = None,
    ) -> None:
        """
        Initialize the MLP policy.
        
        Parameters
        ----------
        layer_sizes : list of int
            List of layer sizes, including input and output dimensions. For example, [2, 16, 16, 1] would create a network with input dimension 2, two hidden layers of size 16, and output dimension 1.
        activations : list of str
            List of activation function names for each hidden layer. The length should be one less than the length of layer_sizes. Supported activations include "relu", "tanh", "sigmoid", "identity", etc.
        u_min : float or list[float] or torch.Tensor or None, optional
            Lower control bound(s). If provided, policy outputs are clamped from below.
            Can be a scalar or per-control-dimension bounds.
        u_max : float or list[float] or torch.Tensor or None, optional
            Upper control bound(s). If provided, policy outputs are clamped from above.
            Can be a scalar or per-control-dimension bounds.
        """
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.activations = list(activations)
        self.mlp = MLP(layer_dims=layer_sizes, activations=activations)

        output_dim = layer_sizes[-1]
        u_min_tensor = self._validate_bound_shape(bound=u_min, output_dim=output_dim, name="u_min")
        u_max_tensor = self._validate_bound_shape(bound=u_max, output_dim=output_dim, name="u_max")

        self.register_buffer("_u_min", u_min_tensor)
        self.register_buffer("_u_max", u_max_tensor)

        self.train_dataset_path: str | None = None
        self.val_dataset_path: str | None = None
        self.global_config: MPCConfig | dict[str, Any] | None = None

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

    def forward(self, x: th.Tensor) -> th.Tensor:
        u = self.mlp(x)
        return th.clamp(u, min=self._u_min, max=self._u_max)

    @staticmethod
    def _serialize_global_config(global_config: Any) -> dict[str, Any] | None:
        """Convert supported global config payloads to a JSON-serializable dict."""
        if global_config is None:
            return None

        to_dict = getattr(global_config, "to_dict", None)
        if callable(to_dict):
            serialized = to_dict()
            if not isinstance(serialized, dict):
                raise TypeError("global_config.to_dict() must return a dict.")
            return serialized

        if is_dataclass(global_config):
            return asdict(global_config)
        if isinstance(global_config, dict):
            return dict(global_config)
        raise TypeError("global_config must provide to_dict(), be a dataclass, or be a dict.")

    @staticmethod
    def _deserialize_global_config(
        global_config_data: dict[str, Any] | None,
    ) -> MPCConfig | dict[str, Any] | None:
        """Reconstruct ``MPCConfig`` from dict payloads when possible."""
        if global_config_data is None:
            return None

        try:
            return MPCConfig.from_dict(global_config_data)
        except (KeyError, TypeError, ValueError):
            __logger__.warning(
                "Failed to parse global_config with MPCConfig.from_dict; keeping raw dict."
            )
            return global_config_data

    def save(
        self,
        path: str | Path,
        global_config: Any = None,
    ) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        resolved_cfg_source = global_config if global_config is not None else getattr(self, "global_config", None)
        resolved_cfg = self._serialize_global_config(resolved_cfg_source)
        self.global_config = self._deserialize_global_config(resolved_cfg)

        # Pytorch checkpoint with model state and architecture metadata
        model_payload = {
            "state_dict": self.state_dict(),
            "layer_sizes": list(self.layer_sizes),
            "activations": list(self.activations),
            "train_data_config": resolved_cfg
        }
        th.save(model_payload, checkpoint_path)
        __logger__.info(f"Saved policy weights and config to {checkpoint_path.parent}")


    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: th.device | str = "cpu",
        strict: bool = True,
    ) -> "MLPPolicy":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found at '{checkpoint_path}'.")

        checkpoint = th.load(checkpoint_path, map_location=map_location, weights_only=True)
        
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise TypeError("Unsupported checkpoint format. Expected a dict with 'state_dict'.")

        state_dict = checkpoint["state_dict"]
        layer_sizes = checkpoint.get("layer_sizes", None)
        activations = checkpoint.get("activations", None)
        raw_global_cfg = checkpoint.get("train_data_config", None)

        if layer_sizes is None or activations is None:
            raise ValueError("Missing architecture metadata in model.pt.")
        
        global_cfg = cls._deserialize_global_config(raw_global_cfg)

        # Model instantiation
        u_min = state_dict.get("_u_min", None)
        u_max = state_dict.get("_u_max", None)
        
        model = cls(layer_sizes=layer_sizes, activations=activations, u_min=u_min, u_max=u_max)
        model.load_state_dict(state_dict, strict=strict)
        model.global_config = global_cfg
        
        __logger__.info(f"Loaded MLPPolicy from {checkpoint_path}")
        return model
