import torch as th
import torch.nn as nn
import json
import numpy as np

from pathlib import Path
from typing import Any
from dataclasses import is_dataclass, asdict

from ..utils.base_models import MLP
from ..utils.package_logger import get_package_logger

__logger__ = get_package_logger(__name__)


class ConfigEncoder(json.JSONEncoder):
    """Custom JSON Encoder that converts Numpy arrays and Tensors to lists."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
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
        self.global_config: dict[str, Any] | None = None

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

    def save(
        self,
        path: str | Path,
        train_dataset_path: str | Path | None = None,
        val_dataset_path: str | Path | None = None,
        global_config: Any = None,
    ) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        resolved_train = str(Path(train_dataset_path)) if train_dataset_path is not None else getattr(self, "train_dataset_path", None)
        resolved_val = str(Path(val_dataset_path)) if val_dataset_path is not None else getattr(self, "val_dataset_path", None)
        
        if global_config is not None:
            resolved_cfg = asdict(global_config) if is_dataclass(global_config) else dict(global_config)
        else:
            resolved_cfg = getattr(self, "global_config", None)

        self.train_dataset_path = resolved_train
        self.val_dataset_path = resolved_val
        self.global_config = resolved_cfg

        # Safetensors
        model_payload = {
            "state_dict": self.state_dict(),
            "layer_sizes": list(self.layer_sizes),
            "activations": list(self.activations),
        }
        th.save(model_payload, checkpoint_path)

        # JSON
        config_payload = {
            "train_dataset_path": resolved_train,
            "val_dataset_path": resolved_val,
            "global_config": resolved_cfg,
        }
        config_path = checkpoint_path.parent / "config.json"
        
        with open(config_path, "w") as f:
            json.dump(config_payload, f, indent=4, cls=ConfigEncoder)
            
        __logger__.info(f"Saved policy weights to {checkpoint_path} and config to {config_path}")


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

        if layer_sizes is None or activations is None:
            raise ValueError("Missing architecture metadata in model.pt.")

        # load json
        config_path = checkpoint_path.parent / "config.json"
        train_path, val_path, global_cfg = None, None, None
        
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json.load(f)
            train_path = config_data.get("train_dataset_path")
            val_path = config_data.get("val_dataset_path")
            global_cfg = config_data.get("global_config")

        # Modell instanziieren
        u_min = state_dict.get("_u_min", None)
        u_max = state_dict.get("_u_max", None)
        
        model = cls(layer_sizes=layer_sizes, activations=activations, u_min=u_min, u_max=u_max)
        model.load_state_dict(state_dict, strict=strict)
        
        model.train_dataset_path = train_path
        model.val_dataset_path = val_path
        model.global_config = global_cfg
        
        __logger__.info(f"Loaded MLPPolicy from {checkpoint_path}")
        return model
