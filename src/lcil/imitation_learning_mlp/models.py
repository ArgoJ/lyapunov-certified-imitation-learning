import torch as th
import torch.nn as nn
from pathlib import Path
from collections.abc import Mapping
from typing import Any


from ..utils.base_models import MLP


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
        global_config: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Save policy weights and architecture metadata to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Target path for ``torch.save``.
        train_dataset_path : str or pathlib.Path or None, optional
            Optional path to the training dataset split associated with this checkpoint.
        val_dataset_path : str or pathlib.Path or None, optional
            Optional path to the validation dataset split associated with this checkpoint.
        global_config : Mapping[str, Any] or None, optional
            Optional JSON-like metadata dict describing global MPC settings used for
            training/rollout reproducibility.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        resolved_train_path = (
            str(Path(train_dataset_path)) if train_dataset_path is not None else self.train_dataset_path
        )
        resolved_val_path = (
            str(Path(val_dataset_path)) if val_dataset_path is not None else self.val_dataset_path
        )
        resolved_global_config = dict(global_config) if global_config is not None else self.global_config

        self.train_dataset_path = resolved_train_path
        self.val_dataset_path = resolved_val_path
        self.global_config = resolved_global_config

        payload = {
            "state_dict": self.state_dict(),
            "layer_sizes": list(self.layer_sizes),
            "activations": list(self.activations),
            "train_dataset_path": resolved_train_path,
            "val_dataset_path": resolved_val_path,
            "global_config": resolved_global_config,
        }
        th.save(payload, checkpoint_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: th.device | str = "cpu",
        strict: bool = True,
    ) -> "MLPPolicy":
        """
        Load an MLP policy model from a checkpoint file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a serialized checkpoint created via ``torch.save``.
            Supports both raw state-dict checkpoints and wrapped checkpoints
            containing a ``"state_dict"`` entry.
        layer_sizes : list of int or None, optional
            Model layer dimensions used to reconstruct the policy architecture.
            If ``None``, values are read from checkpoint metadata.
        activations : list of str or None, optional
            Activation names used to reconstruct the policy architecture.
            If ``None``, values are read from checkpoint metadata.
        map_location : torch.device or str, optional
            Device mapping used during ``torch.load``.
        strict : bool, optional
            Whether to strictly enforce that the loaded checkpoint keys match
            the model keys.

        Returns
        -------
        MLPPolicy
            Loaded policy model with checkpoint weights.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found at '{checkpoint_path}'.")

        checkpoint = th.load(checkpoint_path, map_location=map_location)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise TypeError(
                "Unsupported checkpoint format. Expected a state dict or a dict with 'state_dict'."
            )

        state_dict = checkpoint["state_dict"]
        layer_sizes = checkpoint.get("layer_sizes", None)
        activations = checkpoint.get("activations", None)
        train_dataset_path = checkpoint.get("train_dataset_path", None)
        val_dataset_path = checkpoint.get("val_dataset_path", None)
        global_config = checkpoint.get("global_config", None)

        if layer_sizes is None or activations is None:
            raise ValueError(
                "Missing architecture metadata. Provide 'layer_sizes' and 'activations', "
                "or save checkpoints via MLPPolicy.save(...)."
            )

        u_min = state_dict.get("_u_min", None)
        u_max = state_dict.get("_u_max", None)
        model = cls(layer_sizes=layer_sizes, activations=activations, u_min=u_min, u_max=u_max)
        model.load_state_dict(state_dict, strict=strict)
        model.train_dataset_path = str(train_dataset_path) if train_dataset_path is not None else None
        model.val_dataset_path = str(val_dataset_path) if val_dataset_path is not None else None
        model.global_config = dict(global_config) if isinstance(global_config, Mapping) else None
        return model
