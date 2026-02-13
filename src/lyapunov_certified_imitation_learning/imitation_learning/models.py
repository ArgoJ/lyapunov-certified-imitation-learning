import torch as th
import torch.nn as nn


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
        self.mlp = MLP(layer_dims=layer_sizes, activations=activations)

        output_dim = layer_sizes[-1]
        u_min_tensor = self._validate_bound_shape(bound=u_min, output_dim=output_dim, name="u_min")
        u_max_tensor = self._validate_bound_shape(bound=u_max, output_dim=output_dim, name="u_max")

        self.register_buffer("_u_min", u_min_tensor)
        self.register_buffer("_u_max", u_max_tensor)

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