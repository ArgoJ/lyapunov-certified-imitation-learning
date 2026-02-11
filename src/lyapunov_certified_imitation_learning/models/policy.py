import torch
import torch.nn as nn

from .base_models import MLP


class PolicyNet(nn.Module):
	"""Policy network wrapper using a configurable MLP.

	Parameters
	----------
	layer_dims : list[int]
		Layer sizes including input and output dimensions.
	activations : list[str]
		Activation names for each layer transition.
	"""

	def __init__(self, layer_dims: list[int], activations: list[str]):
		super().__init__()
		self.net = MLP(layer_dims, activations)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)
