from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import growth_utils as _growth
from src.utils.init_utils import InitType


class GrowableMLP(nn.Module):
    def __init__(
        self, layer_dims: list[int], init_type: InitType = InitType.IDENTITY
    ) -> None:
        super().__init__()
        self.init_type = init_type
        self._update_architecture(layer_dims)

    def _update_architecture(self, layer_dims: list[int]) -> None:
        """Update internal layers and dimensions."""
        self.layer_dims = layer_dims
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layer = nn.Linear(in_dim, out_dim)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    def grow_layer(
        self, index: int, n_new: int, init_type: Optional[InitType] = None
    ) -> None:
        """Grow output size of layer[index] and adjust next layer's input."""
        if init_type is None:
            init_type = self.init_type
        _growth.grow_layer(self, index=index, n_new=n_new, init_type=init_type)

    def insert_layer(
        self, index: int, new_size: int, init_type: Optional[InitType] = None
    ) -> None:
        """Insert a new hidden layer at position index."""
        if init_type is None:
            init_type = self.init_type
        _growth.insert_layer(self, index=index, new_size=new_size, init_type=init_type)

    def prune_neurons(self, index: int, neuron_indices: list[int]) -> None:
        """Prune specific neurons from the output of layer[index]."""
        _growth.prune_neurons(self, index=index, neuron_indices=neuron_indices)
