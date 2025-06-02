import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.init_utils import init_weight_block, InitType


class GrowableMLP(nn.Module):
    def __init__(self, layer_sizes, init_type=InitType.IDENTITY):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.init_type = init_type
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            layer.weight.data = init_weight_block(
                layer.in_features, layer.out_features, self.init_type
            )
            layer.bias.data.zero_()

    def grow_layer(self, layer_idx, new_size, init_type=None):
        """Grow a specific layer to a new size."""
        if init_type is None:
            init_type = self.init_type

        old_layer = self.layers[layer_idx]
        old_in_features = old_layer.in_features
        old_out_features = old_layer.out_features

        # Create new layer with larger size
        if layer_idx < len(self.layers) - 1:
            new_layer = nn.Linear(old_in_features, new_size)
            next_layer = nn.Linear(new_size, self.layers[layer_idx + 1].out_features)

            # Initialize new weights
            new_layer.weight.data = init_weight_block(
                old_in_features, new_size, init_type
            )
            next_layer.weight.data = init_weight_block(
                new_size, self.layers[layer_idx + 1].out_features, init_type
            )

            # Copy old biases and zero-initialize new ones
            new_layer.bias.data[:old_out_features] = old_layer.bias.data
            new_layer.bias.data[old_out_features:].zero_()
            next_layer.bias.data = self.layers[layer_idx + 1].bias.data

            # Update layers
            self.layers[layer_idx] = new_layer
            self.layers[layer_idx + 1] = next_layer
            self.layer_sizes[layer_idx + 1] = new_size
        else:
            # Handle output layer growth similarly
            new_layer = nn.Linear(old_in_features, new_size)
            new_layer.weight.data = init_weight_block(
                old_in_features, new_size, init_type
            )
            new_layer.bias.data[:old_out_features] = old_layer.bias.data
            new_layer.bias.data[old_out_features:].zero_()
            self.layers[layer_idx] = new_layer
            self.layer_sizes[layer_idx + 1] = new_size

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)  # No activation on final layer
