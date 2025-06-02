import torch
import torch.nn as nn


class BaseMLP(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        block = []
        for l in range(self.L):
            layer = nn.Linear(layer_dims[l], layer_dims[l + 1])
            # Use Kaiming (He) initialization (optimal for ReLU)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            block.append(layer)
            if l < self.L - 1:
                block.append(nn.ReLU())
        self.model = nn.Sequential(*block)

    def forward(self, x):
        return self.model(x)
