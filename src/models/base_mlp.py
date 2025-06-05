import torch
import torch.nn as nn


class BaseMLP(nn.Module):
    def __init__(self, layer_dims: list[int]) -> None:
        super().__init__()
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        self.layers = []
        for idx in range(self.L):
            layer = nn.Linear(layer_dims[idx], layer_dims[idx + 1])
            # Use Kaiming (He) initialization (optimal for ReLU)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            if idx < self.L - 1:
                self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
