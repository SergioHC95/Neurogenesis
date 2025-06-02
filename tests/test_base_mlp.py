import torch

from src.models.base_mlp import BaseMLP


def test_base_mlp_init():
    model = BaseMLP([10, 20, 5])
    assert isinstance(model, torch.nn.Module)
    assert len(model.layers) == (3 - 1) * 2
    assert model.layers[0].in_features == 10
    assert model.layers[0].out_features == 20
    assert model.layers[2].in_features == 20
    assert model.layers[2].out_features == 5


def test_base_mlp_forward():
    model = BaseMLP([10, 20, 5])
    x = torch.randn(4, 10)
    output = model(x)
    assert output.shape == (4, 5)
    assert not torch.isnan(output).any()
