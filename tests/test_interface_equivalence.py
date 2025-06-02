import pytest
import torch

from src.utils.growth_utils import InitType, grow_layer, insert_layer, prune_neurons


@pytest.mark.parametrize("method", ["wrapper", "function"])
def test_insert_layer_equivalence(growable_mlp, method):
    model = growable_mlp
    input_dim = model.layer_dims[0]
    insert_size = 7
    x = torch.randn(3, input_dim)

    if method == "wrapper":
        model.insert_layer(index=0, new_size=insert_size, init_type=InitType.IDENTITY)
    else:
        insert_layer(model, index=0, new_size=insert_size, init_type=InitType.IDENTITY)

    assert model.layer_dims[1] == insert_size
    assert model.layers[0].out_features == insert_size
    assert model.layers[1].in_features == insert_size

    y = model(x)
    assert y.shape == (3, model.layer_dims[-1])


@pytest.mark.parametrize("method", ["wrapper", "function"])
def test_grow_layer_equivalence(growable_mlp, method):
    model = growable_mlp
    n_new = 4
    in_dim, mid_dim, out_dim = model.layer_dims
    x = torch.randn(2, in_dim)

    if method == "wrapper":
        model.grow_layer(index=0, n_new=n_new, init_type=InitType.IDENTITY)
    else:
        grow_layer(model, index=0, n_new=n_new, init_type=InitType.IDENTITY)

    assert model.layer_dims[1] == mid_dim + n_new
    assert model.layers[0].out_features == mid_dim + n_new
    assert model.layers[1].in_features == mid_dim + n_new

    y = model(x)
    assert y.shape == (2, out_dim)


@pytest.mark.parametrize("method", ["wrapper", "function"])
def test_prune_neurons_equivalence(growable_mlp, method):
    model = growable_mlp
    prune_count = 3
    original_mid = model.layer_dims[1]
    keep_dim = original_mid - prune_count
    x = torch.randn(2, model.layer_dims[0])

    if method == "wrapper":
        model.prune_neurons(index=0, neuron_indices=list(range(prune_count)))
    else:
        prune_neurons(model, index=0, neuron_indices=list(range(prune_count)))

    assert model.layer_dims[1] == keep_dim
    assert model.layers[0].out_features == keep_dim
    assert model.layers[1].in_features == keep_dim

    y = model(x)
    assert y.shape == (2, model.layer_dims[-1])
