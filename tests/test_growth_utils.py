import pytest
import torch

from src.utils.growth_utils import InitType, grow_layer, insert_layer, prune_neurons
from tests.utils import assert_output_preserved


def test_insert_layer_output_shape(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    input_dim = model.layer_dims[0]
    original_hidden = model.layer_dims[1]
    insert_size = 15

    insert_layer(model, index=0, new_size=insert_size, init_type=InitType.ORTHOGONAL)

    assert model.layer_dims == [
        input_dim,
        insert_size,
        original_hidden,
        model.layer_dims[-1],
    ]
    assert len(model.layers) == 3
    assert model.layers[0].weight.shape == (insert_size, input_dim)
    assert model.layers[1].weight.shape[1] == insert_size

    new_output = model(x)
    assert new_output.shape == (x.shape[0], model.layer_dims[-1])


def test_insert_layer_invalid_inputs(growable_mlp):
    model = growable_mlp
    with pytest.raises(IndexError, match="Layer index 5 out of bounds"):
        insert_layer(model, index=5, new_size=8)

    with pytest.raises(ValueError, match="New size must be positive"):
        insert_layer(model, index=0, new_size=-1)


def test_grow_layer_output_shape(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    original_weights = model.layers[0].weight.clone()
    original_out = model.layer_dims[1]
    n_new = 5

    grow_layer(model, index=0, n_new=n_new)

    assert model.layer_dims[1] == original_out + n_new
    assert model.layers[0].weight.shape == (original_out + n_new, model.layer_dims[0])
    assert model.layers[1].in_features == original_out + n_new

    assert torch.allclose(model.layers[0].weight[:original_out], original_weights)
    output = model(x)
    assert output.shape[1] == model.layer_dims[-1]


def test_grow_layer_invalid_inputs(growable_mlp):
    model = growable_mlp

    with pytest.raises(ValueError, match="Number of new neurons must be positive"):
        grow_layer(model, index=0, n_new=0)

    with pytest.raises(IndexError, match="Layer index 5 out of bounds"):
        grow_layer(model, index=5, n_new=3)


def test_prune_neurons_output_shape(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    original_out = model.layer_dims[1]
    prune_indices = list(range(3))

    prune_neurons(model, index=0, neuron_indices=prune_indices)

    assert model.layer_dims[1] == original_out - len(prune_indices)
    assert model.layers[0].out_features == original_out - len(prune_indices)
    assert model.layers[1].in_features == original_out - len(prune_indices)

    new_output = model(x)
    assert new_output.shape == (x.shape[0], model.layer_dims[-1])


def test_prune_neurons_invalid_inputs(growable_mlp):
    model = growable_mlp

    with pytest.raises(ValueError, match="Invalid neuron indices"):
        prune_neurons(model, index=0, neuron_indices=[-1, 100])

    with pytest.raises(ValueError, match="Cannot remove all neurons"):
        prune_neurons(model, index=0, neuron_indices=list(range(model.layer_dims[1])))

    with pytest.raises(IndexError, match="Layer index 5 out of bounds"):
        prune_neurons(model, index=5, neuron_indices=[0])


def test_insert_layer_output_consistency(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    original_out_dim = model.layer_dims[-1]

    insert_layer(model, index=0, new_size=model.layer_dims[1] + 5)

    assert model(x).shape == (x.shape[0], original_out_dim)


def test_grow_layer_output_consistency(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    original_out_dim = model.layer_dims[-1]

    grow_layer(model, index=0, n_new=3)
    assert model(x).shape == (x.shape[0], original_out_dim)

    grow_layer(model, index=1, n_new=2)
    assert model(x).shape == (x.shape[0], model.layer_dims[-1])


def test_prune_neurons_output_consistency(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    original_out_dim = model.layer_dims[-1]

    keep_count = model.layer_dims[1] - 4
    prune_indices = list(range(4))
    prune_neurons(model, index=0, neuron_indices=prune_indices)

    assert model.layer_dims[1] == keep_count
    assert model(x).shape == (x.shape[0], original_out_dim)


def test_identity_growth_preserves_output(growable_mlp, random_input):
    model = growable_mlp
    x = random_input

    assert_output_preserved(
        x,
        operation=grow_layer,
        kwargs={
            "model": model,
            "index": 0,
            "n_new": 10,
            "init_type": InitType.IDENTITY,
        },
    )


def test_orthogonal_growth_preserves_output(growable_mlp, random_input):
    model = growable_mlp
    x = random_input

    assert_output_preserved(
        x,
        operation=grow_layer,
        kwargs={
            "model": model,
            "index": 0,
            "n_new": 10,
            "init_type": InitType.ORTHOGONAL,
        },
    )


def test_crazy_growth_changes_output(growable_mlp, random_input):
    model = growable_mlp
    x = random_input

    with torch.no_grad():
        before = model(x).clone()

    grow_layer(model, index=0, n_new=10, init_type=InitType.CRAZY)

    with torch.no_grad():
        after = model(x).clone()

    diff = (before - after).abs()
    assert diff.max() > 1e-3, "Output did not change despite crazy init!"
