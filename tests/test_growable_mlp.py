import torch

from src.utils.init_utils import InitType
from tests.utils import assert_model_consistency


def test_grow_layer_hidden_output_shape(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    n_new = 5
    in_dim, mid_dim, out_dim = model.layer_dims

    model.grow_layer(index=0, n_new=n_new)

    assert model.layer_dims == [in_dim, mid_dim + n_new, out_dim]
    assert_model_consistency(model)

    output = model(x)
    assert output.shape[1] == out_dim


def test_grow_layer_output_shape_change(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    n_new = 3
    in_dim, mid_dim, out_dim = model.layer_dims

    model.grow_layer(index=1, n_new=n_new)

    assert model.layer_dims == [in_dim, mid_dim, out_dim + n_new]
    assert_model_consistency(model)

    output = model(x)
    assert output.shape == (x.shape[0], out_dim + n_new)
    assert not torch.isnan(output).any()


def test_insert_layer_method_api(growable_mlp):
    model = growable_mlp
    in_dim = model.layer_dims[0]
    insert_size = 16

    model.insert_layer(index=0, new_size=insert_size, init_type=InitType.AUTOINIT)

    assert model.layer_dims[0] == in_dim
    assert model.layer_dims[1] == insert_size
    assert_model_consistency(model)

    output = model(torch.randn(2, in_dim))
    assert output.shape[1] == model.layer_dims[-1]


def test_prune_neurons_method_api(growable_mlp):
    model = growable_mlp
    n_prune = 3
    original_dim = model.layer_dims[1]
    prune_indices = list(range(n_prune))

    model.prune_neurons(index=0, neuron_indices=prune_indices)

    assert model.layer_dims[1] == original_dim - n_prune
    assert_model_consistency(model)

    out = model(torch.randn(1, model.layer_dims[0]))
    assert out.shape[1] == model.layer_dims[-1]


def test_grow_layer_custom_init(growable_mlp):
    model = growable_mlp
    initial_out = model.layer_dims[1]
    n_new = 4

    model.grow_layer(index=0, n_new=n_new, init_type=InitType.AUTOINIT)

    assert model.layer_dims[1] == initial_out + n_new
    assert_model_consistency(model)


def test_update_architecture_resets_layers_correctly(growable_mlp):
    model = growable_mlp
    new_dims = [4, 6, 3]
    x = torch.randn(5, new_dims[0])

    model._update_architecture(new_dims)

    assert model.layer_dims == new_dims
    assert_model_consistency(model)

    out = model(x)
    assert out.shape == (5, 3)


def test_insert_layer_wrapper_calls_growth_utils(growable_mlp):
    model = growable_mlp
    original_dims = model.layer_dims[:]
    model.insert_layer(index=0, new_size=15)

    assert model.layer_dims[0] == original_dims[0]
    assert model.layer_dims[1] == 15
    assert model.layer_dims[2] == original_dims[1]
    assert_model_consistency(model)


def test_grow_hidden_layer_preserves_output_shape(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    original_out = model(x)

    in_dim, hidden_dim, out_dim = model.layer_dims
    n_new = 5
    model.grow_layer(index=0, n_new=n_new)

    assert model.layer_dims == [in_dim, hidden_dim + n_new, out_dim]
    new_out = model(x)

    assert new_out.shape == original_out.shape
    assert not torch.isnan(new_out).any()


def test_grow_output_layer_changes_output_shape(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    original_out_dim = model.layer_dims[-1]
    n_new = 4

    model.grow_layer(index=1, n_new=n_new)

    assert model.layer_dims[-1] == original_out_dim + n_new
    new_out = model(x)

    assert new_out.shape == (x.shape[0], original_out_dim + n_new)
    assert not torch.isnan(new_out).any()


def test_insert_layer_maintains_output_shape(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    original_output = model(x)

    in_dim = model.layer_dims[0]
    model.insert_layer(index=0, new_size=14, init_type=InitType.AUTOINIT)

    assert model.layer_dims[0] == in_dim
    assert model(x).shape == original_output.shape


def test_prune_neurons_preserves_output_shape(growable_mlp, random_input):
    model = growable_mlp
    x = random_input
    output_before = model(x)

    n_prune = 3
    model.prune_neurons(index=0, neuron_indices=list(range(n_prune)))

    assert model(x).shape == output_before.shape


def test_custom_init_does_not_crash(growable_mlp):
    model = growable_mlp
    in_dim = model.layer_dims[0]
    old_hidden = model.layer_dims[1]

    model.grow_layer(index=0, n_new=6, init_type=InitType.AUTOINIT)

    assert model.layer_dims[1] == old_hidden + 6
    assert model.layers[0].weight.shape == (old_hidden + 6, in_dim)
