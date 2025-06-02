import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from models.growable_mlp import GrowableMLP
from utils.growth_utils import insert_layer, grow_layer, prune_neurons


def test_insert_layer():
    """Test inserting a new layer into the model."""
    model = GrowableMLP([10, 20, 5])
    insert_layer(model, index=0, new_size=8)
    assert model.layer_sizes == [10, 8, 20, 5]
    assert len(model.layers) == 3
    assert model.layers[0].weight.shape == (8, 10)
    assert model.layers[1].weight.shape == (20, 8)


def test_grow_layer():
    """Test growing an existing layer by adding new neurons."""
    model = GrowableMLP([10, 20, 5])
    grow_layer(model, index=0, n_new=5)
    assert model.layer_sizes == [10, 25, 5]
    assert model.layers[0].weight.shape == (25, 10)
    assert model.layers[1].weight.shape == (5, 25)


def test_prune_neurons():
    """Test pruning specific neurons from a layer."""
    model = GrowableMLP([10, 20, 5])
    prune_neurons(model, index=0, neuron_indices=[0, 1, 2])
    assert model.layer_sizes == [10, 17, 5]
    assert model.layers[0].weight.shape == (17, 10)
    assert model.layers[1].weight.shape == (5, 17)


if __name__ == "__main__":
    test_insert_layer()
    test_grow_layer()
    test_prune_neurons()
    print("All tests passed.")
