import torch.nn as nn

from .init_utils import InitType, init_weight_block


def insert_layer(
    model: nn.Module, index: int, new_size: int, init_type: InitType = InitType.IDENTITY
) -> None:
    """Insert a new layer with specified size at the given index.

    Args:
        model: The neural network model
        index: Position to insert the new layer
        new_size: Number of neurons in the new layer
        init_type: Initialization strategy for the new weights

    Raises:
        IndexError: If index is out of bounds
        ValueError: If new_size <= 0
    """
    if not (0 <= index < len(model.layers)):
        raise IndexError(f"Layer index {index} out of bounds")
    if new_size <= 0:
        raise ValueError(f"New size must be positive, got {new_size}")

    prev_out = model.layer_dims[index]
    next_in = model.layer_dims[index + 1]

    layer1 = nn.Linear(prev_out, new_size)
    layer2 = nn.Linear(new_size, next_in)

    # Initialize new layers
    layer1.weight.data = init_weight_block(prev_out, new_size, init_type)
    layer2.weight.data = init_weight_block(new_size, next_in, init_type)
    layer1.bias.data.zero_()
    layer2.bias.data.zero_()

    model.layers = (
        model.layers[:index]
        + nn.ModuleList([layer1, layer2])
        + model.layers[index + 1 :]
    )
    model.layer_dims = (
        model.layer_dims[: index + 1] + [new_size] + model.layer_dims[index + 1 :]
    )


def grow_layer(
    model: nn.Module, index: int, n_new: int, init_type: InitType = InitType.IDENTITY
) -> None:
    """Grow the output dimension of the transformation from layer_dims[index] to layer_dims[index + 1].

    Args:
        model: The neural network model with model.layers and model.layer_dims
        index: Index of layer to grow (0 = input layer → first hidden layer)
        n_new: Number of neurons to add
        init_type: Initialization strategy for the new weights

    Raises:
        IndexError: If index is out of bounds
        ValueError: If n_new <= 0
    """
    if not (0 <= index < len(model.layers)):
        raise IndexError(f"Layer index {index} out of bounds")
    if n_new <= 0:
        raise ValueError(f"Number of new neurons must be positive, got {n_new}")

    old_layer = model.layers[index]
    in_dim = old_layer.in_features
    old_out = old_layer.out_features
    new_out = old_out + n_new

    # Create and initialize the grown layer
    new_layer = nn.Linear(in_dim, new_out)
    new_layer.weight.data[:old_out, :] = old_layer.weight.data
    new_layer.bias.data[:old_out] = old_layer.bias.data
    new_layer.weight.data[old_out:, :] = init_weight_block(in_dim, n_new, init_type)
    new_layer.bias.data[old_out:] = 0.0

    model.layers[index] = new_layer
    model.layer_dims[index + 1] = new_out  # update dimension bookkeepingw

    # Update the input dimension of the next layer, if it exists
    if index + 1 < len(model.layers):
        old_next = model.layers[index + 1]
        next_out = old_next.out_features

        # Create new next layer with updated input dimension
        new_next = nn.Linear(new_out, next_out)
        new_next.weight.data[:, :old_out] = old_next.weight.data

        # Initialize the new columns instead of zeroing them
        new_init_block = init_weight_block(new_out, next_out, init_type)
        new_next.weight.data[:, old_out:] = new_init_block[:, old_out:]

        new_next.bias.data = old_next.bias.data  # preserve bias

        model.layers[index + 1] = new_next


def prune_neurons(model: nn.Module, index: int, neuron_indices: list[int]) -> None:
    """Remove specified neurons from the output of layer `index`, reducing layer_dims[index + 1].

    Args:
        model: The neural network model with model.layers and model.layer_dims
        index: Index of the layer whose output neurons are to be pruned
               (0 = input → first hidden layer)
        neuron_indices: Indices of output neurons to remove

    Raises:
        IndexError: If index is out of bounds
        ValueError: If neuron indices are invalid
    """
    if not (0 <= index < len(model.layers)):
        raise IndexError(f"Layer index {index} out of bounds")

    old_layer = model.layers[index]
    if any(i >= old_layer.out_features or i < 0 for i in neuron_indices):
        raise ValueError("Invalid neuron indices")

    keep_indices = [i for i in range(old_layer.out_features) if i not in neuron_indices]
    if not keep_indices:
        raise ValueError("Cannot remove all neurons from a layer")

    in_dim = old_layer.in_features
    out_dim = len(keep_indices)

    # Create pruned layer
    new_layer = nn.Linear(in_dim, out_dim)
    new_layer.weight.data = old_layer.weight.data[keep_indices, :]
    new_layer.bias.data = old_layer.bias.data[keep_indices]

    model.layers[index] = new_layer
    model.layer_dims[index + 1] = out_dim

    # Update next layer if it exists
    if index + 1 < len(model.layers):
        next_layer = model.layers[index + 1]
        new_next = nn.Linear(out_dim, next_layer.out_features)
        new_next.weight.data = next_layer.weight.data[:, keep_indices]
        new_next.bias.data = next_layer.bias.data

        model.layers[index + 1] = new_next
        model.layer_dims[index + 1] = out_dim  # Now input to next layer
