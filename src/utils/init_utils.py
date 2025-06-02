import math
from enum import Enum
from typing import Literal

import torch


class InitType(Enum):
    """
    Enumeration of initialization types for weight matrices.

    Values:
        IDENTITY: Initialize weights as an identity matrix (with zero padding if needed)
        ORTHOGONAL: Initialize weights as an orthogonal matrix (with zero padding if needed)
        AUTOINIT: Initialize weights using AutoInit-style initialization
        CRAZY: Initialize weights with large random values
    """

    IDENTITY = "identity"
    ORTHOGONAL = "orthogonal"
    AUTOINIT = "autoinit"
    CRAZY = "crazy"


def _init_identity_block(in_dim: int, out_dim: int) -> torch.Tensor:
    """
    Initialize an identity weight matrix of size (out_dim, in_dim).
    Creates a matrix filled with zeros except for an identity matrix in the top-left corner.
    If dimensions don't match, the identity portion uses the smaller dimension.
    """

    W = torch.zeros(out_dim, in_dim)
    min_dim = min(in_dim, out_dim)
    W[:min_dim, :min_dim] = torch.eye(min_dim)

    return W


def _init_orthogonal_block(in_dim: int, out_dim: int) -> torch.Tensor:
    """
    Initialize an orthogonal weight matrix of size (out_dim, in_dim).
    If dimensions don't match, pads with zeros similar to identity_block.
    """
    W = torch.zeros(out_dim, in_dim)
    min_dim = min(in_dim, out_dim)
    q = torch.nn.init.orthogonal_(torch.empty(min_dim, min_dim))
    W[:min_dim, :min_dim] = q
    return W


def _autoinit_weight_block(
    in_dim: int,
    out_dim: int,
    nonlinearity: Literal["relu", "tanh", "sigmoid", "gelu", "linear"] = "relu",
    mode: Literal["fan_in", "fan_out", "fan_avg"] = "fan_in",
) -> torch.Tensor:
    """
    AutoInit-style initialization that preserves signal variance through layers.

    Args:
        in_dim: input dimensionality
        out_dim: output dimensionality
        nonlinearity: activation function used after this layer
        mode: how to compute the fan scaling ('fan_in', 'fan_out', or 'fan_avg')

    Returns:
        Initialized weight tensor of shape (out_dim, in_dim)
    """
    if mode == "fan_in":
        fan = in_dim
    elif mode == "fan_out":
        fan = out_dim
    elif mode == "fan_avg":
        fan = (in_dim + out_dim) / 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Gain is based on activation function
    gain = torch.nn.init.calculate_gain(nonlinearity)

    std = gain / math.sqrt(fan)
    return torch.empty(out_dim, in_dim).normal_(mean=0.0, std=std)


def _init_crazy_block(in_dim: int, out_dim: int) -> torch.Tensor:
    return torch.randn(out_dim, in_dim) * 100  # BIG values


def init_weight_block(
    in_dim: int, out_dim: int, init_type: InitType = InitType.AUTOINIT
) -> torch.Tensor:
    """
    Initialize a weight matrix of size (out_dim, in_dim) using the specified initialization type.
    """
    if init_type == InitType.IDENTITY:
        return _init_identity_block(in_dim, out_dim)
    elif init_type == InitType.ORTHOGONAL:
        return _init_orthogonal_block(in_dim, out_dim)
    elif init_type == InitType.AUTOINIT:
        return _autoinit_weight_block(
            in_dim, out_dim, nonlinearity="relu", mode="fan_in"
        )
    elif init_type == InitType.CRAZY:
        return _init_crazy_block(in_dim, out_dim)
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")
