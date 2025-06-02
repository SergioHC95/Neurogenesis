import torch
from enum import Enum


class InitType(Enum):
    """
    Enumeration of initialization types for weight matrices.

    Values:
        IDENTITY: Initialize weights as an identity matrix (with zero padding if needed)
        ORTHOGONAL: Initialize weights as an orthogonal matrix (with zero padding if needed)
    """

    IDENTITY = "identity"
    ORTHOGONAL = "orthogonal"


def _init_identity_block(in_dim, out_dim):
    """
    Initialize an identity weight matrix of size (out_dim, in_dim).
    Creates a matrix filled with zeros except for an identity matrix in the top-left corner.
    If dimensions don't match, the identity portion uses the smaller dimension.
    """
    W = torch.zeros(out_dim, in_dim)
    min_dim = min(in_dim, out_dim)
    W[:min_dim, :min_dim] = torch.eye(min_dim)
    return W


def _init_orthogonal_block(in_dim, out_dim):
    """
    Initialize an orthogonal weight matrix of size (out_dim, in_dim).
    If dimensions don't match, pads with zeros similar to identity_block.
    """
    W = torch.zeros(out_dim, in_dim)
    min_dim = min(in_dim, out_dim)
    q = torch.nn.init.orthogonal_(torch.empty(min_dim, min_dim))
    W[:min_dim, :min_dim] = q
    return W


def init_weight_block(in_dim, out_dim, init_type=InitType.IDENTITY):
    """
    Initialize a weight matrix of size (out_dim, in_dim) using the specified initialization type.
    """
    if init_type == InitType.IDENTITY:
        return _init_identity_block(in_dim, out_dim)
    elif init_type == InitType.ORTHOGONAL:
        return _init_orthogonal_block(in_dim, out_dim)
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")
