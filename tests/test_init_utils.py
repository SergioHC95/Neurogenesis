import pytest
import torch
from src.utils.init_utils import init_weight_block, InitType


def test_init_weight_block_identity():
    """Test identity initialization."""
    in_dim, out_dim = 5, 7
    W = init_weight_block(in_dim, out_dim, InitType.IDENTITY)

    # Check shape
    assert W.shape == (out_dim, in_dim)

    # Check identity part
    min_dim = min(in_dim, out_dim)
    assert torch.allclose(W[:min_dim, :min_dim], torch.eye(min_dim))

    # Check padding is zeros
    assert torch.allclose(W[min_dim:, :], torch.zeros(out_dim - min_dim, in_dim))
    assert torch.allclose(W[:, min_dim:], torch.zeros(out_dim, in_dim - min_dim))


def test_init_weight_block_orthogonal():
    """Test orthogonal initialization."""
    in_dim, out_dim = 5, 5
    W = init_weight_block(in_dim, out_dim, InitType.ORTHOGONAL)

    # Check shape
    assert W.shape == (out_dim, in_dim)

    # Check orthogonality
    WWT = W @ W.T
    assert torch.allclose(WWT, torch.eye(out_dim), atol=1e-6)


@pytest.mark.parametrize("init_type", [InitType.IDENTITY, InitType.ORTHOGONAL])
def test_init_weight_block_various_sizes(init_type):
    """Test initialization with different matrix sizes."""
    test_sizes = [(3, 5), (5, 3), (4, 4)]

    for in_dim, out_dim in test_sizes:
        W = init_weight_block(in_dim, out_dim, init_type)
        assert W.shape == (out_dim, in_dim)
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()
