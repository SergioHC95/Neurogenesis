import pytest
import torch

from src.utils.init_utils import InitType, _autoinit_weight_block, init_weight_block


def test_init_weight_block_identity() -> None:
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


def test_init_weight_block_orthogonal() -> None:
    """Test orthogonal initialization."""
    in_dim, out_dim = 5, 5
    W = init_weight_block(in_dim, out_dim, InitType.ORTHOGONAL)

    # Check shape
    assert W.shape == (out_dim, in_dim)

    # Check orthogonality
    WWT = W @ W.T
    assert torch.allclose(WWT, torch.eye(out_dim), atol=1e-6)


@pytest.mark.parametrize(
    "init_type", [InitType.IDENTITY, InitType.ORTHOGONAL, InitType.AUTOINIT]
)
def test_init_weight_block_various_sizes(init_type: InitType) -> None:
    """Test initialization with different matrix sizes."""
    test_sizes = [(3, 5), (5, 3), (4, 4)]

    for in_dim, out_dim in test_sizes:
        W = init_weight_block(in_dim, out_dim, init_type)
        assert W.shape == (out_dim, in_dim)
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()


def test_init_weight_block_invalid_type():
    with pytest.raises(ValueError, match="Unknown initialization type"):
        init_weight_block(10, 10, init_type="invalid")  # Invalid enum


# Test CRAZY init
def test_init_weight_block_crazy():
    W = init_weight_block(10, 10, InitType.CRAZY)
    assert W.shape == (10, 10)
    assert W.abs().mean() > 10  # sanity check


# Test all _autoinit_weight_block modes
@pytest.mark.parametrize("mode", ["fan_in", "fan_out", "fan_avg"])
def test_autoinit_weight_block_modes(mode):
    W = _autoinit_weight_block(10, 20, nonlinearity="relu", mode=mode)
    assert W.shape == (20, 10)
    assert torch.isfinite(W).all()


# Trigger ValueError for invalid autoinit mode
def test_autoinit_invalid_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        _autoinit_weight_block(10, 10, mode="nonsense")
