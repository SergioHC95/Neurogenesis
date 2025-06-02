import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.models.growable_mlp import GrowableMLP


@pytest.fixture(autouse=True)
def silence_loguru_logs():
    logger.remove()
    logger.add(lambda *_: None, level="CRITICAL")


@pytest.fixture
def growable_mlp():
    """Fixture for a basic GrowableMLP with 7 → 13 → 3 layer dims."""
    return GrowableMLP([7, 13, 3])


@pytest.fixture
def random_input():
    """Fixture for a standard input tensor."""
    return torch.randn(3, 7)
