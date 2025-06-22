import os
import warnings

import matplotlib
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.growable_mlp import GrowableMLP
from src.training.trainer import BaseTrainer, NotebookTrainer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use("Agg")

# =========================
# Fixtures
# =========================


@pytest.fixture
def dummy_data():
    X = torch.randn(100, 3)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)


@pytest.fixture
def dummy_model():
    return nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2))


@pytest.fixture
def dummy_optimizer(dummy_model):
    return optim.Adam(dummy_model.parameters(), lr=1e-3)


@pytest.fixture
def dummy_loss():
    return nn.CrossEntropyLoss()


@pytest.fixture
def sample_model():
    return GrowableMLP([10, 20, 5])


# =========================
# Tests
# =========================


def test_base_trainer(tmp_path, dummy_model, dummy_optimizer, dummy_loss, dummy_data):
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer = BaseTrainer(
        model=dummy_model,
        optim=dummy_optimizer,
        loss_f=dummy_loss,
        train_loader=dummy_data,
        val_loader=dummy_data,
        device="cpu",
        path=str(checkpoint_path),
    )

    assert isinstance(trainer, BaseTrainer)

    # Train
    initial_weights = dummy_model[0].weight.clone()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="FigureCanvasAgg is non-interactive"
        )
        trainer.train(epochs=1, print_every=1)
    assert not torch.allclose(initial_weights, dummy_model[0].weight)

    # Evaluate
    acc = trainer.evaluate(dummy_data)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

    # Checkpoint
    trainer.save_checkpoint()
    assert checkpoint_path.exists()

    resumed_trainer = BaseTrainer(
        model=dummy_model,
        optim=dummy_optimizer,
        loss_f=dummy_loss,
        train_loader=dummy_data,
        val_loader=dummy_data,
        device="cpu",
        path=str(checkpoint_path),
        resume_if_exists=True,
    )

    assert resumed_trainer.epoch == trainer.epoch
    assert resumed_trainer.train_losses[-1] == trainer.train_losses[-1]
    assert resumed_trainer.val_loss_log[-1] == trainer.val_loss_log[-1]


def test_plot_learning_curve(
    tmp_path, dummy_model, dummy_optimizer, dummy_loss, dummy_data
):
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer = NotebookTrainer(
        model=dummy_model,
        optim=dummy_optimizer,
        loss_f=dummy_loss,
        train_loader=dummy_data,
        val_loader=dummy_data,
        device="cpu",
        path=str(checkpoint_path),
    )
    trainer.train_losses = [1.0, 0.9]
    trainer.val_loss_log = [(0, 1.1), (1, 1.0)]

    plot_path = tmp_path / "plot.png"
    trainer.plot_learning_curve(save=False, show=False)
    trainer.plot_learning_curve(save=True, show=False, path=str(plot_path))
    assert plot_path.exists()
