import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.models.base_mlp import TorchMLP
from src.training.trainer import Trainer
from dataclasses import dataclass

# ========================
# Experiment Configuration
# ========================

@dataclass
class Config:
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    layer_dims: list = (784, 256, 128, 10)
    device: str = "cpu"

cfg = Config()
print(f"Using device: {cfg.device}")
print(cfg)

# Compute normalization statistics
raw_transform = transforms.ToTensor()
raw_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=raw_transform)
raw_loader = DataLoader(raw_dataset, batch_size=len(raw_dataset), shuffle=False)
images, _ = next(iter(raw_loader))
mean = images.mean().item()
std = images.std().item()

# Define normalization transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),  # flatten
    transforms.Normalize((mean,), (std,))
])

# Reload datasets with normalization
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
trainset, valset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=cfg.batch_size)
test_loader = DataLoader(testset, batch_size=cfg.batch_size)

# Initialize model and trainer
model = TorchMLP(cfg.layer_dims).to(cfg.device)
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
loss_fn = nn.CrossEntropyLoss()

trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader,
                  path="./results/checkpoints/mnist_mlp.pt", resume_if_exists=False)

# Train and test the model
trainer.train(epochs=cfg.epochs)
trainer.plot_learning_curve(save=True, path=f"./results/plots/mnist_mlp_e{cfg.epochs}.png")
test_acc = trainer.evaluate(test_loader)
print(f"Final Test Accuracy: {test_acc:.2%}")

