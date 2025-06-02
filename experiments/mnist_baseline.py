from dataclasses import dataclass
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch

from src.models.base_mlp import BaseMLP
from src.training.custom_trainers import NotebookFullFeaturedTrainer

# ========================
# Experiment Configuration
# ========================

@dataclass
class Config:
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 20
    layer_dims: tuple[int, ...] = (784, 256, 128, 10)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seeds: tuple = (0, 1, 2)
    run_types: tuple = ("minibatch", "batch")  # run both modes

def prepare_data(cfg: Config):
    # Compute normalization stats
    raw_transform = transforms.ToTensor()
    raw_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=raw_transform)
    raw_loader = DataLoader(raw_dataset, batch_size=len(raw_dataset), shuffle=False)
    images, _ = next(iter(raw_loader))
    mean = images.mean().item()
    std = images.std().item()

    # Define normalization + flatten transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
        transforms.Normalize((mean,), (std,)),
    ])

    # Datasets/loaders
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    return trainset, valset, testset

def get_loader(dataset, batch_size, batch_type="minibatch"):
    if batch_type == "batch":
        return DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def run_experiment(cfg: Config):
    trainset, valset, testset = prepare_data(cfg)

    for batch_type in cfg.run_types:
        for seed in cfg.seeds:
            print(f"\n--- Seed {seed}, {batch_type} mode ---")
            torch.manual_seed(seed)

            train_loader = get_loader(trainset, cfg.batch_size, batch_type)
            val_loader = get_loader(valset, cfg.batch_size, batch_type)
            test_loader = get_loader(testset, cfg.batch_size, batch_type)

            model = BaseMLP(list(cfg.layer_dims)).to(cfg.device)
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
            loss_fn = nn.CrossEntropyLoss()

            # Setup output/log paths
            expname = f"mnist_{batch_type}_seed{seed}"
            log_dir = f"./results/logs/{expname}"
            profile_dir = f"./results/profiling/{expname}"
            checkpoint_path = f"./results/checkpoints/{expname}.pt"
            plot_path = f"./results/plots/{expname}_curve.png"

            trainer = NotebookFullFeaturedTrainer(
                model,
                optimizer,
                loss_fn,
                train_loader,
                val_loader,
                path=checkpoint_path,
                log_dir=log_dir,
                profile_dir=profile_dir,
                resume_if_exists=False,
            )

            trainer.train(epochs=cfg.epochs)
            trainer.plot_learning_curve(save=True, path=plot_path)
            test_acc = trainer.evaluate(test_loader)
            print(f"Final Test Accuracy: {test_acc:.2%}")
            trainer.close()  # remove hooks etc.

if __name__ == "__main__":
    cfg = Config()
    print(f"Using device: {cfg.device}")
    print(cfg)
    run_experiment(cfg)
