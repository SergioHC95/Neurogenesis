from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.models.base_mlp import BaseMLP
from src.training.custom_trainers import NotebookFullFeaturedTrainer

# ========================
# Experiment Configuration
# ========================


@dataclass
class Config:
    batch_size: int = 128
    lr: float = 1e-3
    wd: float = 1e-4
    epochs: int = 20
    layer_dims: tuple[int, ...] = (784, 256, 128, 10)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seeds: tuple = (0, 1, 2)
    run_types: tuple = ("minibatch", "batch")  # run both modes


def prepare_data(cfg: Config, generator=None):
    # Compute normalization stats
    raw_transform = transforms.ToTensor()
    raw_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=raw_transform
    )
    raw_loader = DataLoader(
        raw_dataset, batch_size=len(raw_dataset), shuffle=False, num_workers=0
    )
    images, _ = next(iter(raw_loader))
    mean = images.mean().item()
    std = images.std().item()

    # Define normalization + flatten transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    # Datasets/loaders
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    trainset, valset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    return trainset, valset, testset


def get_loader(dataset, batch_size, batch_type="minibatch"):
    if batch_type == "batch":
        return DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=0)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def run_experiment(cfg: Config = Config(), exp_prefix: str = ""):
    # Ensure prefix ends with _ if given and not empty
    prefix = (exp_prefix + "_") if exp_prefix else ""
    for batch_type in cfg.run_types:
        for seed in cfg.seeds:
            print(f"\n--- Seed {seed}, {batch_type} mode ---")
            print(f"Config: {cfg}")

            torch.manual_seed(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            trainset, valset, testset = prepare_data(cfg, generator=g)
            train_loader = get_loader(trainset, cfg.batch_size, batch_type)
            val_loader = get_loader(valset, cfg.batch_size, batch_type)
            test_loader = get_loader(testset, cfg.batch_size, batch_type)

            model = BaseMLP(list(cfg.layer_dims)).to(cfg.device)
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
            loss_fn = nn.CrossEntropyLoss()

            expname = f"{prefix}mnist_{batch_type}_seed{seed}"
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
                log_csv=True,
                log_tb=True,
                profile_dir=profile_dir,
                profile_batches=1,  # profile only first batch each epoch for FLOPs
                resume_if_exists=False,
            )

            trainer.train(epochs=cfg.epochs, print_every=1, validate_every=1, plot=True)
            trainer.plot_learning_curve(save=True, path=plot_path)
            test_acc = trainer.evaluate(test_loader)
            print(f"Final Test Accuracy: {test_acc:.2%}")

            print(
                f"Logs saved to: {log_dir}/metrics.csv (CSV), {log_dir} (TensorBoard)"
            )
            print(f"Profiling stats saved to: {profile_dir}/profile_stats.csv")
            print(f"Checkpoint saved to: {checkpoint_path}")
            print(f"Plot saved to: {plot_path}")

            trainer.close()


if __name__ == "__main__":
    cfg = Config()
    print(f"Using device: {cfg.device}")
    print(cfg)
    run_experiment(cfg)
