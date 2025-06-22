"""
Experiment/sweep/data utilities for MNIST project.
Includes:
    - Config update/promotion/demotion for sweeps
    - Cartesian product over sweep params
    - Experiment naming
    - Pretty experiment headers
    - Data preparation/loaders
"""

import itertools

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from experiments.config import ExperimentConfig


def update_sweepconfig(base_cfg: dict, **kwargs):
    """
    Promotes/demotes sweep or scalar fields in the config.
    Returns a new sweep config dictionary.
    """
    cfg = dict(base_cfg) if not isinstance(base_cfg, dict) else base_cfg.copy()
    for k, v in kwargs.items():
        if k.startswith("list_"):
            field_name = k[5:]
            cfg[k] = list(v) if isinstance(v, (list, tuple)) else [v]
            cfg.pop(field_name, None)
        else:
            cfg[k] = v
            cfg.pop("list_" + k, None)
    return cfg


def normalize_sweep_vals(d: dict):
    """
    Ensures all sweep values are tuples/lists.
    """
    norm = {}
    for k, v in d.items():
        if k.startswith("list_"):
            if isinstance(v, (list, tuple)) and not isinstance(v, str):
                norm[k] = tuple(v)
            else:
                norm[k] = (v,)
    return norm


def iter_experiment_configs(sweep_cfg: dict):
    """
    Generates configs for all sweep combinations.
    """
    sweep_vals = normalize_sweep_vals(sweep_cfg)
    sweep_keys = list(sweep_vals.keys())
    static_vars = {k: v for k, v in sweep_cfg.items() if not k.startswith("list_")}
    exp_keys = [k[5:] for k in sweep_keys]
    for vals in itertools.product(*(sweep_vals[k] for k in sweep_keys)):
        exp_dict = dict(zip(exp_keys, vals))
        exp_dict.update(static_vars)
        yield ExperimentConfig(**exp_dict), exp_keys


def make_expname(run_cfg: ExperimentConfig, swept_keys: list, prefix: str = ""):
    """
    Build experiment name from swept keys and current run config.
    """
    parts = [prefix] if prefix else []
    for k in swept_keys:
        field = k[5:] if k.startswith("list_") else k
        parts.append(f"{field}-{getattr(run_cfg, field)}")
    return "_".join(parts)


def print_exp_header(expname: str, run_cfg: ExperimentConfig):
    """
    Print a pretty experiment header with config.
    """
    bar = "=" * (len(expname) + 18)
    print(f"\n{bar}\n   Experiment: {expname}\n{bar}")
    for k, v in vars(run_cfg).items():
        print(f"  {k:<12}: {v}")
    print("-" * len(bar))


def prepare_data(generator: torch.Generator = None):
    """
    Prepare normalized and split MNIST datasets.
    """
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
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
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
    """
    Returns a DataLoader for the dataset with the right batch size/type.
    """
    if batch_type == "batch":
        return DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=0)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
