from dataclasses import dataclass

import torch


@dataclass
class ExperimentConfig:
    seed: int
    run_type: str
    batch_size: int
    lr: float
    wd: float
    epochs: int
    layer_dims: tuple
    device: str


base_sweep_cfg = {
    "seed": 0,
    "run_type": "minibatch",
    "lr": 1e-3,
    "wd": 1e-4,
    "batch_size": 128,
    "epochs": 20,
    "layer_dims": (784, 256, 128, 10),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
