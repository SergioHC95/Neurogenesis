"""
Serial MNIST MLP Experiment Sweep
---------------------------------
- Flexible sweep config (dict-based, promote/demote sweep fields on the fly)
- Each experiment prints full config and results
- One run at a time (useful for debugging, notebooks, or logging)
"""

import torch
import torch.nn as nn
import torch.optim as optim

from experiments.config import base_sweep_cfg
from experiments.experimenter import Experiment
from experiments.utils import (
    get_loader,
    iter_experiment_configs,
    make_expname,
    prepare_data,
    print_exp_header,
    update_sweepconfig,
)
from src.models.base_mlp import BaseMLP
from src.training.custom_trainers import KitchenSinkTrainer


def run_experiment(cfg_update=None, prefix: str = ""):
    """
    Serial experiment runner.
    - Optionally update sweep config (promote/demote sweep vars)
    - Iterates over all runs, running one at a time
    - Prints header and results for each experiment
    """
    if cfg_update is None:
        sweep_cfg = base_sweep_cfg.copy()
    else:
        sweep_cfg = update_sweepconfig(base_cfg=base_sweep_cfg, **cfg_update)

    for run_cfg, swept_keys in iter_experiment_configs(sweep_cfg):
        expname = make_expname(run_cfg, swept_keys, prefix=prefix)
        print_exp_header(expname, run_cfg)

        torch.manual_seed(run_cfg.seed)
        g = torch.Generator()
        g.manual_seed(run_cfg.seed)

        trainset, valset, testset = prepare_data(generator=g)
        train_loader = get_loader(trainset, run_cfg.batch_size, run_cfg.run_type)
        val_loader = get_loader(valset, run_cfg.batch_size, run_cfg.run_type)
        test_loader = get_loader(testset, run_cfg.batch_size, run_cfg.run_type)

        def build_model():
            return BaseMLP(list(run_cfg.layer_dims)).to(run_cfg.device)

        def build_optimizer(model):
            return optim.AdamW(
                model.parameters(), lr=run_cfg.lr, weight_decay=run_cfg.wd
            )

        loss_fn = nn.CrossEntropyLoss()

        exp = Experiment(
            experiment_name=expname,
            TrainerClass=KitchenSinkTrainer,
            model_builder=build_model,
            optimizer_builder=build_optimizer,
            loss_f=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            config=run_cfg,
        )
        exp.run(epochs=run_cfg.epochs)
        test_acc = exp.trainer.evaluate(test_loader)
        print(f"Final Test Accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    # Example usage: run the default sweep
    print(f"Using device: {base_sweep_cfg.get('device', 'cuda')}")
    print(base_sweep_cfg)
    run_experiment(prefix="mnist")
