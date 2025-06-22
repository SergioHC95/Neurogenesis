"""
Parallelized MNIST MLP Experiment Sweep
---------------------------------------

- Uses dataclass configs for sweep and per-run variables
- Builds experiment configs from a sweep config (any variable can be swept via `list_` prefix)
- Runs all experiments in parallel using ProcessPoolExecutor (CPU)
- Each run is fully independent and reproducible
"""

from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
from src.training.custom_trainers import HeadlessTrainer

# ======= Per-Experiment Logic for Parallel Execution =======


def run_single_experiment(args):
    """
    Runs a single experiment, prints header, and returns (expname, final test acc).
    Intended to be called in a separate process.
    """
    run_cfg, swept_keys, prefix = args
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
        return optim.AdamW(model.parameters(), lr=run_cfg.lr, weight_decay=run_cfg.wd)

    loss_fn = nn.CrossEntropyLoss()

    exp = Experiment(
        experiment_name=expname,
        TrainerClass=HeadlessTrainer,
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
    return expname, test_acc


# ======= Parallel Experiment Runner =======


def run_experiment(cfg_update=None, prefix: str = "", max_workers=4):
    """
    Main parallel experiment runner.
    - Builds sweep config (optionally updated by user)
    - Runs all (config, expname) pairs in parallel processes
    - Prints and returns results
    """
    if cfg_update is None:
        sweep_cfg = base_sweep_cfg.copy()
    else:
        sweep_cfg = update_sweepconfig(base_cfg=base_sweep_cfg, **cfg_update)

    experiment_args = [
        (run_cfg, swept_keys, prefix)
        for run_cfg, swept_keys in iter_experiment_configs(sweep_cfg)
    ]

    results = []
    print(f"Launching {len(experiment_args)} experiments with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_experiment, args) for args in experiment_args
        ]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Sweep progress"
        ):
            try:
                expname, test_acc = fut.result()
                print(f"[DONE] {expname}: Test acc = {test_acc:.2%}")
            except Exception as e:
                print(f"[ERROR]: {e}")

    print("\nSweep finished. Summary:")
    for expname, acc in results:
        print(f"{expname}: {acc:.2%}")
    return results


# ======= Entry Point =======
if __name__ == "__main__":
    print(f"Using device: {base_sweep_cfg.device}")
    print(base_sweep_cfg)
    # Example: run default sweep, or provide your own via cfg_update
    run_experiment(prefix="mnist", max_workers=4)
