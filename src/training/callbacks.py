# src/training/callbacks.py

import csv
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython.display import clear_output
from torch.profiler import ProfilerActivity, profile
from torch.utils.tensorboard import SummaryWriter


class Callback:
    """
    Base class for all Trainer callbacks.

    Subclasses can implement any of the hooks below to run code at key points
    during training/validation.
    """

    def on_train_start(self, trainer):
        """
        Called before the start of training.
        Args:
            trainer (Trainer): The main Trainer instance.
        """
        pass

    def on_epoch_start(self, trainer, epoch):
        """
        Called at the start of every epoch.
        Args:
            trainer (Trainer): The main Trainer instance.
            epoch (int): The epoch number (starting from 0).
        """
        pass

    def on_train_batch_end(self, trainer, batch, logs):
        """
        Called at the end of every training batch.
        Args:
            trainer (Trainer): The main Trainer instance.
            batch: The batch as provided by the DataLoader.
            logs (dict): Dictionary with current loss, outputs, inputs, targets, etc.
        """
        pass

    def on_epoch_end(self, trainer, epoch, logs):
        """
        Called at the end of every epoch (after training & validation).
        Args:
            trainer (Trainer): The main Trainer instance.
            epoch (int): The epoch number.
            logs (dict): Dict with 'train_loss', 'val_loss', 'val_acc'.
        """
        pass

    def on_train_end(self, trainer):
        """
        Called at the very end of training (after all epochs).
        Args:
            trainer (Trainer): The main Trainer instance.
        """
        pass


class LoggingCallback(Callback):
    """
    Logs training and validation metrics to CSV and/or TensorBoard.

    Args:
        log_dir (str): Directory for log files.
        log_csv (bool): If True, logs to CSV (metrics.csv).
        log_tb (bool): If True, logs to TensorBoard.
    """

    def __init__(self, log_dir=None, log_csv=True, log_tb=True):
        super().__init__()  # Inherit from Callback!
        self.log_dir = log_dir
        self.log_csv = log_csv
        self.log_tb = log_tb

        self.csv_path = None
        self.writer = None

        self._fields = ["epoch", "train_loss", "val_loss", "val_acc"]
        self._csv_file = None

    def on_train_start(self, trainer):
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            if self.log_csv:
                self.csv_path = os.path.join(self.log_dir, "metrics.csv")
                self._csv_file = open(self.csv_path, mode="a", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=self._fields
                )
                if os.stat(self.csv_path).st_size == 0:
                    self._csv_writer.writeheader()
            if self.log_tb and SummaryWriter is not None:
                self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_epoch_end(self, trainer, epoch, logs):
        if self.writer:
            if logs["train_loss"] is not None:
                self.writer.add_scalar("Loss/Train", logs["train_loss"], epoch)
            if logs["val_loss"] is not None:
                self.writer.add_scalar("Loss/Val", logs["val_loss"], epoch)
            if logs["val_acc"] is not None:
                self.writer.add_scalar("Accuracy/Val", logs["val_acc"], epoch)
        if self._csv_file:
            row = {
                "epoch": epoch,
                "train_loss": logs["train_loss"],
                "val_loss": logs["val_loss"],
                "val_acc": logs["val_acc"],
            }
            self._csv_writer.writerow(row)
            self._csv_file.flush()

    def on_train_end(self, trainer):
        if self.writer:
            self.writer.close()
            self.writer = None
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None


class ProfilingCallback(Callback):
    """
    Profiles training cost per epoch:
      - Tracks total parameter updates (optimizer steps)
      - Profiles FLOPs for the first N batches of each epoch
      - Optionally logs per-epoch stats to CSV

    Args:
        profile_batches (int): Number of batches per epoch to profile for FLOPs (default: 1)
        profile_dir (str): Directory to save profiling stats as CSV (default: None disables saving)
    """

    def __init__(self, profile_batches=1, profile_dir=None):
        super().__init__()
        self.profile_batches = profile_batches
        self.profile_dir = profile_dir

        self.param_updates = 0
        self.flop_stats = []  # List of FLOPs for each epoch

        if self.profile_dir:
            os.makedirs(self.profile_dir, exist_ok=True)
            self.profile_csv = os.path.join(self.profile_dir, "profile_stats.csv")
        else:
            self.profile_csv = None

        # Internal state for first-N-batches-per-epoch profiling
        self._profile_this_epoch = False
        self._profiled_batches = 0
        self._this_epoch_flops = 0

        # For remembering the current epoch
        self._current_epoch = None

    def on_epoch_start(self, trainer, epoch):
        self._profile_this_epoch = True
        self._profiled_batches = 0
        self._this_epoch_flops = 0
        self._current_epoch = epoch

    def on_train_batch_end(self, trainer, batch, logs):
        self.param_updates += 1

        # Only profile for first N batches of each epoch
        if not profile or not ProfilerActivity:
            return
        if self._profile_this_epoch and self._profiled_batches < self.profile_batches:
            inputs = logs["inputs"]
            targets = logs["targets"]

            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
            ) as prof:
                outputs = trainer.model(inputs)
                loss = trainer.loss_f(outputs, targets)
                loss.backward()
                # Note: we do not step optimizer here, to not double-step

            # Estimate total FLOPs for this batch
            batch_flops = sum(
                (evt.flops if hasattr(evt, "flops") and evt.flops is not None else 0)
                for evt in prof.key_averages()
            )
            self._this_epoch_flops += batch_flops
            self._profiled_batches += 1

        # Stop profiling for the rest of the epoch after N batches
        if self._profiled_batches >= self.profile_batches:
            self._profile_this_epoch = False

    def on_epoch_end(self, trainer, epoch, logs):
        # Store epoch-level FLOPs estimate (from first profiled batch(es))
        self.flop_stats.append(self._this_epoch_flops)

    def on_train_end(self, trainer):
        # Save profiling stats to CSV if requested
        if not self.profile_csv:
            return
        with open(self.profile_csv, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "flops", "param_updates"])
            writer.writeheader()
            for i, flops in enumerate(self.flop_stats):
                writer.writerow(
                    {
                        "epoch": i + 1,
                        "flops": flops,
                        "param_updates": self.param_updates,
                    }
                )


class DiagnosticsCallback(Callback):
    """
    Tracks and saves per-layer diagnostics for all nn.Linear layers during training.
    Collected statistics include:
        - Weight norm per epoch
        - Gradient norm per backward pass
        - Activation mean, std, and dead neuron fraction per forward pass

    All diagnostics are saved to a JSON file at training end.

    Args:
        diagnostics_dir (str): Directory to save diagnostics file (default: './diagnostics').

    Attributes:
        diagnostic_stats (dict): Stores all collected stats, indexed by layer name.
        _diagnostic_handles (list): Stores PyTorch hook handles for later removal.
    """

    def __init__(self, diagnostics_dir="./diagnostics"):
        """
        Initializes the DiagnosticsCallback.

        Args:
            diagnostics_dir (str): Where to save diagnostics.json after training.
        """
        super().__init__()
        self.diagnostics_dir = diagnostics_dir
        os.makedirs(self.diagnostics_dir, exist_ok=True)
        self._diagnostic_handles = []
        self.diagnostic_stats = {}  # {layer_name: {"weight_norm": [...], ...}}

    def on_train_start(self, trainer):
        """
        Registers forward and backward hooks on all nn.Linear layers of the model
        to track activations and gradients during training.

        Args:
            trainer (Trainer): The Trainer instance.
        """
        model = trainer.model
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.diagnostic_stats[name] = {
                    "weight_norm": [],
                    "grad_norm": [],
                    "activation_mean": [],
                    "activation_std": [],
                    "dead_fraction": [],
                }

                # Backward hook for gradient norm
                def grad_hook(mod, grad_input, grad_output, layer_name=name):
                    grad = mod.weight.grad
                    if grad is not None:
                        self.diagnostic_stats[layer_name]["grad_norm"].append(
                            grad.norm().item()
                        )

                handle1 = module.register_full_backward_hook(grad_hook)
                self._diagnostic_handles.append(handle1)

                # Forward hook for activation stats
                def activation_hook(mod, inp, out, layer_name=name):
                    act = out.detach()
                    self.diagnostic_stats[layer_name]["activation_mean"].append(
                        act.mean().item()
                    )
                    self.diagnostic_stats[layer_name]["activation_std"].append(
                        act.std().item()
                    )
                    self.diagnostic_stats[layer_name]["dead_fraction"].append(
                        (act == 0).float().mean().item()
                    )

                handle2 = module.register_forward_hook(activation_hook)
                self._diagnostic_handles.append(handle2)

    def on_epoch_start(self, trainer, epoch):
        """
        At the start of each epoch, records the current weight norm
        for every tracked nn.Linear layer.

        Args:
            trainer (Trainer): The Trainer instance.
            epoch (int): The epoch number.
        """
        model = trainer.model
        for name, module in model.named_modules():
            if name in self.diagnostic_stats and isinstance(module, nn.Linear):
                self.diagnostic_stats[name]["weight_norm"].append(
                    module.weight.data.norm().item()
                )

    def on_train_end(self, trainer):
        """
        Removes all hooks from the model to avoid memory leaks,
        and saves all collected diagnostics as a JSON file.
        """
        # Remove hooks
        for h in self._diagnostic_handles:
            h.remove()
        self._diagnostic_handles.clear()

        # Save diagnostics as JSON
        out_path = os.path.join(self.diagnostics_dir, "diagnostics_stats.json")
        with open(out_path, "w") as f:
            json.dump(self.diagnostic_stats, f, indent=2)
        print(f"[DiagnosticsCallback] Saved diagnostics to {out_path}")


class NotebookPlotCallback(Callback):
    """
    Live notebook plotting callback: draws/updates learning curves after each epoch.
    Shows training loss, validation loss, and validation accuracy.
    Optionally saves a final plot to disk at training end.

    Args:
        plot_path (str or None): Path to save the final plot image (optional).
        show (bool): If True, show the plot inline (default: True).
    """

    def __init__(self, plot_path=None, show=True):
        """
        Initializes the NotebookPlotCallback.

        Args:
            plot_path (str or None): Where to save the plot at the end of training.
            show (bool): Whether to display the plot in the notebook after each epoch.
        """
        super().__init__()
        self.plot_path = plot_path
        self.show = show

    def on_epoch_end(self, trainer, epoch, logs):
        """
        Draws (and optionally saves) the current learning curves at the end of each epoch.

        Args:
            trainer (Trainer): The Trainer instance.
            epoch (int): The current epoch.
            logs (dict): Dictionary of latest metrics.
        """
        self.plot_learning_curve(trainer, save=False, show=self.show)

    def on_train_end(self, trainer):
        """
        Optionally saves the final learning curve plot after training.

        Args:
            trainer (Trainer): The Trainer instance.
        """
        if self.plot_path is not None:
            self.plot_learning_curve(
                trainer, save=True, path=self.plot_path, show=False
            )

    def plot_learning_curve(self, trainer, save=False, path=None, show=True):
        """
        Plots the training and validation loss, plus validation accuracy, as curves.

        Args:
            trainer (Trainer): The Trainer instance.
            save (bool): Whether to save the plot to disk.
            path (str): Path to save the plot (if save is True).
            show (bool): Whether to display the plot inline (in Jupyter/Colab).
        """
        clear_output(wait=True)
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Train loss curve
        if trainer.train_losses:
            ax1.plot(
                range(1, len(trainer.train_losses) + 1),
                trainer.train_losses,
                label="Train Loss",
                color="tab:blue",
                linewidth=2,
            )
        # Val loss curve
        if trainer.val_loss_log:
            val_epochs, val_losses = zip(*trainer.val_loss_log)
            ax1.plot(
                [e + 1 for e in val_epochs],  # 1-based for user-friendliness
                val_losses,
                label="Val Loss",
                color="tab:orange",
                linestyle="--",
                linewidth=2,
            )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Validation accuracy (second y-axis)
        if trainer.val_acc_log:
            ax2 = ax1.twinx()
            acc_epochs, accs = zip(*trainer.val_acc_log)
            ax2.plot(
                [e + 1 for e in acc_epochs],
                [100 * acc for acc in accs],
                label="Val Accuracy",
                color="tab:green",
                linestyle=":",
                linewidth=2,
            )
            ax2.set_ylabel("Validation Accuracy (%)")
            ax2.legend(loc="upper right", fontsize=10)

        plt.title("Learning Curve", fontsize=14)
        plt.tight_layout()

        if save and path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            plt.close()
            print(f"[NotebookPlotCallback] Plot saved to {path}")
        elif show:
            plt.show()
        else:
            plt.close()


class CheckpointCallback(Callback):
    """
    Checkpoints model, optimizer, and training state after each epoch (or at specified intervals).

    Args:
        checkpoint_path (str): File path for saving the checkpoint.
        save_every (int): How often to save (in epochs). Default: 1 (every epoch).
        save_final (bool): Save final checkpoint at end of training (default: True).
    """

    def __init__(self, checkpoint_path, save_every=1, save_final=True):
        """
        Args:
            checkpoint_path (str): Where to save checkpoint (e.g., './results/checkpoints/run1.pt').
            save_every (int): How often to save (default: every epoch).
            save_final (bool): Whether to save at training end.
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.save_every = save_every
        self.save_final = save_final

    def on_epoch_end(self, trainer, epoch, logs):
        """
        Save checkpoint at end of epoch (according to save_every).

        Args:
            trainer (Trainer): The Trainer instance.
            epoch (int): The epoch number.
            logs (dict): Training/validation metrics (not used here).
        """
        self._save_checkpoint(trainer, self.checkpoint_path)

    def on_train_end(self, trainer):
        """
        Optionally save a final checkpoint after training.

        Args:
            trainer (Trainer): The Trainer instance.
        """
        self._save_checkpoint(trainer, self.checkpoint_path)

    @staticmethod
    def _save_checkpoint(trainer, path):
        """
        Internal helper to actually save the checkpoint.

        Args:
            trainer (Trainer): The Trainer instance.
            path (str): The path to save to.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optim.state_dict(),
            "epoch": trainer.epoch,
            "train_losses": trainer.train_losses,
            "val_loss_log": trainer.val_loss_log,
            "val_acc_log": trainer.val_acc_log,
            "device": str(trainer.device),
        }
        torch.save(checkpoint, path)
