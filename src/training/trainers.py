import csv
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import clear_output, display
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import ProfilerActivity

from tqdm import tqdm


class BaseTrainer:
    """
    Minimal PyTorch trainer for core training, validation, and checkpointing.

    Args:
        model (nn.Module): Model to train.
        optim (Optimizer): Optimizer for model parameters.
        loss_f (callable): Loss function.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader, optional): Validation data loader.
        path (str): Path for checkpoint file.
        resume_if_exists (bool): Resume from checkpoint if available.
        gradient_clip (float): Maximum gradient norm (for clipping).
        device (str or torch.device, optional): Device for training ('cpu' or 'cuda').

    Attributes:
        model (nn.Module): The neural network being trained.
        optim (Optimizer): Optimizer for parameters.
        loss_f (callable): Loss function.
        train_loader (DataLoader): Loader for training batches.
        val_loader (DataLoader): Loader for validation batches.
        gradient_clip (float): Gradient clipping threshold.
        device (str): Training device.
        path (str): Path for checkpoint.
        epoch (int): Current epoch.
        train_losses (list): Training losses by epoch.
        val_losses (list): Validation losses by epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        optim: Optimizer,
        loss_f,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        path: str = "./results/checkpoints/checkpoint.pt",
        resume_if_exists: bool = True,
        gradient_clip: float = 1.0,
        device: str | torch.device = None,
    ):
        if device is not None and device == "cuda" and not torch.cuda.is_available():
            print(
                "Device 'cuda' was requested but is not available. Falling back to CPU."
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optim = optim
        self.loss_f = loss_f
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_clip = gradient_clip
        self.path = path
        if resume_if_exists and os.path.exists(self.path):
            self.load_checkpoint()
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.epoch = 0
            self.train_losses = []
            self.val_losses = []

    def preprocess_batch(self, batch):
        """
        Move batch tensors to the configured device.
        Args:
            batch (tuple): (inputs, targets)
        Returns:
            tuple: (inputs, targets) on correct device
        """
        inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)

    def train_epoch(self):
        """
        Run one training epoch.
        Returns:
            float: Average training loss for this epoch.
        """
        self.model.train()
        use_tqdm = sys.stdout.isatty()
        loader = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch+1} [Train]",
            unit="batch",
            leave=False,
            disable=not use_tqdm,
        )
        total_loss = 0.0
        for batch in loader:
            inputs, targets = self.preprocess_batch(batch)
            self.optim.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_f(outputs, targets)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
            self.optim.step()
            total_loss += loss.item()
            loader.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self):
        """
        Run one validation epoch.
        Returns:
            float: Average validation loss for this epoch.
        """
        self.model.eval()
        use_tqdm = sys.stdout.isatty()
        loader = tqdm(
            self.val_loader,
            desc=f"Epoch {self.epoch+1} [Val]",
            unit="batch",
            leave=False,
            disable=not use_tqdm,
        )
        total_loss = 0.0
        for batch in loader:
            inputs, targets = self.preprocess_batch(batch)
            outputs = self.model(inputs)
            loss = self.loss_f(outputs, targets)
            total_loss += loss.item()
            loader.set_postfix(loss=loss.item())
        return total_loss / len(self.val_loader)

    def train(self, epochs=100, print_every=10, validate_every=10):
        """
        Train the model for a given number of epochs.

        Args:
            epochs (int): Number of training epochs.
            print_every (int): Interval for printing progress.
            validate_every (int): Interval for validation.

        Returns:
            tuple: (list of training losses, list of validation losses)
        """
        end_epoch = self.epoch + epochs
        for epoch in range(self.epoch, end_epoch):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            val_loss = None
            if self.val_loader and (
                epoch % validate_every == 0 or epoch == 0 or epoch == end_epoch - 1
            ):
                val_loss = self.val_epoch()
                self.val_losses.append(val_loss)
            self.epoch += 1
            self.save_checkpoint()
            if epoch == 0 or (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch+1}/{end_epoch} | Train loss: {train_loss:.4f}"
                    + (f", Val loss: {val_loss:.4f}" if val_loss else "")
                )
        print("Training completed")
        return self.train_losses, self.val_losses

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        """
        Evaluate the model on a given dataset.
        Args:
            loader (DataLoader): DataLoader to evaluate.
        Returns:
            float: Accuracy on the dataset.
        """
        self.model.eval()
        correct = total = 0
        for batch in loader:
            inputs, targets = self.preprocess_batch(batch)
            y_preds = self.model(inputs).argmax(dim=1)
            correct += (y_preds == targets).sum().item()
            total += targets.size(0)
        acc = correct / total
        return acc

    def save_checkpoint(self):
        """
        Save model, optimizer, and training state to a checkpoint file.
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "epoch": self.epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "device": str(self.device),
        }
        torch.save(checkpoint, self.path)

    def load_checkpoint(self):
        """
        Load model, optimizer, and training state from a checkpoint file.
        """
        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint["model"])
        self.optim.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        print(
            f"Resumed on {self.device} from epoch {self.epoch}: Train loss {self.train_losses[-1]:.4f}"
            + (f", Val loss {self.val_losses[-1]:.4f}" if self.val_losses else "")
        )


class DiagnosticsTrainer(BaseTrainer):
    """
    Trainer subclass that adds in-memory diagnostics tracking:
    - Gradient norm per layer (nn.Linear)
    - Weight norm per layer (nn.Linear)
    - Activation stats: mean, std, dead neuron fraction per layer

    Attributes:
        diagnostic_stats (dict): Per-layer stats history.
        _diagnostic_handles (list): Handles to registered hooks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diagnostic_handles = []
        self.diagnostic_stats = {}  # {layer_name: {"weight_norm": [...], ...}}
        self._register_diagnostics()

    def _register_diagnostics(self):
        """
        Register hooks on all nn.Linear layers to collect stats during training/validation.
        """
        for name, module in self.model.named_modules():
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
                    # Dead fraction: fraction of zero activations (for ReLU output)
                    self.diagnostic_stats[layer_name]["dead_fraction"].append(
                        (act == 0).float().mean().item()
                    )

                handle2 = module.register_forward_hook(activation_hook)
                self._diagnostic_handles.append(handle2)

    def _collect_weight_norms(self):
        """
        Collect and record current weight norms for all tracked layers.
        """
        for name, module in self.model.named_modules():
            if name in self.diagnostic_stats and isinstance(module, nn.Linear):
                self.diagnostic_stats[name]["weight_norm"].append(
                    module.weight.data.norm().item()
                )

    def train_epoch(self):
        self._collect_weight_norms()
        result = super().train_epoch()
        return result

    def val_epoch(self):
        self._collect_weight_norms()
        result = super().val_epoch()
        return result

    def close(self):
        """
        Remove all diagnostic hooks (call at end of training or before model deletion).
        """
        for h in self._diagnostic_handles:
            h.remove()
        self._diagnostic_handles.clear()


class LoggingTrainer(BaseTrainer):
    """
    Trainer subclass for logging training and validation losses/accuracy.
    - Logs to CSV and/or TensorBoard as configured.
    - Does NOT add diagnostics or plotting.

    Args:
        log_dir (str, optional): Directory for log files (CSV, TensorBoard).
        log_csv (bool): If True, log to CSV.
        log_tb (bool): If True, log to TensorBoard.
    """

    def __init__(self, *args, log_dir=None, log_csv=True, log_tb=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        self.log_csv = log_csv
        self.log_tb = log_tb

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.csv_path = (
                os.path.join(self.log_dir, "metrics.csv") if self.log_csv else None
            )
            self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_tb else None
        else:
            self.csv_path = None
            self.writer = None

    def _log_epoch(self, epoch, train_loss, val_loss=None):
        """
        Log the latest epoch's results to CSV and/or TensorBoard.
        """
        if self.writer:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            if val_loss is not None:
                self.writer.add_scalar("Loss/Val", val_loss, epoch)

        if self.csv_path:
            file_exists = os.path.isfile(self.csv_path)
            row = {"epoch": epoch, "train_loss": train_loss}
            if val_loss is not None:
                row["val_loss"] = val_loss
            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

    def train(self, epochs=100, print_every=10, validate_every=10):
        """
        Train the model, logging metrics after each epoch.
        """
        end_epoch = self.epoch + epochs
        for epoch in range(self.epoch, end_epoch):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            val_loss = None
            if self.val_loader and (
                epoch % validate_every == 0 or epoch == 0 or epoch == end_epoch - 1
            ):
                val_loss = self.val_epoch()
                self.val_losses.append(val_loss)
            self.epoch += 1
            self.save_checkpoint()

            self._log_epoch(self.epoch, train_loss, val_loss)

            if epoch == 0 or (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch+1}/{end_epoch} | Train loss: {train_loss:.4f}"
                    + (f", Val loss: {val_loss:.4f}" if val_loss else "")
                )
        print("Training completed")
        return self.train_losses, self.val_losses

    def close(self):
        if self.writer:
            self.writer.close()


class NotebookTrainer(BaseTrainer):
    """
    Trainer for notebooks: shows live learning curves and interactive feedback.
    - Plots curves live during training
    - Prints summary table at end
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_learning_curve(self, save=False, path=None, show=True):
        """
        Live plot of train/val loss during training (clears output).
        """
        clear_output(wait=True)
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Train Loss", color="blue")
        if self.val_losses:
            plt.plot(self.val_losses, label="Val Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save and path:
            plt.savefig(path)
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()

    def _display_table(self, last_n=5):
        """
        Display the last n epochs of losses in a pandas DataFrame.
        """
        data = {
            "Epoch": list(range(self.epoch - last_n + 1, self.epoch + 1)),
            "Train Loss": self.train_losses[-last_n:],
        }
        if self.val_losses:
            data["Val Loss"] = self.val_losses[-last_n:]
        df = pd.DataFrame(data)
        display(df)

    def train(self, epochs=100, print_every=1, validate_every=1, plot=True, table=True):
        """
        Train with live interactive feedback.
        """
        end_epoch = self.epoch + epochs
        for epoch in range(self.epoch, end_epoch):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            val_loss = None
            if self.val_loader and (
                epoch % validate_every == 0 or epoch == 0 or epoch == end_epoch - 1
            ):
                val_loss = self.val_epoch()
                self.val_losses.append(val_loss)
            self.epoch += 1
            self.save_checkpoint()
            # Live feedback
            if plot:
                self.plot_learning_curve(show=True)
            if table and (epoch == 0 or (epoch + 1) % print_every == 0):
                print(f"Epoch {epoch+1}/{end_epoch}")
                self._display_table()
        print("Training completed")
        if plot:
            self.plot_learning_curve(show=True)
        if table:
            self._display_table(last_n=min(10, self.epoch))



class ProfilingTrainer(BaseTrainer):
    """
    Trainer subclass for profiling training cost:
      - Tracks parameter updates
      - Profiles FLOPs for first batch of each epoch
      - Saves summary stats to CSV

    Args:
        profile_batches (int): Number of batches per epoch to profile for FLOPs (default: 1)
        profile_dir (str, optional): Directory to save profiling stats (default: None disables saving)
    Attributes:
        param_updates (int): Total optimizer steps taken.
        flop_stats (list): List of profiled FLOPs (per epoch).
    """
    def __init__(self, *args, profile_batches=1, profile_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_updates = 0
        self.flop_stats = []
        self.profile_batches = profile_batches
        self.profile_dir = profile_dir
        if self.profile_dir:
            os.makedirs(self.profile_dir, exist_ok=True)
            self.profile_csv = os.path.join(self.profile_dir, "profile_stats.csv")
        else:
            self.profile_csv = None

    def train_epoch(self):
        self.model.train()
        use_tqdm = sys.stdout.isatty()
        loader = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]", unit="batch", leave=False, disable=not use_tqdm)
        total_loss = 0.0
        epoch_flops = 0

        for i, batch in enumerate(loader):
            inputs, targets = self.preprocess_batch(batch)
            self.optim.zero_grad()
            # Profile only the first N batches for FLOPs (typically N=1 for speed)
            if i < self.profile_batches:
                with profile(
                    activities=[ProfilerActivity.CPU], 
                    record_shapes=True, 
                    profile_memory=True, 
                    with_flops=True
                ) as prof:
                    outputs = self.model(inputs)
                    loss = self.loss_f(outputs, targets)
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                    self.optim.step()
                # Estimate total FLOPs for this batch
                batch_flops = sum(
                    (evt.flops if hasattr(evt, 'flops') and evt.flops is not None else 0)
                    for evt in prof.key_averages()
                )
                epoch_flops += batch_flops
            else:
                outputs = self.model(inputs)
                loss = self.loss_f(outputs, targets)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                self.optim.step()
            total_loss += loss.item()
            self.param_updates += 1
            loader.set_postfix(loss=loss.item())
        # Store epoch-level FLOPs estimate (from first profiled batch)
        self.flop_stats.append(epoch_flops)
        return total_loss / len(self.train_loader)

    def save_profile_stats(self):
        """
        Save per-epoch FLOPs and total parameter updates to CSV.
        """
        if not self.profile_csv:
            return
        with open(self.profile_csv, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "flops", "param_updates"])
            writer.writeheader()
            for i, flops in enumerate(self.flop_stats):
                writer.writerow({"epoch": i+1, "flops": flops, "param_updates": self.param_updates})

    def train(self, epochs=100, print_every=10, validate_every=10):
        end_epoch = self.epoch + epochs
        for epoch in range(self.epoch, end_epoch):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            val_loss = None
            if self.val_loader and (epoch % validate_every == 0 or epoch == 0 or epoch == end_epoch - 1):
                val_loss = self.val_epoch()
                self.val_losses.append(val_loss)
            self.epoch += 1
            self.save_checkpoint()
            if epoch == 0 or (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch+1}/{end_epoch} | Train loss: {train_loss:.4f}"
                    + (f", Val loss: {val_loss:.4f}" if val_loss else "")
                )
        print("Training completed")
        self.save_profile_stats()
        return self.train_losses, self.val_losses

    def get_flop_stats(self):
        """
        Returns per-epoch profiled FLOPs.
        """
        return self.flop_stats

    def get_param_update_count(self):
        """
        Returns total number of parameter updates (optimizer steps).
        """
        return self.param_updates