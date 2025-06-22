import os

import torch

try:
    from tqdm.notebook import tqdm as tqdm_notebook
except ImportError:
    tqdm_notebook = None
    from tqdm import tqdm


class Trainer:
    """
    Unified, extensible training loop for supervised PyTorch experiments.
    All extensibility (logging, profiling, diagnostics, plotting, etc.) is handled via callbacks.

    Attributes:
        model: The neural network to be trained.
        optimizer: The optimizer for parameter updates.
        loss_f: Loss function (e.g., torch.nn.CrossEntropyLoss).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data (optional).
        device: Device to use for training ('cpu' or 'cuda').
        callbacks: List of callback objects.
        train_losses: Training losses for each epoch.
        val_loss_log: Validation losses for each epoch (epoch, loss).
        val_acc_log: Validation accuracies for each epoch (epoch, acc).
        epoch: Current epoch (updated per epoch).
    """

    def __init__(
        self,
        model,
        optim,
        loss_f,
        train_loader,
        val_loader=None,
        device=None,
        callbacks=None,
        resume_from=None,
        **kwargs,
    ):
        """
        Args:
            model: PyTorch model to train.
            optim: Optimizer (e.g., Adam, SGD).
            loss_f: Loss function (callable).
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            device: 'cuda', 'cpu', or torch.device (default: autodetect).
            callbacks: List of Callback instances (default: []).
            resume_from: Path to checkpoint to resume training from (default: None).
            **kwargs: Any additional attributes to set (e.g., config params).
        """
        self.model = model
        self.optim = optim
        self.loss_f = loss_f
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.callbacks = callbacks or []

        # State for logging/stats
        self.train_losses = []
        self.val_loss_log = []
        self.val_acc_log = []
        self.epoch = 0

        # Arbitrary kwargs can be set as attributes (e.g., batch_size, log_dir, etc)
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Resume training from checkpoint if provided
        if resume_from is not None and os.path.exists(resume_from):
            self._load_checkpoint(resume_from)

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optim.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint.get("epoch", 0)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_loss_log = checkpoint.get("val_loss_log", [])
        self.val_acc_log = checkpoint.get("val_acc_log", [])
        print(
            f"[Trainer] Resumed from checkpoint: {path} "
            f"(epoch {self.epoch}, train_loss {self.train_losses[-1] if self.train_losses else 'NA'})"
        )

    def train(self, epochs=1, print_every=1, validate_every=1, show_tqdm=True):
        """
        Main training loop.

        Args:
            epochs: Number of epochs to train.
            print_every: Interval for printing progress (passed to callbacks if desired).
            validate_every: Interval for validation (passed to callbacks if desired).
            show_tqdm: Whether to show tqdm progress bars per epoch (default: True).
        """
        # --- Training start ---
        for cb in self.callbacks:
            cb.on_train_start(self)

        end_epoch = self.epoch + epochs
        for epoch in range(self.epoch, end_epoch):
            self.epoch = epoch
            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            train_loss = self.train_epoch(show_tqdm=show_tqdm)
            self.train_losses.append(train_loss)

            val_loss = None
            val_acc = None
            if self.val_loader and (
                epoch % validate_every == 0 or epoch == 0 or epoch == end_epoch - 1
            ):
                val_loss = self.val_epoch(show_tqdm=show_tqdm)
                self.val_loss_log.append((epoch, val_loss))
                val_acc = self.evaluate(self.val_loader)
                self.val_acc_log.append((epoch, val_acc))

            # Per-epoch callback
            logs = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)

            # Print summary line
            if (epoch + 1) % print_every == 0 or epoch == end_epoch - 1:
                msg = f"Epoch {epoch+1}/{end_epoch} | Train loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f", Val loss: {val_loss:.4f}"
                if val_acc is not None:
                    msg += f", Val acc: {val_acc:.2%}"
                print(msg, flush=True)

        # --- Training end ---
        for cb in self.callbacks:
            cb.on_train_end(self)

    def train_epoch(self, show_tqdm=True):
        """
        Run one training epoch, with tqdm progress bar.
        Returns average training loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0

        # Choose tqdm/notebook tqdm or fallback
        if show_tqdm:
            desc = f"Train Epoch {self.epoch+1}"
            tqdm_args = {"desc": desc, "leave": False, "colour": "darkred"}
            if tqdm_notebook is not None:
                data_iter = tqdm_notebook(self.train_loader, **tqdm_args)
            else:
                data_iter = tqdm(self.train_loader, **tqdm_args)
        else:
            data_iter = self.train_loader

        for batch in data_iter:
            # Accept (inputs, targets) or (inputs, targets, extra)
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, targets = batch[:2]
            else:
                raise ValueError("Batch must be a tuple (inputs, targets)")
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optim.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_f(outputs, targets)
            loss.backward()
            self.optim.step()
            total_loss += loss.item()

            # Per-batch callback
            logs = {
                "loss": loss.item(),
                "outputs": outputs,
                "inputs": inputs,
                "targets": targets,
            }
            for cb in self.callbacks:
                cb.on_train_batch_end(self, batch, logs)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self, show_tqdm=True):
        """
        Run one validation epoch, with tqdm progress bar.
        Returns average validation loss for this epoch.
        """
        self.model.eval()
        total_loss = 0.0

        # Choose tqdm/notebook tqdm or fallback
        if show_tqdm:
            desc = f"Val Epoch {self.epoch+1}"
            tqdm_args = {"desc": desc, "leave": False, "colour": "darkred"}
            if tqdm_notebook is not None:
                data_iter = tqdm_notebook(self.val_loader, **tqdm_args)
            else:
                data_iter = tqdm(self.val_loader, **tqdm_args)
        else:
            data_iter = self.val_loader

        for batch in data_iter:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, targets = batch[:2]
            else:
                raise ValueError("Batch must be a tuple (inputs, targets)")
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_f(outputs, targets)
            total_loss += loss.item()
        return total_loss / len(self.val_loader)

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Evaluate the model on any DataLoader (e.g., test set).
        Returns:
            accuracy (float)
        """
        self.model.eval()
        correct, total = 0, 0
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, targets = batch[:2]
            else:
                raise ValueError("Batch must be a tuple (inputs, targets)")
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            preds = self.model(inputs).argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        return correct / total if total > 0 else 0.0

    def get_train_log(self):
        """
        Returns:
            Tuple of (train_losses, val_loss_log, val_acc_log) collected during training.
        """
        return self.train_losses, self.val_loss_log, self.val_acc_log
