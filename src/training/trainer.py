import os
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math


class Trainer:
    def __init__(self,model,optim,loss_f,train_loader,
        val_loader=None,
        path="./results/checkpoints/checkpoint.pt",
        resume_if_exists=True,
        gradient_clip=1.0,
        device=None,
    ):
        self.model = model
        self.optim = optim
        self.loss_f = loss_f
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_clip = gradient_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.path = path
        if resume_if_exists and os.path.exists(self.path):
            self.load_checkpoint()
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.epoch = 0
            self.train_losses = []
            self.val_losses = []

    def preprocess_batch(self, batch):
        inputs, targets = batch
        # Override if needed (e.g. flatten inputs, apply transforms)
        return inputs.to(self.device), targets.to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        loader = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch+1} [Train]",
            unit="batch",
            leave=False,
        )

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
        self.model.eval()
        total_loss = 0
        loader = tqdm(
            self.val_loader,
            desc=f"Epoch {self.epoch+1} [Val]",
            unit="batch",
            leave=False,
        )

        for batch in loader:
            inputs, targets = self.preprocess_batch(batch)
            outputs = self.model(inputs)
            loss = self.loss_f(outputs, targets)
            total_loss += loss.item()
            loader.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader)

    def train(self, epochs=100, print_every=10, validate_every=10, plot=True):
        end_epoch = self.epoch + epochs

        for epoch in range(self.epoch, end_epoch):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            val_loss = None
            if self.val_loader and (epoch % validate_every == 0):
                val_loss = self.val_epoch()
                self.val_losses.append(val_loss)

            self.epoch += 1
            self.save_checkpoint()

            if epoch == 0 or (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch+1}/{end_epoch}; Train loss: {train_loss:.4f}"
                    + (f", Val loss: {val_loss:.4f}" if val_loss else "")
                )

            if plot and ((epoch + 1) % print_every == 0 or epoch == end_epoch - 1):
                self.plot_learning_curve()

        return self.train_losses, self.val_losses

    @torch.no_grad()
    def evaluate(self, loader):
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
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "epoch": self.epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        torch.save(checkpoint, self.path)

    def load_checkpoint(self):
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

    def plot_learning_curve(
        self, save=False, path="./results/plots/learning_curve.png"
    ):
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

        if save:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            print(f"Saved plot to {path}")
        else:
            plt.show()
