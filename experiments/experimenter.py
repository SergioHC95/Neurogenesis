import json
import os
from datetime import datetime

# Import your Trainer and callbacks

# You can import your custom recipe trainers too if you wish.


class Experiment:
    """
    Manages all experiment setup, path construction, and Trainer instantiation.
    All files for a run go into a unique experiment folder.

    Args:
        experiment_name (str): Name of the experiment (will be prepended to folder and files).
        TrainerClass (type): The Trainer or Trainer subclass to use.
        model_builder (callable): Function that builds/returns the model.
        optimizer_builder (callable): Function that builds/returns the optimizer.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        config (dict or object): Optional config to save for reproducibility.
        save_config (bool): Whether to save config in experiment folder.
        extra_callbacks (list): Additional callbacks to pass to Trainer.
        resume_from (str, optional): Path to checkpoint to load weights/optimizer state from.
        **trainer_kwargs: Passed through to the Trainer.
    """

    def __init__(
        self,
        experiment_name,
        TrainerClass,
        model_builder,
        optimizer_builder,
        loss_f,
        train_loader,
        val_loader=None,
        config=None,
        save_config=True,
        extra_callbacks=None,
        resume_from=None,
        **trainer_kwargs,
    ):
        # Timestamp for unique folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_name = experiment_name
        self.timestamp = timestamp
        self.exp_dir = os.path.join("results", f"{experiment_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # Define all file paths for this run
        self.log_dir = self.exp_dir
        self.profile_dir = self.exp_dir
        self.diagnostics_dir = self.exp_dir
        self.checkpoint_path = os.path.join(self.exp_dir, "checkpoint.pt")
        self.plot_path = os.path.join(self.exp_dir, "learning_curve.png")
        self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")
        self.metrics_csv = os.path.join(self.exp_dir, "metrics.csv")
        self.profile_csv = os.path.join(self.exp_dir, "profile_stats.csv")
        self.diagnostics_json = os.path.join(self.exp_dir, "diagnostics_stats.json")

        # Save config if provided
        if config is not None and save_config:
            config_path = os.path.join(self.exp_dir, "config.json")
            try:
                with open(config_path, "w") as f:
                    if isinstance(config, dict):
                        json.dump(config, f, indent=2)
                    else:
                        # Try to convert to dict if possible
                        json.dump(vars(config), f, indent=2)
                # print(f"[Experiment] Saved config to {config_path}")
            except Exception as e:
                print(f"[Experiment] Could not save config: {e}")

        # Build model and optimizer
        self.model = model_builder()
        self.optim = optimizer_builder(self.model)

        self.loss_f = loss_f
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Only load a checkpoint if resume_from is set
        if resume_from is not None:
            if not os.path.isfile(resume_from):
                raise FileNotFoundError(
                    f"resume_from path does not exist: {resume_from}"
                )
            resume_path = resume_from
            print(f"[Experiment] Resuming weights/state from: {resume_path}")
        else:
            resume_path = None

        self.trainer = TrainerClass(
            self.model,
            self.optim,
            self.loss_f,
            self.train_loader,
            self.val_loader,
            diagnostics_dir=self.diagnostics_dir,
            log_dir=self.log_dir,
            profile_dir=self.profile_dir,
            checkpoint_path=self.checkpoint_path,
            plot_path=self.plot_path,
            extra_callbacks=extra_callbacks,
            resume_from=resume_path,
            **trainer_kwargs,
        )

    def run(self, epochs=1, **train_kwargs):
        """
        Runs the training process for the specified number of epochs.

        Args:
            epochs (int): Number of epochs to train.
            **train_kwargs: Any additional Trainer.train() keyword arguments.
        """
        print(f"\n=== Running Experiment: {self.exp_name} @ {self.timestamp} ===")
        # print(f"  Results saved to: {self.exp_dir}")
        self.trainer.train(epochs=epochs, **train_kwargs)
        print(f"[Experiment] Done. All files for this run are in: {self.exp_dir}")

    def get_exp_dir(self):
        """
        Returns:
            str: The root folder for this experiment's outputs.
        """
        return self.exp_dir

    def get_trainer(self):
        """
        Returns:
            Trainer: The Trainer instance used for this experiment.
        """
        return self.trainer
