# custom_trainers.py

from src.training.callbacks import (
    CheckpointCallback,
    DiagnosticsCallback,
    LoggingCallback,
    NotebookPlotCallback,
    ProfilingCallback,
)
from src.training.trainer import Trainer


class KitchenSinkTrainer(Trainer):
    """
    Trainer with all logging, profiling, diagnostics, notebook plotting, and checkpointing.
    Ideal for comprehensive experiments with maximum monitoring and recovery.
    """

    def __init__(
        self,
        model,
        optim,
        loss_f,
        train_loader,
        val_loader=None,
        *,
        diagnostics_dir,
        log_dir,
        profile_dir,
        checkpoint_path,
        plot_path,
        extra_callbacks=None,
        resume_from=None,
        **kwargs,
    ):
        cb = [
            LoggingCallback(log_dir=log_dir, log_csv=True, log_tb=True),
            ProfilingCallback(profile_dir=profile_dir),
            DiagnosticsCallback(diagnostics_dir=diagnostics_dir),
            NotebookPlotCallback(plot_path=plot_path),
            CheckpointCallback(checkpoint_path=checkpoint_path),
        ]
        if extra_callbacks:
            cb.extend(extra_callbacks)
        super().__init__(
            model=model,
            optim=optim,
            loss_f=loss_f,
            train_loader=train_loader,
            val_loader=val_loader,
            callbacks=cb,
            resume_from=resume_from,
            **kwargs,
        )


class HeadlessTrainer(Trainer):
    """
    Trainer with all logging, profiling, diagnostics, and checkpointing,
    but no live notebook plotting (for cluster/terminal use).
    """

    def __init__(
        self,
        model,
        optim,
        loss_f,
        train_loader,
        val_loader=None,
        *,
        diagnostics_dir,
        log_dir,
        profile_dir,
        checkpoint_path,
        extra_callbacks=None,
        resume_from=None,
        **kwargs,
    ):
        cb = [
            LoggingCallback(log_dir=log_dir, log_csv=True, log_tb=True),
            ProfilingCallback(profile_dir=profile_dir),
            DiagnosticsCallback(diagnostics_dir=diagnostics_dir),
            CheckpointCallback(checkpoint_path=checkpoint_path),
        ]
        if extra_callbacks:
            cb.extend(extra_callbacks)
        super().__init__(
            model=model,
            optim=optim,
            loss_f=loss_f,
            train_loader=train_loader,
            val_loader=val_loader,
            callbacks=cb,
            resume_from=resume_from,
            **kwargs,
        )


class NotebookLiteTrainer(Trainer):
    """
    Trainer with just live notebook plotting for rapid, minimal experimentation.
    No logs, profiling, diagnostics, or checkpointing.
    """

    def __init__(
        self,
        model,
        optim,
        loss_f,
        train_loader,
        val_loader=None,
        *,
        plot_path=None,
        extra_callbacks=None,
        resume_from=None,
        **kwargs,
    ):
        cb = [NotebookPlotCallback(plot_path=plot_path)]
        if extra_callbacks:
            cb.extend(extra_callbacks)
        super().__init__(
            model=model,
            optim=optim,
            loss_f=loss_f,
            train_loader=train_loader,
            val_loader=val_loader,
            callbacks=cb,
            resume_from=resume_from,
            **kwargs,
        )


class DiagnosticsTrainer(Trainer):
    """
    Trainer with diagnostics only—collects layer stats, but no logging, plotting, or checkpointing.
    """

    def __init__(
        self,
        model,
        optim,
        loss_f,
        train_loader,
        val_loader=None,
        *,
        diagnostics_dir,
        extra_callbacks=None,
        resume_from=None,
        **kwargs,
    ):
        cb = [DiagnosticsCallback(diagnostics_dir=diagnostics_dir)]
        if extra_callbacks:
            cb.extend(extra_callbacks)
        super().__init__(
            model=model,
            optim=optim,
            loss_f=loss_f,
            train_loader=train_loader,
            val_loader=val_loader,
            callbacks=cb,
            resume_from=resume_from,
            **kwargs,
        )


class LoggingTrainer(Trainer):
    """
    Trainer with logging only (CSV, TensorBoard); no diagnostics, profiling, or plotting.
    """

    def __init__(
        self,
        model,
        optim,
        loss_f,
        train_loader,
        val_loader=None,
        *,
        log_dir,
        extra_callbacks=None,
        resume_from=None,
        **kwargs,
    ):
        cb = [LoggingCallback(log_dir=log_dir, log_csv=True, log_tb=True)]
        if extra_callbacks:
            cb.extend(extra_callbacks)
        super().__init__(
            model=model,
            optim=optim,
            loss_f=loss_f,
            train_loader=train_loader,
            val_loader=val_loader,
            callbacks=cb,
            resume_from=resume_from,
            **kwargs,
        )
