# custom_trainers.py

from src.training.trainers import (
    DiagnosticsTrainer,
    LoggingTrainer,
    NotebookTrainer,
    ProfilingTrainer,
)


class DiagnosticsLoggingTrainer(DiagnosticsTrainer, LoggingTrainer):
    """
    Combines diagnostics (stats/hooks) and logging (CSV, TensorBoard).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NotebookDiagnosticsTrainer(NotebookTrainer, DiagnosticsTrainer):
    """
    For notebooks: live plots + diagnostics (no logging/profiling by default).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NotebookLoggingTrainer(NotebookTrainer, LoggingTrainer):
    """
    For notebooks: live plots + CSV/TensorBoard logging.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ProfilingLoggingTrainer(ProfilingTrainer, LoggingTrainer):
    """
    Tracks profiling (FLOPs, updates) and logs all core metrics (CSV/TensorBoard).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ProfilingDiagnosticsTrainer(ProfilingTrainer, DiagnosticsTrainer):
    """
    Profiles (FLOPs, updates) and tracks diagnostics (gradient/activation stats).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ProfilingDiagnosticsLoggingTrainer(
    ProfilingTrainer, DiagnosticsTrainer, LoggingTrainer
):
    """
    Profiling (FLOPs/updates) + diagnostics + logs (CSV/TensorBoard).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NotebookFullFeaturedTrainer(
    NotebookTrainer, ProfilingTrainer, DiagnosticsTrainer, LoggingTrainer
):
    """
    The kitchen sink: notebook-friendly, diagnostics, logging, and profiling.
    Interactive live plots, diagnostics, logging, and profiling in one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
