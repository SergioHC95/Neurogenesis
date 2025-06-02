import sys

from loguru import logger


def configure_logging() -> None:
    # Remove default logger
    logger.remove()

    # Console logging (stderr)
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}:{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )

    # File logging (optional)
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        enqueue=True,  # Async logging for performance
    )
