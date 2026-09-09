"""Logging system for RiceGuard research experiments.

Supports standardized formatting:
    `YYYY-MM-DD HH:MM:SS | LEVEL | module | message`
with simultaneous output to console and experiment log files.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from src.utils.paths import get_logs_dir


class ColoredFormatter(logging.Formatter):
    """Custom formatter with standard timestamp and module formatting."""

    def format(self, record: logging.LogRecord) -> str:
        # Standardize module name
        record.short_name = record.name.split(".")[-1]
        return super().format(record)


def setup_logger(
    name: str = "riceguard",
    log_file: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> logging.Logger:
    """Configure and return a structured logger instance.

    Args:
        name: Logger name (e.g. 'training', 'evaluation').
        log_file: Optional path or filename for log output. Defaults to logs/<name>.log.
        log_level: Logging severity level (default: logging.INFO).
        log_to_console: If True, stream logs to sys.stdout.
        log_to_file: If True, write logs to target file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers on re-initialization
    if logger.handlers:
        logger.handlers.clear()

    # Format: 2026-09-08 18:00:00 | INFO | module | message
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_to_file:
        if log_file is None:
            logs_dir = get_logs_dir()
            log_path = logs_dir / f"{name}.log"
        else:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "riceguard") -> logging.Logger:
    """Retrieve an existing logger or create one with default settings."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name=name)
    return logger
