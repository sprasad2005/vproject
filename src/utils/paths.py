"""Path management utility for the RiceGuard project.

Provides dynamic project root resolution and relative path generation
so the codebase remains portable across environments without hardcoded paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Dynamically determine and return the absolute Path to the project root directory."""
    # This file is located at <project_root>/src/utils/paths.py
    return Path(__file__).resolve().parent.parent.parent


def get_path(*relative_parts: Union[str, Path]) -> Path:
    """Resolve a path relative to the project root.

    Args:
        *relative_parts: Subdirectories or file paths relative to project root.

    Returns:
        Path: Absolute resolved Path.
    """
    root = get_project_root()
    return root.joinpath(*relative_parts)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure that a directory exists, creating parents if necessary.

    Args:
        path: Path or string representing directory path.

    Returns:
        Path: Resolved directory Path.
    """
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# Common predefined path helpers
def get_configs_dir() -> Path:
    """Return path to configs/ directory."""
    return get_path("configs")


def get_data_dir() -> Path:
    """Return path to data/ directory."""
    return get_path("data")


def get_experiments_dir() -> Path:
    """Return path to experiments/ directory."""
    return ensure_dir(get_path("experiments"))


def get_logs_dir() -> Path:
    """Return path to logs/ directory."""
    return ensure_dir(get_path("logs"))


def get_checkpoints_dir() -> Path:
    """Return path to checkpoints/ directory."""
    return ensure_dir(get_path("checkpoints"))


def get_results_dir() -> Path:
    """Return path to results/ directory."""
    return ensure_dir(get_path("results"))
