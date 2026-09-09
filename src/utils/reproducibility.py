"""Experiment tracking and reproducibility infrastructure for RiceGuard."""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from src.utils.paths import ensure_dir, get_experiments_dir
from src.utils.seed import get_reproducibility_info


def get_git_commit_hash() -> str:
    """Retrieve the current Git commit SHA hash if inside a git repository."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        )
        return output.strip()
    except Exception:
        return "not_a_git_repo_or_git_missing"


def init_experiment(
    experiment_name: str,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    model_name: Optional[str] = None,
    dataset_version: str = "v1.0",
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Initialize a reproducible experiment directory with metadata tracking.

    Directory structure:
        experiments/<experiment_name>_<timestamp>/
            config.yaml
            logs/
            checkpoints/
            metrics/
            figures/
            environment.json
            results.json

    Args:
        experiment_name: Descriptive name for the experiment.
        config: Experiment configuration dictionary.
        seed: Random seed used for run.
        model_name: Model architecture tag.
        dataset_version: Dataset version identifier.
        base_dir: Base directory for experiments (default: experiments/).

    Returns:
        Dict[str, Path]: Map of created experiment subdirectories.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir_name = f"{experiment_name}_{timestamp}"

    root_exp_dir = Path(base_dir) if base_dir else get_experiments_dir()
    exp_dir = ensure_dir(root_exp_dir / exp_dir_name)

    # Create standard experiment subdirectories
    subdirs = {
        "root": exp_dir,
        "logs": ensure_dir(exp_dir / "logs"),
        "checkpoints": ensure_dir(exp_dir / "checkpoints"),
        "metrics": ensure_dir(exp_dir / "metrics"),
        "figures": ensure_dir(exp_dir / "figures"),
    }

    # Gather full environmental and hardware provenance
    env_info = get_reproducibility_info()
    git_hash = get_git_commit_hash()

    metadata = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "seed": seed,
        "model_name": model_name or "unspecified",
        "dataset_version": dataset_version,
        "git_commit_hash": git_hash,
        "environment": env_info,
    }

    # Save environment.json
    with open(exp_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save config.yaml
    if config is not None:
        with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Initialize empty results.json placeholder
    initial_results = {
        "experiment_name": experiment_name,
        "status": "INITIALIZED",
        "timestamp": timestamp,
        "metrics": {},
    }
    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(initial_results, f, indent=2)

    return subdirs
