"""Repository-relative locations shared by the paper and public mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def repository_root() -> Path:
    """Return the repository root relative to this source module."""
    return Path(__file__).resolve().parents[3]


def experiment_root(name: str = "digits") -> Path:
    """Return the data directory for a named vision experiment."""
    return repository_root() / "vision_tokens" / name


def prepare_folders(folder: Path) -> Path:
    """Create the standard experiment subdirectories and return their root."""
    folder = Path(folder)
    names = ("dataset", "checkpoints", "activations", "retrieval", "results")
    for name in names:
        (folder / name).mkdir(parents=True, exist_ok=True)
    return folder


def load_digit_cache(folder: Path, seed: int) -> dict[str, np.ndarray]:
    """Load all cached digit experiment arrays for a seed."""
    folder = Path(folder)
    result = {}
    paths = [folder / "dataset/digits_codexgen.npz",
             folder / f"activations/digits_seed_{seed}_codexgen.npz",
             folder / f"retrieval/digits_seed_{seed}_codexgen.npz"]
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            result.update({key: arrays[key] for key in arrays.files})
    return result
