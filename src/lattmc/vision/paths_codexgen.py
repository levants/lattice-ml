"""Repository-relative locations shared by the paper and public mirror."""

from pathlib import Path

import numpy as np


def repository_root():
    return Path(__file__).resolve().parents[3]


def experiment_root(name="digits"):
    return repository_root() / "vision_tokens" / name


def prepare_folders(folder):
    folder = Path(folder)
    names = ("dataset", "checkpoints", "activations", "retrieval", "results")
    for name in names:
        (folder / name).mkdir(parents=True, exist_ok=True)
    return folder


def load_digit_cache(folder, seed):
    folder = Path(folder)
    result = {}
    paths = [folder / "dataset/digits_codexgen.npz",
             folder / f"activations/digits_seed_{seed}_codexgen.npz",
             folder / f"retrieval/digits_seed_{seed}_codexgen.npz"]
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            result.update({key: arrays[key] for key in arrays.files})
    return result
