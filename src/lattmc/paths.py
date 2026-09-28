"""Utility module for default paths."""

from pathlib import Path
from typing import Tuple, Any

from joblib import dump, load


def init_paths() -> Tuple[Path, Path]:
    """Initialize paths for OWT tokens and transcoder activations.

    Returns:
        Tuple[Path, Path]: The paths.
    """
    OWT_PATH = Path().resolve().parents[1]
    PATH = OWT_PATH / 'notebooks' / 'transcoders' / 'data'

    return OWT_PATH, PATH


def gpt2_paths() -> Tuple[Path, Path]:
    """Initialize paths for sampled GPT-2 tokens and transcoder activations.

    Returns:
        Tuple[Path, Path]: The paths.
    """
    OWT_PATH, PATH = init_paths()
    GPT2 = PATH / 'transcoders' / 'gpt2'
    OWT_TOKENS_DIR = GPT2 / 'owt_tokens'
    TOKENS_PATH = OWT_TOKENS_DIR / 'owt_tokens_torch.pt'
    OWT_TOKENS_DIR.mkdir(exist_ok=True, parents=True)

    return GPT2, TOKENS_PATH


def tr_paths() -> Tuple[Path, Path]:
    """Initialize paths for sampled GPT-2 tokens and transcoder activations.

    Returns:
        Tuple[Path, Path]: The paths.
    """
    return gpt2_paths()


def sae_paths() -> Tuple[Path, Path]:
    """Initialize paths for sampled GPT-2tokens and SAE activations.

    Returns:
        Tuple[Path, Path]: The paths.
    """
    OWT_PATH, PATH = init_paths()
    SAE_PATH = OWT_PATH / 'notebooks' / 'sae' / 'data'
    GPT2_TR, TOKENS_PATH = gpt2_paths()
    GPT2 = SAE_PATH / 'sae' / 'gpt2'
    GPT2.mkdir(exist_ok=True, parents=True)

    return GPT2, TOKENS_PATH


def _tmp_path() -> Path:
    """Get the path to the temporary directory.

    Returns:
        Path: The path to the temporary directory.
    """
    tmp_path = Path().resolve().parents[1] / 'notebooks' / 'data' / 'tmp'
    tmp_path.mkdir(parents=True, exist_ok=True)

    return tmp_path


def save_tmp(obj: Any, file_name: str = 'tmp.joblib'):
    """Save the object to a temporary file.

    Args:
        obj (Any): The object to save.
        file_name (str): The name of the file to save the object to.
    """
    dump(obj, _tmp_path() / file_name)


def load_tmp(file_name: str = 'tmp.joblib') -> Any:
    """Load the object from a temporary file.

    Args:
        file_name (str): The name of the file to load the object from.
    Returns:
        Any: The loaded object.
    """
    return load(_tmp_path() / file_name)
