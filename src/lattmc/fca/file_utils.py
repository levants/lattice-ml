"""File utilities for Lattice-theoretic Formal Concept Analysis (FCA)."""

from pathlib import Path
from typing import Union


def exists(file_path: Union[str, Path]) -> bool:
    """Check if a file exists on the path.

    Args:
        file_path (Union[str, Path]): The path to the file.
    Returns:
        bool: True if the file exists, False otherwise.
    """
    return Path(file_path).exists()


def not_exists(file_path: Union[str, Path]) -> bool:
    """Check if a file does not exist on the path.

    Args:
        file_path (Union[str, Path]): The path to the file.
    Returns:
        bool: True if the file does not exist, False otherwise.
    """
    return not exists(file_path)


def save_obj(obj: Any, file_path: Union[str, Path]) -> None:
    """Serialize and save an object to a file using joblib.

    Args:
        obj (Any): The object to save.
        file_path (Union[str, Path]): The path to the file.
    """
    joblib.dump(obj, file_path)


def load_obj(file_path: Union[str, Path]) -> Any:
    """Deserialize and load an object from a file using joblib.

    Args:
        file_path (Union[str, Path]): The path to the file.
    Returns:
        Any: The loaded object.
    """
    return joblib.load(file_path)
