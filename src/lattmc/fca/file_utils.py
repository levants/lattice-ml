from pathlib import Path
from typing import Union


def not_exists(file_path: Union[str, Path]) -> bool:
    return not Path(file_path).exists()
