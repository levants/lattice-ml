def green(s: str) -> str:
    """
    Wrap `s` in ANSI escape codes for green text.
    Resets styling at the end.
    """
    return f"\033[32m{s}\033[0m"


def with_background(text: str, bg_code: int) -> str:
    """
    Wrap `text` in ANSI escape codes for background color `bg_code`.
    Resets styling at the end.
    """
    return f"\033[{bg_code}m{text}\033[0m"
