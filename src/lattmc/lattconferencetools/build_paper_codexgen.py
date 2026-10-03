"""Build reference metadata, then the main, appendix, and selected PDFs."""

from __future__ import annotations

from .paths_codexgen import PAPER, CACHE, PROTOCOL, NOTEBOOKS

import subprocess


def main() -> None:
    """Build reference metadata, then the main, appendix, and selected PDFs."""
    directory = PAPER
    sources = (
        "lattconference-combined",
        "lattconference-main",
        "lattconference-appendix",
        "lattconference",
    )
    for source in sources:
        subprocess.run(
            [
                "latexmk", "-pdf", "-synctex=1",
                "-interaction=nonstopmode", "-halt-on-error",
                "-file-line-error", f"{source}.tex",
            ],
            cwd=directory,
            check=True,
        )


if __name__ == "__main__":
    main()
