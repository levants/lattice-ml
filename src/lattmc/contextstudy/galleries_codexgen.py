"""Compatibility entry point for complete measured token-witness galleries."""

import argparse
from pathlib import Path

from .witnessrender_codexgen import contexts, families, main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path)
    main(parser.parse_args().paper)
