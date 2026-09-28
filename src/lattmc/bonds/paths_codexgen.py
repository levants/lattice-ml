"""Shared repository paths, independent of the working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / 'texs/crossbonds/surrogates'
PACKAGE = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / 'notebooks/bonds'
OUT = NOTEBOOKS / 'data'
NOTEBOOK = NOTEBOOKS / 'bonds_layers_codexgen.ipynb'
