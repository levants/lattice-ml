"""Resolve separate manuscript, code, cache, and release roots."""

from __future__ import annotations

import os
from pathlib import Path


def repository() -> Path:
    """Locate the repository root, honoring an explicit environment override.
    """
    override = os.environ.get('MY_PAPERS_REPOSITORY')
    if override:
        return Path(override).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file():
            return parent
    raise FileNotFoundError('Set MY_PAPERS_REPOSITORY to the project root.')


ROOT = repository()
PAPER = Path(os.environ.get(
    'LATTCONFERENCE_PAPER',
    ROOT / 'texs/sparsesurrs/lattconference')).resolve()
CACHE = ROOT / 'data/activation_studies/legacy/lattconference'
RELEASE = ROOT / 'artifacts/releases/activation_studies/legacy/lattconference'
PROTOCOL = (ROOT / 'experiments/activation_studies/legacy/lattconference'
            / 'CONFERENCE_PROTOCOL.md')
NOTEBOOKS = ROOT / 'notebooks/sae'
TEMPLATES = ROOT / 'artifacts/templates/lattconference'
ORGANIZATION = ROOT / 'experiments/activation_studies/organization_v2'
