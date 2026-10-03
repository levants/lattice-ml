"""Regenerate historical Cat/dog tables without inventing missing row IDs."""

from __future__ import annotations
from typing import Any
from pathlib import Path

import json

from lattmc.latex.texttables_codexgen import paired, tex, write
from lattmc.latex.naming_codexgen import panel
from .witnesses_codexgen import DEST
from .witnessrender_codexgen import LEGEND, entry


def records() -> list[dict[str, Any]]:
    """Load and combine records from the historical replay artifacts."""
    rows = json.loads((DEST / 'legacy.json').read_text())['records']
    excerpts = json.loads((DEST / 'legacy_excerpts.json').read_text())
    for i, row in enumerate(rows, 1):
        row['display_id'] = f'L{i:03d}'
        if row['status'] == 'U':
            row['excerpt'] = excerpts[row['id']]
    return rows


def main(paper: Path) -> None:
    """Write historical replay tables for the paper."""
    rows = records()
    for layer in (8, 11):
        for mode in ('exact', 'floor'):
            cells = []
            for kind in ('tc', 'sae'):
                items = []
                for row in rows:
                    if (row['family'], row['layer'], row['mode']) != (
                            kind, layer, mode):
                        continue
                    if row['status'] == 'U':
                        candidates = ', '.join(map(str, row['candidates']))
                        items.append(r'\item \textbf{U} ' +
                            row['display_id'] + ': ``' +
                            tex(row['excerpt']) + "''" +
                            r'\par\emph{Unresolved historical row identity; '
                            'candidate rows: ' + candidates +
                            '. No activation classification or highlight '
                            'is assigned.}')
                    else:
                        items.append(entry(row))
                cells.append(items)
            stem = f'layer{layer}-comparison'
            if mode == 'floor':
                stem += '-min-act'
            text = ('Exact full-code' if mode == 'exact' else 'Support-floor')
            write(paper / 'tables' / (stem + '.tex'), paired([
                (panel('tc', layer), panel('sae', layer),
                 *cells)], text + ' Cat/dog meet. All original example '
                'slots remain in their original order. Identifiable items '
                'have targeted measured replays; U denotes an unresolved '
                'historical identity and retains its original excerpt '
                'without activation highlighting. ' + LEGEND,
                'tab:' + stem))
