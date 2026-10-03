"""Generate traceable contextual tables from audited activation records."""

from __future__ import annotations
from collections.abc import Sequence

import argparse
import json
from pathlib import Path
import re
import textwrap
import unicodedata

from transformers import AutoTokenizer

from .audit_codexgen import DATA, ROOT
from .galleries_codexgen import contexts
from lattmc.latex.texttables_codexgen import styles

TOKENIZER = None

LABELS = {'nyc': 'NYC', 'rio': 'Rio', 'animals': 'Cat/dog',
          'sports': 'Sports'}


def tex(text: str) -> str:
    """Escape text for inclusion in LaTeX reports."""
    replacements = {'\u2019': "'", '\u2018': "'", '\u201c': '"',
                    '\u201d': '"', '\u2014': '--', '\u2013': '-',
                    '\u2026': '...'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.encode('ascii', 'replace').decode()
    text = re.sub(r'\s+', ' ', text)
    escaped = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
               '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
               '}': r'\}', '~': r'\textasciitilde{}',
               '^': r'\textasciicircum{}'}
    return ''.join(escaped.get(c, c) for c in text)


def write(path: Path, lines: Sequence[str]) -> None:
    """Write wrapped LaTeX lines to a report artifact."""
    from lattmc.latex.texttables_codexgen import write as shared_write
    shared_write(path, lines)


def main(paper: Path | None = None) -> None:
    """Generate contextual retrieval report tables from cached evidence."""
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(
        'gpt2', local_files_only=True,
    )
    audit = json.loads((DATA / 'audit.json').read_text())
    assert audit['checks_passed']
    reports = [json.loads(p.read_text())
               for p in sorted(DATA.glob('*layer*.json'))]
    reports.sort(key=lambda r: (r['kind'], r['layer']))
    records = []
    for report in reports:
        for case in report['cases']:
            for entry in case['rows']:
                records.append((report, case, entry))
    if paper is None:
        paper = ROOT / 'texs/sparsesurrs/lattconference'
    tables = Path(paper) / 'tables'
    lines = [r'\begin{table}[htbp]', r'\centering\SampleTableSetup',
             r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Model/block & Query & $|G(u)|$ & Eligible & Changed & Losses'
             r' & Trials \\', r'\midrule']
    for c in audit['conditions']:
        model = c['kind'].upper()
        lines.append(
            f"{model}/{c['layer']} & {LABELS[c['case']]} & "
            f"{c['extent']:,} & {c['eligible']:,} & {c['changes']} & "
            f"{c['losses']} & {c['prefix']} " + r'\\',
        )
    lines.extend([r'\bottomrule\end{tabular}',
                  r'\caption{Contextual witness audit. Eligible rows exclude'
                  ' the source and the stated whole-word matches. Each'
                  ' condition contributes three sampled rows. Trials counts'
                  ' nonidentity prefix permutations; Changed counts material'
                  ' changes of the diagnostic activation and Losses counts'
                  ' downward crossings of its query threshold. These are'
                  ' token-witness outcomes, not whole-row retrieval'
                  ' failures.}',
                  r'\label{tab:context-counts}', r'\end{table}'])
    write(tables / 'context-counts.tex', lines)
    styles(paper)
    contexts(Path(paper))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path)
    main(parser.parse_args().paper)
