"""Render context and matched-depth tables from audited saved results."""

from __future__ import annotations
from collections.abc import Sequence

import argparse
import json
from pathlib import Path
import re
import textwrap
import unicodedata

from .depth_codexgen import OUT, ROOT
from .galleries_codexgen import families
from lattmc.latex.texttables_codexgen import styles

NAMES = dict(gemma_matryoshka='Gemma-2', pythia_topk='Pythia',
             smol_topk='SmolLM2', qwen_transcoder='Qwen TC')
LABELS = ('Company', 'Education', 'Artist', 'Athlete', 'Office holder',
          'Transport', 'Building', 'Natural place', 'Village', 'Animal',
          'Plant', 'Album', 'Film', 'Written work')


def tex(value: str) -> str:
    """Normalize text and escape special characters for LaTeX."""
    text = unicodedata.normalize('NFKD', value)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.replace('\u2019', "'").replace('\u2013', '-')
    text = text.replace('\u2014', '--').replace('\u201c', '"')
    text = text.replace('\u201d', '"')
    text = ''.join(c if ord(c) < 128 else f'[U+{ord(c):04X}] ' for c in text)
    escapes = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
               '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
               '}': r'\}', '~': r'\textasciitilde{}',
               '^': r'\textasciicircum{}'}
    return ''.join(escapes.get(c, c) for c in re.sub(r'\s+', ' ', text))


def write(path: Path, lines: Sequence[str]) -> None:
    """Write wrapped LaTeX source lines to a table file."""
    from lattmc.latex.texttables_codexgen import write as shared_write
    shared_write(path, lines)


def table(columns: str, head: str) -> list[str]:
    """Open a LaTeX table with the specified columns and heading."""
    return [r'\begin{table}[htbp]', r'\centering\SampleTableSetup',
            r'\begin{tabular}{' + columns + '}', r'\toprule',
            head + r' \\', r'\midrule']


def end(
    lines: list[str],
    caption: str,
    label: str,
    env: str = 'tabular',
) -> list[str]:
    """Close a table and append its caption and label."""
    return lines + [r'\bottomrule\end{' + env + '}',
                    r'\caption{' + caption + '}',
                    r'\label{tab:' + label + '}', r'\end{table}']


def interval(x: dict[str, float]) -> str:
    """Format a signed percentage difference and confidence interval."""
    return f"{100*x['mean']:+.2f} [{100*x['low']:+.2f}, {100*x['high']:+.2f}]"


def main(paper: Path | None = None) -> None:
    """Generate matched-depth tables from the cached results."""
    paper = Path(paper or ROOT / 'texs/sparsesurrs/lattconference')
    tables = paper / 'tables'
    metrics = json.loads((OUT / 'context_metrics.json').read_text())
    assert json.loads((OUT / 'gallery_audit.json').read_text())['status'] == (
        'passed')
    reports = [json.loads((OUT / f'{name}_gallery.json').read_text())
               for name in NAMES]
    lines = table('llrrrl', 'Corpus & Block & Graded & Single & SAE cos.'
                  r' & $\Delta$ graded vs.\ 3')
    for d in metrics['depth']:
        m = d['methods']
        lines.append(f"{'AG' if d['dataset']=='ag_news' else 'DB'} & "
                     f"{d['layer']} & {100*m['graded']['mean']:.2f} & "
                     f"{100*m['best_single']['mean']:.2f} & "
                     f"{100*m['sae_cosine']['mean']:.2f} & "
                     + ('reference' if d['layer'] == 3 else
                        interval(d['delta_vs_3'])) + r' \\')
    write(tables / 'depth-main.tex', end(lines,
        'Matched SmolLM2-135M MLP-output TopK comparison. Macro AP is in '
        'percent; changes and paired 95\\% conditional intervals are in '
        'percentage points. Identical documents, source draws, dictionary '
        'width, and token sparsity are used at all blocks. Other readouts '
        'and coverage measures appear in \\appcref{app:depth-results}.',
        'depth-main'))
    lines = table('llrrrrrr', 'Corpus & Block & Graded & Support & Single'
                  ' & SAE cos. & Dense cos. & TF-IDF')
    for d in metrics['depth']:
        lines.append(('AG' if d['dataset'] == 'ag_news' else 'DB') +
                     f" & {d['layer']} & " + ' & '.join(
                         f"{100*v['mean']:.2f}" for v in d['methods'].values())
                     + r' \\')
    write(tables / 'depth-readouts.tex', end(lines,
        'All six matched-depth readouts, macro AP in percent. Every method '
        'uses the same three-positive source draws and calibration split. '
        'TF-IDF is unchanged across blocks because its inputs are unchanged.',
        'depth-readouts'))
    lines = table('llrrrrrr', 'Corpus & Block & Prec. & Recall & Size'
                  ' & Jaccard & Low AP & Low TF-IDF')
    for d in metrics['depth']:
        lines.append(('AG' if d['dataset'] == 'ag_news' else 'DB') +
                     f" & {d['layer']} & {100*d['precision']:.1f} & "
                     f"{100*d['recall']:.1f} & {d['extent']:.1f} & "
                     f"{d['extent_jaccard_vs_3']:.3f} & "
                     f"{100*d['low_ap']:.2f} & {100*d['low_tfidf']:.2f} "
                     + r'\\')
    write(tables / 'depth-coverage.tex', end(lines,
        'Calibrated graded extents and lower lexical-overlap evaluation. '
        'Precision, recall, and AP are percentages; Size is the mean number '
        'of retrieved test documents, and Jaccard compares each extent with '
        'its block-3 counterpart. Low AP and Low TF-IDF use the same '
        'task-specific half-test subset. Its mean positive prevalence is '
        '19.2\\% for AG and 1.3\\% for DB; neither subset is '
        'lexically disjoint.',
        'depth-coverage'))
    lines = table('llrrrr', 'Model/block & Context & Strict & Broad'
                  ' & Sibling & Sibling lift')
    for g in metrics['granularity']:
        if g['model'] == 'smol_topk':
            continue
        name = NAMES.get(g['model'], g['model'])
        for v in g['values']:
            m = v['methods']
            lines.append(name + ' & ' + v['context'] + ' & ' +
                         ' & '.join(f"{100*m[k]['mean']:.2f}"
                                    for k in ('strict', 'broad', 'sibling')) +
                         f" & {m['sibling']['lift']:.2f} " + r'\\')
    write(tables / 'depth-granularity.tex', end(lines,
        'Exploratory contextual granularity on DBpedia: geographic means '
        'NaturalPlace plus Village; biological means Animal plus Plant. '
        'Strict retains the source class, Broad includes its sibling, and '
        'Sibling excludes the source class from evaluation and targets only '
        'the sibling. AP is in percent. Constant-score AP baselines are '
        '7.14\\%, 14.29\\%, and 7.69\\%, respectively; Sibling lift divides '
        'AP by 7.69\\%. No query or threshold is retuned to broader labels.',
        'depth-granularity'))
    lines = [r'\begin{table}[htbp]', r'\centering\SampleTableSetup',
             r'\begin{tabularx}{\textwidth}{llX}', r'\toprule',
             r'Model & Query & Three training-source titles \\',
             r'\midrule']
    for name, report in zip(NAMES, reports):
        for label in (7, 9):
            row = next(r for r in report['records']
                       if r['task_label'] == label)
            lines.append(NAMES[name] + ' & ' + LABELS[label] + ' & ' +
                         tex('; '.join(row['source_titles'])) + r' \\')
    write(tables / 'depth-sources.tex', end(lines,
        'Training sources for the ranked galleries. All four checkpoints '
        'use the same recorded document identities within each query. '
        'Titles identify the source items; their full descriptions, not '
        'the title words alone, supply max-pooled activation queries.',
        'depth-sources', 'tabularx'))
    styles(paper)
    families(paper)
    print('Generated shared-style galleries and numerical tables.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path)
    main(parser.parse_args().paper)
