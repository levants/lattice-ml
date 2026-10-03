"""Generate publication tables directly from all frozen results."""

from __future__ import annotations
from typing import Any
from collections.abc import Sequence

import argparse
import json
from lattmc.latex.naming_codexgen import named_table
from pathlib import Path

import numpy as np

NAMES = {
    'pythia_topk': 'Pythia', 'smol_topk': 'SmolLM2',
    'qwen_transcoder': 'Qwen TC', 'gemma_matryoshka': 'Mat-32k',
    'gemma_matryoshka_512': 'Mat-512',
    'gemma_matryoshka_2048': 'Mat-2048',
}
DATA = {'ag_news': 'AG News', 'dbpedia_14': 'DBpedia'}
METHODS = {'graded': 'Graded', 'support': 'Support',
           'matched': 'Matched support', 'single': 'First feature',
           'full': 'Full query', 'best_single': 'Best single',
           'sae_cosine': 'Code cosine', 'sae_probe': 'Code probe',
           'dense_cosine': 'Dense cosine', 'dense_probe': 'Dense probe',
           'tfidf_cosine': 'TF-IDF cosine', 'tfidf_probe': 'TF-IDF probe',
           'random': 'Random source'}


def read(path: Path) -> dict[str, Any]:
    """Load a JSON report from disk."""
    return json.loads(path.read_text())


@named_table
def table(
    label: str,
    caption: str,
    columns: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> str:
    """Format a labeled LaTeX table and wrap long source lines."""
    lines = [r'\begin{table}[tbp]', r'\centering', r'\small',
             r'\setlength{\tabcolsep}{4pt}',
             r'\caption{' + caption + '}',
             r'\label{tab:families-' + label + '}',
             r'\begin{tabular}{' + columns + '}', r'\toprule',
             ' & '.join(headers) + r' \\', r'\midrule']
    lines += [' & '.join(map(str, row)) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    # TeX ignores a newline between cells; wrap long rows at cell boundaries.
    wrapped = []
    for line in lines:
        while len(line) > 79 and ' & ' in line:
            cut = line.rfind(' & ', 0, 76)
            if cut < 0:
                break
            wrapped.append(line[:cut + 2])
            line = '  ' + line[cut + 3:]
        wrapped.append(line)
    import textwrap
    return '\n'.join('\n'.join(textwrap.wrap(
        line, width=79, break_long_words=False, break_on_hyphens=False))
        if len(line) > 79 else line for line in wrapped)


def generate(output: Path, tables: Path) -> None:
    """Generate LaTeX tables from cached evaluation summaries."""
    tables.mkdir(parents=True, exist_ok=True)
    reports = {(d, n, p): read(output / d / f'{n}_{p}_results.json')
               for d in DATA for n in NAMES for p in ('max', 'mean')}
    main, ablation, diagnostics, coincidence, sensitivity = [], [], [], [], []
    for d in DATA:
        for n in NAMES:
            r = reports[d, n, 'max']
            s = r['summary']['3']
            def val(method: str) -> str:
                """Format the method's mean average precision as a percentage.
                """
                return f"{100 * s['methods'][method]['mean']:.1f}"
            diff = s['paired_graded']['support']
            ci = (f"{100 * diff['mean']:.1f} "
                  f"[{100 * diff['low']:.1f}, {100 * diff['high']:.1f}]")
            main.append([DATA[d], NAMES[n], val('graded'), val('support'),
                         val('best_single'), val('dense_probe')])
            ablation.append([DATA[d], NAMES[n], ci,
                             val('matched'), val('single'),
                             str(s['full_empty'])])
            extraction = read(output / d / f'{n}_extraction.json')
            ex = extraction['statistics']
            width = extraction['files']['max']['shape'][1]
            diagnostics.append([
                DATA[d], NAMES[n], f'{width:,}', f"{ex['test_tokens']:,}",
                f"{ex['mean_token_l0']:.1f}",
                f"{ex['uncentered_nmse']:.4f}"])
            scores = np.load(output / d / f'{n}_max_scores.npz')
            both = only_g = only_s = neither = 0
            for i, record in enumerate(r['records']):
                if record['shot'] != 3:
                    continue
                g = scores['graded'][i] >= record['thresholds']['graded']
                b = scores['support'][i] >= .5
                both += int(sum(g & b))
                only_g += int(sum(g & ~b))
                only_s += int(sum(~g & b))
                neither += int(sum(~g & ~b))
            coincidence.append([DATA[d], NAMES[n], both, only_g,
                                only_s, neither])
            half = s['half_test_ap']['graded']
            low = s['low_overlap']['graded']
            sensitivity.append([
                DATA[d], NAMES[n],
                f'{100 * min(half):.1f}--{100 * max(half):.1f}',
                f"{100 * low['ap']:.1f}",
                f"{100 * low['prevalence']:.1f}", low['valid_tasks']])
    specs = [
        ('main', 'Additional-family test AP (\\%). '
         'Three sources; max pooling. '
         'Macro averages over all classes and ten source draws. Best single '
         'uses calibration feature selection; dense probe uses matched '
         'positive and negative source examples. Full results and paired '
         'intervals appear in the appendix.',
         'llrrrr', ['Corpus', 'View', 'Graded', 'Support', 'Best single',
                     'Dense probe'], main),
        ('checkpoints', 'Measured test-token activity and uncentered '
         'reconstruction error. Mat-512 and Mat-2048 are nested prefix views; '
         'Mat-32k is the full dictionary. Both pooling rules '
         'use the same token activations.', 'llrrrr',
         ['Corpus', 'View', 'Width', 'Tokens', 'Mean $L_{0}$', 'NMSE'],
         diagnostics),
        ('ablation', 'Max-pooling, three-source ablations. The paired '
         'graded-minus-support difference is in AP percentage points, with '
         'a conditional 95\\% source-draw interval. Matched support and first '
         'feature entries are AP percentages. Empty counts refer to full '
         'queries at their calibrated threshold (40 tasks per AG News '
         'checkpoint; 140 per DBpedia checkpoint).', 'llrrrr',
         ['Corpus', 'View', '$\\Delta$ [95\\% interval]', 'Matched',
          'First', 'Empty'], ablation),
        ('coincidence', 'Retrieval coincidence at calibrated thresholds: '
         'graded (G) versus independently calibrated support (S), max '
         'pooling and three sources. Counts pool repeated query--document '
         'decisions, not independent documents.', 'llrrrr',
         ['Corpus', 'View', 'Both', 'G only', 'S only', 'Neither'],
         coincidence),
        ('sensitivity', 'Sensitivity of graded retrieval, max pooling and '
         'three sources. Half-test gives the range of macro AP across three '
         'stratified subsamples. Low-overlap AP and positive prevalence '
         '(both percentages) use only tasks with both labels; the final '
         'column counts those tasks.', 'llrrrr',
         ['Corpus', 'View', 'Half-test AP', 'Low AP', 'Prevalence', 'Tasks'],
         sensitivity),
    ]
    for label, caption, columns, headers, rows in specs:
        (tables / f'families-{label}.tex').write_text(
            table(label, caption, columns, headers, rows))
    complete = []
    for d in DATA:
        for pooling in ('max', 'mean'):
            for shot in ('3', '10'):
                rows = []
                for method, label in METHODS.items():
                    values = [reports[d, n, pooling]['summary'][shot]
                              ['methods'][method]['mean'] for n in NAMES]
                    rows.append([label] + [f'{100 * v:.2f}' for v in values])
                caption = (f'{DATA[d]}: all methods, {pooling} pooling, '
                           f'{shot} positive sources. Macro test AP (\\%).')
                label = f'complete-{d.replace("_", "-")}-{pooling}-{shot}'
                complete.append(table(label, caption, 'lrrrrrr',
                                      ['Method'] + list(NAMES.values()), rows))
    (tables / 'families-all.tex').write_text('\n'.join(complete))
    confusion_tables = []
    paired_tables = []
    for d in DATA:
        confusion = []
        for n in NAMES:
            summary = reports[d, n, 'max']['summary']['3']
            for method in ('graded', 'support', 'best_single', 'dense_probe'):
                c = summary['confusion'][method]
                confusion.append([NAMES[n], METHODS[method]]
                                 + [c[k] for k in ('tp', 'fp', 'fn', 'tn')])
        confusion_tables.append(table(
            'confusion-' + d.replace('_', '-'),
            DATA[d] + ': calibrated confusion counts, maximum pooling and '
            'three sources. Counts reuse test documents across queries.',
            'llrrrr', ['View', 'Method', 'TP', 'FP', 'FN', 'TN'], confusion))
        for pooling in ('max', 'mean'):
            rows = []
            for n in NAMES:
                for shot in ('3', '10'):
                    item = reports[d, n, pooling]['summary'][shot]
                    row = [NAMES[n], shot]
                    for method in ('support', 'best_single'):
                        v = item['paired_graded'][method]
                        row.append(f"{100*v['mean']:.1f} "
                                   f"[{100*v['low']:.1f}, "
                                   f"{100*v['high']:.1f}]")
                    row += [item['full_empty'], item['zero_queries']]
                    rows.append(row)
            paired_tables.append(table(
                'paired-' + d.replace('_', '-') + '-' + pooling,
                DATA[d] + ': ' + pooling + ' pooling. Graded-minus-baseline '
                'AP differences in percentage points with conditional '
                '95\\% source-draw intervals. Empty counts full-query '
                'extents; zero counts source meets with no positive entry.',
                'llrrrr', ['View', 'Sources', 'Versus support',
                           'Versus single', 'Empty', 'Zero'], rows))
    (tables / 'families-confusion.tex').write_text(
        '\n'.join(confusion_tables))
    (tables / 'families-paired.tex').write_text('\n'.join(paired_tables))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tables', type=Path, required=True)
    args = parser.parse_args()
    generate(args.output, args.tables)
