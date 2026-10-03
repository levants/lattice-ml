"""Generate manuscript tables from the executed graded-query manifests."""

from __future__ import annotations
from typing import Any
from collections.abc import Sequence

from .paths_codexgen import PAPER, CACHE, PROTOCOL, NOTEBOOKS

import json
import textwrap

import numpy as np

from .activation_experiments_codexgen import ALPHAS, CASES, coincidence

HERE = PAPER
RESULTS = CACHE / 'activation_results'
TABLES = HERE / 'tables'
NAMES = dict(nyc='New York City', animals='Cat/dog', rio='Rio de Janeiro',
             richmond='Richmond Hill', park='National Park Service',
             sports='Sports names')
SHORT = dict(nyc='NYC', animals='Cat/dog', rio='Rio', richmond='Richmond',
             park='Park Service', sports='Sports')
REPORTS = {}
ARRAYS = {}
for kind in ['sae', 'tc']:
    for layer in [0, 8, 11]:
        path = RESULTS / f'{kind}_layer{layer}'
        REPORTS[kind, layer] = json.loads(
            path.with_suffix('.json').read_text(),
        )
        ARRAYS[kind, layer] = np.load(path.with_suffix('.npz'))


def record(
    kind: str,
    layer: int,
    case: str,
    operation: str,
    alpha: float | None,
) -> dict[str, Any]:
    """Retrieve the report record for a specific query condition."""
    return next(r for r in REPORTS[kind, layer]['records']
                if r['case'] == case and r['operation'] == operation
                and r['alpha'] == alpha)


def mask(
    kind: str,
    layer: int,
    case: str,
    operation: str = 'meet',
    alpha: float | None = 0.5,
) -> np.ndarray:
    """Reconstruct the Boolean extent mask for a query condition."""
    indices = ARRAYS[kind, layer][f'{case}_{operation}_{alpha}']
    result = np.zeros(25600, dtype=bool)
    result[indices] = True
    return result


def fmt(value: object) -> str:
    """Convert a table value to its textual representation."""
    return str(value)


def table(
    name: str,
    columns: str,
    header: Sequence[object],
    rows: Sequence[Sequence[object]],
    caption: str,
    label: str,
    placement: str = 'htbp',
) -> None:
    """Write a captioned LaTeX table to the experiment table directory."""
    lines = [r'\begin{table}[' + placement + ']', r'    \centering',
             r'    \SampleTableSetup', r'    \begin{tabular}{' + columns + '}',
             r'        \toprule']
    for row in [header, *rows]:
        line = ' & '.join(map(str, row)) + r' \\'
        lines.extend('        ' + line for line in textwrap.wrap(
            line, width=69, break_long_words=False, break_on_hyphens=False,
        ))
        if row is header:
            lines.append(r'        \midrule')
    lines += [r'        \bottomrule', r'    \end{tabular}', r'    \caption{']
    lines += ['        ' + line for line in textwrap.wrap(
        caption, width=70, break_long_words=False, break_on_hyphens=False,
    )]
    lines += [r'    }\label{' + label + '}', r'\end{table}', '']
    result = '\n'.join(lines)
    assert max(map(len, result.splitlines())) <= 79, name
    from lattmc.latex.naming_codexgen import annotate, wrap
    (TABLES / f'{name}.tex').write_text(wrap(annotate(result)))


cases = REPORTS['sae', 0]['cases']
table(
    'graded-source-cases', 'llrr',
    ['Case', 'Source positions', 'Row', '$|L|$'],
    [[NAMES[c['name']], ', '.join(map(str, c['positions'])),
      c['row'], c['lexical_count']] for c in cases],
    'Fixed source occurrences and lexical-reference sizes. Row and position '
    'indices are zero-based and include the initial special token. '
    '$L$ is the literal reference set defined in '
    r'\cref{sub:graded-measurements}; $N=25{,}600$ for every case.',
    'tab:graded-source-cases',
)
header = ['Model', r'$\ell$', '$k$', '$n_{+}$', '$n_{0.25}$', '$n_{0.5}$',
          '$n_{0.75}$', '$n_{1}$']
for case in ['nyc', 'animals']:
    rows = []
    for kind in ['sae', 'tc']:
        for layer in [0, 8, 11]:
            base = record(kind, layer, case, 'meet', None)
            rows.append([kind.upper(), layer, base['active_coordinates'],
                         base['count'], *[record(
                             kind, layer, case, 'meet', a,
                         )['count'] for a in ALPHAS]])
    table(
        f'graded-{case}', 'lrrrrrrr', header, rows,
        f'Full-code meet retrieval for {NAMES[case]}. '
        '$k$ counts positive query coordinates, $n_{+}$ is the binary '
        r'positive-support baseline, and $n_{\alpha}$ counts all rows '
        r'dominating $\alpha m_{H}$. Every count includes the source row. '
        'All activation levels use the same coordinate pattern.',
        f'tab:graded-{case}',
    )

cross_rows, cross_data = [], []
for case in ['nyc', 'animals']:
    for layer in [0, 8, 11]:
        c = coincidence(mask('sae', layer, case), mask('tc', layer, case))
        cross_data.append(dict(case=case, layer=layer, **c))
        cross_rows.append([SHORT[case], layer, c['both'], c['only_left'],
                           c['only_right'], c['neither'],
                           f"{c['jaccard']:.4f}"])
table(
    'graded-coincidence', 'lrrrrrr',
    ['Case', r'$\ell$', '$n_{11}$', '$n_{10}$', '$n_{01}$', '$n_{00}$',
     r'$\mathcal{J}$'], cross_rows,
    r'Cross-model coincidence for full-code meets at $\alpha=0.5$. '
    '$A$ is the SAE extent and $B$ the transcoder extent: columns are '
    'both, SAE only, transcoder only, and neither. Each row sums to '
    '$25{,}600$ before the Jaccard column. The common source is included.',
    'tab:graded-coincidence',
)
lex_rows = []
for case in ['nyc', 'animals']:
    for kind in ['sae', 'tc']:
        for layer in [0, 8, 11]:
            r = record(kind, layer, case, 'meet', 0.5)
            c = r['lexical']
            lex_rows.append([
                SHORT[case], kind.upper(), layer, c['both'], c['only_left'],
                c['only_right'], c['neither'],
                f"{100 * r['lexical_hit_fraction']:.2f}",
                f"{100 * r['lexical_coverage']:.1f}",
            ])
table(
    'graded-lexical', 'llrrrrrrr',
    ['Case', 'Model', r'$\ell$', '$n_{11}$', '$n_{10}$', '$n_{01}$',
     '$n_{00}$', r'$h\ (\%)$', r'$c\ (\%)$'], lex_rows,
    r'Lexical coincidence for full-code meets at $\alpha=0.5$. '
    'Here $A$ is the retrieved extent and $B=L$ is the literal reference: '
    'both, retrieved only, lexical only, and neither. The final columns '
    'give lexical hit fraction and lexical coverage from '
    r'\cref{eq:lexical-fractions}. These are not semantic accuracy scores.',
    'tab:graded-lexical',
)
for operation in ['meet', 'join']:
    rows = []
    for case in CASES:
        for kind in ['sae', 'tc']:
            for layer in [0, 8, 11]:
                b = record(kind, layer, case, operation, None)
                rows.append([
                    SHORT[case], kind.upper(), layer,
                    b['active_coordinates'], b['count'],
                    *[record(kind, layer, case, operation, a)['count']
                      for a in ALPHAS],
                ])
    table(
        f'graded-all-{operation}s', 'llrrrrrrr', ['Case', *header], rows,
        f'Complete full-vector {operation} counts for all six cases. '
        '$k$ is the number of active coordinates; $n_{+}$ is the '
        'positive-support count. The four remaining counts impose '
        'the source-derived amplitudes at the stated levels. '
        'A count of one includes only the source row.',
        f'tab:graded-all-{operation}s', placement='p',
    )
pair_rows = []
for kind in ['sae', 'tc']:
    for layer in [0, 8, 11]:
        b = record(kind, layer, 'rio', 'pair_meet', None)
        pair_rows.append([kind.upper(), layer, b['active_coordinates'],
                          b['count'], *[record(
                              kind, layer, 'rio', 'pair_meet', a,
                          )['count'] for a in ALPHAS]])
table('graded-rio-pair', 'lrrrrrrr', header, pair_rows,
      'Rio/de pairwise full-code meet, using row 5411 and positions 15 and '
      '16. Unlike the historical comparison, both models use identical '
      'query rules. The three-token results appear in '
      r'\cref{tab:graded-all-meets}.', 'tab:graded-rio-pair')

closure_rows = []
for case in ['nyc', 'animals']:
    for kind in ['sae', 'tc']:
        for layer in [0, 8, 11]:
            r = record(kind, layer, case, 'meet', 0.5)
            closure_rows.append([
                SHORT[case], kind.upper(), layer, r['count'],
                r['active_coordinates'], r['closed_active_coordinates'],
                r['closure_added_coordinates'],
            ])
table('graded-closures', 'llrrrrr',
      ['Case', 'Model', r'$\ell$', '$|A|$', '$k$', r'$k_{\mathrm{closed}}$',
       'Added'], closure_rows,
      r'Full-extent sequence-level closure at $\alpha=0.5$ for the main '
      'meet queries. Added counts newly positive intent coordinates, '
      'not all increased amplitudes. Every row satisfies '
      '$G(F(A))=A$; closure therefore preserves the reported extent.',
      'tab:graded-closures')

layer_rows = []
for case in ['nyc', 'animals']:
    for kind in ['sae', 'tc']:
        for first, second in [(0, 8), (8, 11)]:
            c = coincidence(mask(kind, first, case), mask(kind, second, case))
            layer_rows.append([
                SHORT[case], kind.upper(), f'{first}/{second}',
                c['both'], c['only_left'], c['only_right'], c['neither'],
                f"{c['jaccard']:.4f}",
            ])
table('graded-layer-coincidence', 'lllrrrrr',
      ['Case', 'Model', 'Layers', '$n_{11}$', '$n_{10}$', '$n_{01}$',
       '$n_{00}$', r'$\mathcal{J}$'], layer_rows,
      'Within-model coincidence of full-code meet extents at '
      r'$\alpha=0.5$. $A$ uses the first layer and $B$ the second. '
      'The table compares corpus rows, not latent coordinate identities.',
      'tab:graded-layer-coincidence')

families = []
for (kind, layer), report in REPORTS.items():
    for r in report['records']:
        if r['alpha'] != 1:
            continue
        b = record(kind, layer, r['case'], r['operation'], None)
        families.append(dict(
            kind=kind, layer=layer, case=r['case'], operation=r['operation'],
            zero_query=r['active_coordinates'] == 0,
            shrinks=r['count'] < b['count'],
            count=r['count'], baseline=b['count'],
            retained=r['count'] / b['count'],
        ))
summary = dict(
    evaluated_extents=sum(len(r['records']) for r in REPORTS.values()),
    query_families=len(families),
    zero_families=sum(f['zero_query'] for f in families),
    shrinking_families=sum(f['shrinks'] for f in families),
    singleton_exact_families=sum(f['count'] == 1 for f in families),
    cross_model=cross_data,
)
(RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
