"""Render complete query witnesses consistently in TeX and notebooks."""

from __future__ import annotations
from typing import Any

import html
import json
import os
from pathlib import Path

from lattmc.latex.texttables_codexgen import paired, styles, tex, write, HEX
from .witnesses_codexgen import DEST, ROOT

from lattmc.latex.naming_codexgen import NAMES, panel
CASES = dict(nyc='NYC', rio='Rio', animals='Cat/dog', sports='Sports',
             **{'7': 'NaturalPlace', '9': 'Animal'})
LEGEND = (r'Yellow D marks distributed satisfaction; S means a single '
          r'token satisfies the entire query, and R means rejection. Rose '
          r'marks every displayed token meeting at least one coordinate; '
          r'bold rose marks whole-query tokens. These are activation '
          r'predicates, not semantic judgments. Requirements use the same '
          r'$\alpha u$ for both classifications. Token positions, complete '
          r'coordinate requirements, and maxima are in '
          r'\appcref{app:witness-inspection}.')


def records() -> list[dict[str, Any]]:
    """Load the combined witness ledger used by all rendered galleries."""
    rows = json.loads((DEST / 'records.json').read_text())['records']
    for i, row in enumerate(rows, 1):
        row['display_id'] = f'W{i:03d}'
    return rows


def bounds(row: dict[str, Any], short: bool) -> tuple[int, int]:
    """Choose the visible token interval for a witness excerpt."""
    valid = row['valid_positions']
    if not short:
        return min(valid), max(valid) + 1
    # Same window policy for every model; never select lexical matches.
    lo = max(min(valid), row['diagnostic'] - 32)
    hi = min(max(valid) + 1, lo + 64)
    return max(min(valid), hi - 64), hi


def snippet(
    row: dict[str, Any],
    short: bool = False,
    as_html: bool = False,
) -> str:
    """Render a witness excerpt with token highlights in LaTeX or HTML."""
    lo, hi = bounds(row, short)
    partial, whole = set(row['partial']), set(row['whole'])
    pieces = []
    for p in range(lo, hi):
        text = row['pieces'][p]
        if as_html:
            value = html.escape(text)
            if p in partial:
                value = f'<mark style="background:#{HEX}">{value}</mark>'
            if p in whole and row['status'] != 'N':
                value = '<b>' + value + '</b>'
            token_id = row['tokens'][p]
            pieces.append(f'<span title="position {p}; token {token_id}">'
                          + value + '</span>')
        else:
            value = tex(text)
            leading = r'\ ' if value.startswith(' ') else ''
            trailing = r'\ ' if value.endswith(' ') and value.strip() else ''
            value = value.strip()
            if p in partial:
                value = value or r'\textvisiblespace{}'
                if p in whole:
                    value = r'\textbf{' + value + '}'
                value = r'\TokenHighlight{' + value + '}'
            pieces.append(leading + value + trailing)
    before = '... ' if lo > min(row['valid_positions']) else ''
    after = ' ...' if hi <= max(row['valid_positions']) else ''
    # TeX comments retain the measured adjacency of subword pieces.
    if as_html:
        return before + ''.join(pieces) + after
    lines, current = [], before
    for piece in pieces:
        if len(current + piece) > 74:
            lines.append(current + '%')
            current = ''
        current += piece
    lines.append(current + after)
    return '\n'.join(lines)


def metadata(row: dict[str, Any]) -> str:
    """Describe the query, source, and witness status of a gallery row."""
    source = ','.join(map(str, row['sources']))
    if row['source_kind'] == 'token':
        source += '@(' + ','.join(map(str, row['source_positions'])) + ')'
    identity = str(row['row'])
    if 'dataset_identity' in row:
        d = row['dataset_identity']
        identity += f" ({d['source_split']}:{d['source_id']})"
    default = 'selected' if row['source_kind'] == 'document' else 'full'
    rule = row.get('mode', default)
    case = CASES[str(row['case'])]
    return (f"block {row['layer']}; {row['source_kind']} {rule} meet "
            f"({case}): {source}; "
            f"item {identity}; whole-token witnesses: {len(row['whole'])}; "
            f"$d={len(row['query'])}$, $\\alpha={row['alpha']:g}$; "
            f"$s={row['score']:.3f}$; "
            + ('member' if row['member'] else 'nonmember'))


def entry(row: dict[str, Any], short: bool = False) -> str:
    """Format a labeled LaTeX witness item with its metadata."""
    status = row['status']
    badge = (r'\DistributedMark{D}' if status == 'D' else
             r'\textbf{' + status + '}')
    return (r'\item ' + badge + ' ' + row['display_id'] + ' ``' +
            snippet(row, short) + "''" + r'\par' + "\n" +
            r'\emph{' + metadata(row) + '}')


def html_gallery(rows: list[dict[str, Any]] | None = None) -> str:
    """Build the HTML witness gallery from the selected ledger records."""
    rows = records() if rows is None else rows
    output = ['<p><b>S</b>: single-token; '
              '<mark style="background:#FFEF9F">D: distributed</mark>; '
              '<b>R</b>: rejected. Rose: coordinate witness; bold: whole '
              'query. Yellow is not a semantic correctness label.</p>']
    for row in rows:
        badge = row['status']
        if badge == 'D':
            badge = '<mark style="background:#FFEF9F">D</mark>'
        heading = panel(row['family'], row['layer'])
        heading = heading.replace(r'\newline ', '; ')
        output += [f'<h4>{row["display_id"]}: {heading}</h4>',
                   '<p>' + badge + ' ' + snippet(row, as_html=True) + '</p>',
                   '<p>' + html.escape(metadata(row)) + '</p>',
                   '<details><summary>Original decoded text</summary>'
                   + html.escape(row['decoded_text']) + '</details>',
                   '<details><summary>All coordinate witnesses</summary>',
                   '<table><tr><th>Coordinate</th><th>Requirement</th>'
                   '<th>Maximal token</th><th>Activation</th>'
                   '<th>Pass</th></tr>']
        for j, w in zip(row['coordinates'], row['witnesses']):
            output.append(f'<tr><td>{j}</td><td>{w["requirement"]:.9g}</td>'
                          f'<td>{w["position"]}</td>'
                          f'<td>{w["value"]:.9g}</td>'
                          f'<td>{w["meets"]}</td></tr>')
        output += ['</table></details>']
    return '\n'.join(output)


def families(paper: Path) -> None:
    """Write paired cross-family witness tables for the paper."""
    rows = [r for r in records() if r['group'] == 'family']
    panels = []
    for left, right, case in [('gemma_matryoshka', 'smol_topk', 7),
                               ('pythia_topk', 'qwen_transcoder', 9)]:
        selected = [next(r for r in rows if r['family'] == f and
                         r['case'] == case and r['rank'] == 1)
                    for f in (left, right)]
        panels.append((panel(left), panel(right),
                       [entry(selected[0], True)], [entry(selected[1], True)]))
    write(paper / 'tables/depth-highlighted.tex', paired(panels,
        'The same four rank-one DBpedia examples, with full-query witness '
        'classification and 64-token excerpts. ' + LEGEND,
        'tab:depth-highlighted'))
    for case, stem in [(7, 'geographic'), (9, 'biological')]:
        lines = []
        for k, (left, right) in enumerate([
                ('gemma_matryoshka', 'pythia_topk'),
                ('smol_topk', 'qwen_transcoder')]):
            cells = [[entry(r) for r in rows if r['family'] == f
                      and r['case'] == case] for f in (left, right)]
            label = 'tab:depth-gallery-' + stem + ('-continued' if k else '')
            lines += paired([(panel(left), panel(right), *cells)],
                f'Complete {CASES[str(case)]} gallery, panel {k+1}: '
                'all three ranked rows and all valid text positions per '
                'checkpoint. Mismatches and rejected rows are retained. '
                + LEGEND, label)
        write(paper / f'tables/depth-gallery-{stem}.tex', lines)


def contexts(paper: Path) -> None:
    """Write contextual-replay witness tables for the paper."""
    rows = [r for r in records() if r['group'] == 'context']
    panels = []
    for layer, case, item in [(11, 'rio', 3118), (11, 'animals', 2311),
                              (8, 'animals', 14155)]:
        left = [r for r in rows if r['family'] == 'tc' and
                r['layer'] == layer and r['case'] == case]
        right = [r for r in rows if r['family'] == 'sae' and
                 r['layer'] == layer and r['case'] == case]
        i = next(i for i, r in enumerate(left) if r['row'] == item)
        panels.append((panel('tc', layer), panel('sae', layer),
                       [entry(left[i], True)], [entry(right[i], True)]))
    write(paper / 'tables/context-highlighted.tex', paired(panels,
        'Three previously selected transcoder examples and the SAE rows '
        'at the same within-condition sample ordinal. Each excerpt uses '
        'the same 64-token window policy. The complete sample retains all '
        '48 rows. ' + LEGEND, 'tab:context-highlighted'))
    lines = []
    for layer in (8, 11):
        for stem, cases in [('geographic', ('nyc', 'rio')),
                            ('other', ('animals', 'sports'))]:
            for k, case in enumerate(cases):
                cells = [[entry(r) for r in rows if r['family'] == f and
                          r['layer'] == layer and r['case'] == case]
                         for f in ('tc', 'sae')]
                label = f'tab:context-layer{layer}-{stem}'
                if k:
                    label += '-continued'
                table = paired([(panel('tc', layer),
                                  panel('sae', layer), *cells)],
                    f'Complete {CASES[case]} sample at block {layer}, '
                    'with all 128 cached positions inspected. ' + LEGEND,
                    label)
                name = f'context-layer{layer}-{case}'
                write(paper / 'tables' / (name + '.tex'), table)
                lines.append(r'\input{tables/' + name + '}')
    write(paper / 'tables/context-all.tex', lines)


def ledger(paper: Path) -> None:
    """Write the witness-status ledger and its explanation."""
    lines = [r'\begingroup\scriptsize',
             r'\begin{longtable}{llrrrrl}',
             r'\caption{Every required coordinate of the replayed '
             r'gallery items. '
             r'$q_{j}=\alpha u_{j}$; $p$ is its first maximal valid token '
             r'and $a=z_{p,j}$. Comparisons use unrounded stored values; '
             r'displayed numbers have six significant digits.}',
             r'\label{tab:witness-ledger}\\', r'\toprule',
             r'Item & Status & $j$ & $q_{j}$ & $p$ & $a$ & Pass \\',
             r'\midrule\endfirsthead', r'\toprule',
             r'Item & Status & $j$ & $q_{j}$ & $p$ & $a$ & Pass \\',
             r'\midrule\endhead', r'\bottomrule\endfoot']
    from .legacytables_codexgen import records as legacy_records
    rows = records() + [r for r in legacy_records() if r['status'] != 'U']
    for r in rows:
        for j, w in zip(r['coordinates'], r['witnesses']):
            lines += [f"{r['display_id']} & {r['status']} & {j} & "
                      f"{w['requirement']:.6g} & {w['position']} & "
                      f"{w['value']:.6g} & " +
                      ('yes' if w['meets'] else 'no') + r' \\']
    lines += [r'\end{longtable}', r'\endgroup']
    write(paper / 'tables/witness-ledger.tex', lines)


def main(paper: Path | None = None) -> None:
    """Generate the witness galleries, paper tables, and ledger."""
    paper = Path(paper or os.environ.get(
        'LATTCONFERENCE_PAPER', ROOT / 'texs/sparsesurrs/lattconference'))
    styles(paper)
    families(paper)
    contexts(paper)
    from .legacytables_codexgen import main as legacy_tables
    legacy_tables(paper)
    ledger(paper)


if __name__ == '__main__':
    main()
