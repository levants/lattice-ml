"""Generate closure-added coordinate tables from completed exact records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lattmc.contextstudy.methodtables_codexgen import table
from lattmc.contextstudy.semantictables_codexgen import pair_table
from lattmc.contextstudy.operations_codexgen import Record, save


def load(out: Path, name: str) -> Record:
    """Read one completed experiment record without running inference."""
    return json.loads((out / name).read_text())


def generate(out: Path, paper: Path) -> None:
    """Render exact extent counts, representative traces and replay limits."""
    tables = paper / 'tables'
    selected = [
        ('tc11', 'research_three', 'full', 'meet', 'Research full meet'),
        ('tc11', 'mixed_pair', 'rank1', 'join', 'Virus/Clippers rank-1 join'),
        ('tc11', 'biological_four', 'rank1', 'join', 'Biological rank-1 join'),
        ('tc11', 'cat_dog', 'full', 'meet', 'Cat/dog full meet'),
        ('tc11', 'cat_dog', 'rank23', 'join', 'Cat/dog rank-2/3 join'),
        ('sae11', 'new_de', 'rank23', 'join', 'New/de rank-2/3 join'),
    ]
    rows, cases = [], []
    for condition, group, rule, op, title in selected:
        data = load(out, f'gpt2_{condition}.json')
        r = next(r for r in data['records'] if r['group'] == group
                 and r['rule'] == rule and r['operation'] == op)
        cases.append(dict(condition=condition, title=title, **r))
        rows.append(title + ' & ' +
                    ('SAE' if condition.startswith('sae') else 'TC') +
                    ' & ' + ' & '.join(str(r[k]) for k in
                    ('original_positive', 'added_positive', 'original_count',
                     'h_count', 'additional_count')) + r' \\')
    table(tables / 'closure-added-cases.tex', 'tab:closure-added-cases',
          'Extents selected by coordinates introduced by closure',
          r'>{\raggedright\arraybackslash}Xlrrrrr',
          r'Original query $u$ & Surrogate & $|\mathrm{supp}(u)|$ & '
          r'$|\mathrm{supp}(h)|$ & $|G(u)|$ & $|G(h)|$ & Added rows', rows,
          'GPT-2-small, block 11: ReLU SAE at residual-pre; ReLU transcoder '
          '(TC) from normalized MLP input to MLP output. h keeps only '
          'newly positive coordinates of FG(u), at their unchanged closure '
          'amplitudes. Added rows means G(h) minus G(u), relative to '
          '25,600 cached rows including BOS. Sources and ranks are those '
          'of the preceding contextual-reading study. Counts use exact '
          'cached inequalities. All six TC sports-join members fail h on '
          'fresh replay by small numerical shortfalls; this is reported '
          'separately, without changing the cached counts or thresholds.')
    rows, summary = [], {}
    for kind in ('sae', 'tc'):
        for block in (0, 8, 11):
            name = f'{kind}{block}'
            records = load(out, f'gpt2_{name}.json')['records']
            proper = [r for r in records if 0 < r['original_count'] < 25600]
            values = dict(
                proper=len(proper), same=sum(r['same_extent'] for r in proper),
                expanded=sum(r['additional_count'] > 0 for r in proper),
                became_universal=sum(r['h_count'] == 25600 for r in proper),
                original_universal=sum(r['original_count'] == 25600
                                       for r in records),
                original_empty=sum(r['empty_original'] for r in records),
                zero_h=sum(r['zero_h'] for r in records))
            summary[name] = values
            rows.append(kind.upper() + f' & {block} & ' + ' & '.join(
                str(values[k]) for k in ('proper', 'same', 'expanded',
                'became_universal', 'original_universal', 'original_empty'))
                + r' \\')
    table(tables / 'closure-added-grid.tex', 'tab:closure-added-grid',
          'Closure-added extents across six GPT-2 dictionaries',
          'llrrrrrr', 'Surrogate & Block & Proper & Same & Expanded & '
          'To all & Already all & Empty', rows,
          'Each row contains 110 queries: 55 meets and 55 joins. Proper '
          'means $0<|G(u)|<25{,}600$; Same and Expanded partition those '
          'queries. To all is the subset of expansions reaching the whole '
          'corpus. Already all and Empty count original extents, kept '
          'outside the informative comparison. All 190 empty cases also '
          'have empty G(h); they use the conventional empty meet (top). '
          'No h is zero. Repeated descriptions and universal baselines '
          'are retained; these are query counts, not independent trials. '
          'SAEs read residual-pre; TCs read normalized MLP input and '
          'approximate MLP output. Block indices are zero-based.')
    data = load(out, 'tc11_witnesses.json')['records']
    chosen = [next(r for r in data if r['row'] == row and
                   r['query_name'] == 'h') for row in (43, 5872)]
    pair_table(paper, 'closure-added-texts',
        'Research evidence and organizational reporting after projection',
        'Biotransformation studies', 'Entertainment-company appointments',
        chosen, 'GPT-2-small / ReLU transcoder, block 11, normalized MLP '
        'input to MLP output. The experiments/measurements/analysis meet '
        'uses 3457@49, 5411@94, and 5411@115. Its added-coordinate '
        'description h has 190 positive requirements and 661 cached '
        'members; these rows are among the 562 additions to G(u). '
        'Both also satisfy h on exact replay but fail respectively two '
        'and six original requirements. Rose marks tokens satisfying any '
        'positive h requirement; bold rose would mark a whole-query token. '
        'Yellow D means distributed satisfaction; no individual token '
        'satisfies all 190 requirements. Full original 128-token rows '
        'are shown. The topical difference is an interpretation, not '
        'a relevance label or a fitted selection criterion.')
    rows = []
    for r in data:
        if not r['replay_changed']:
            continue
        q = np.array(r['query']['values'])
        maxima = np.array(r['maxima'])
        rows.append(f'{r["row"]} & {len(r["failed_coordinates"])} & '
                    f'{(q-maxima).max():.3g}' + r' \\')
    table(tables / 'closure-added-replay.tex', 'tab:closure-added-replay',
          'Exact closure thresholds and replay sensitivity', 'lrr',
          'Row & Failed requirements & Largest shortfall', rows,
          'GPT-2-small / ReLU transcoder, block 11. The 1,076-coordinate '
          'h from the virus/Clippers rank-1 join selects all six listed '
          'rows in the original cache. All fail h in fresh CPU replay; '
          'the original two-coordinate u still passes in every row. '
          'Failed requirements counts replayed tokenwise maxima below '
          'their cached closure minima. Largest shortfall is the largest '
          'positive difference between threshold and replay maximum. '
          'No epsilon or relaxed amplitude is used to repair membership. '
          'These are R on replay, not verified distributed h witnesses.')
    save(out / 'summary.json', dict(conditions=summary, cases=cases))


def main() -> None:
    """Regenerate all four tables without rerunning model inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    args = parser.parse_args()
    generate(args.out, args.paper)


if __name__ == '__main__':
    main()
