"""Render exploratory semantic evidence with the shared witness conventions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lattmc.contextstudy.methodtables_codexgen import table, write_tex
from lattmc.contextstudy.witnessrender_codexgen import snippet
from lattmc.latex.texttables_codexgen import paired, tex
from lattmc.contextstudy.operations_codexgen import Record, save


def load(out: Path, name: str) -> Record:
    """Read one completed study record."""
    return json.loads((out / name).read_text())


def entry(record: Record, short: bool = False) -> str:
    """Render original token pieces using measured witness positions."""
    row = dict(record)
    row['valid_positions'] = row.get('valid', list(range(128)))
    row['diagnostic'] = (row['partial'] or row['valid_positions'])[0]
    status = row['status']
    badge = r'\DistributedMark{D}' if status == 'D' else status
    return (r'\item ' + badge + f', row {row["row"]}: ``'
            + snippet(row, short=short) + "''")


def pair_table(paper: Path, name: str, title: str, left: str, right: str,
               rows: list[Record], note: str, short: bool = False) -> None:
    """Write one paired example with a short title and full local metadata."""
    content = paired([(left, right, [entry(rows[0], short)],
                       [entry(rows[1], short)])], title + '. ' + note,
                     f'tab:{name}')
    content = [line.replace(r'\caption{', r'\caption[' + title + ']{')
               for line in content]
    write_tex(paper / 'tables' / f'{name}.tex', content)


def witness(out: Path, condition: str, group: str, rule: str,
            operation: str, row: int) -> Record:
    """Find the unique saved witness for a fully specified query and row."""
    records = load(out, f'{condition}_witnesses.json')['records']
    return next(r for r in records if r['group'] == group and
                r['rule'] == rule and r['operation'] == operation and
                r['row'] == row)


def generate(out: Path, paper: Path) -> None:
    """Create tables and summaries from verified records, without inference."""
    tables = paper / 'tables'
    legend = ('Rose marks every displayed token satisfying at least one '
              'positive requirement; bold rose marks a whole-query token. '
              'Yellow D means distributed item satisfaction, not a semantic '
              'judgment. All comparisons use unrounded amplitudes and no '
              'epsilon. Row and token positions are zero-based. ')
    model = ('GPT-2-small / ReLU transcoder, block 11; '
             'normalized MLP input to MLP output. ')
    a = witness(out, 'tc11', 'research_three', 'full', 'meet', 16621)
    b = witness(out, 'tc11', 'mixed_pair', 'rank1', 'join', 9155)
    pair_table(paper, 'reading-contexts',
        'Shared research evidence and combined contextual roles',
        r'Research meet: $G(u\wedge v\wedge w)$',
        r'Component join: $G(p\vee q)$', [a, b], model +
        'Left: the experiments/measurements/analysis meet has 18 positive '
        'coordinates, 99 members and 97 additional members beyond its '
        'constituents. Right: rank-1 virus/Clippers components require '
        '$4355:11.57799$ and $13040:23.46306$, yielding 6 members from '
        'constituent extents of 537 and 198. ' + legend +
        'Displayed windows have at most 64 tokens; full rows and complete '
        'witness ledgers are retained. These examples were chosen to '
        'explain observed patterns, not estimate semantic precision.', True)
    rows = []
    summary = {'conditions': {}, 'coincidence': []}
    for kind in ('sae', 'tc'):
        sets = []
        for block in (0, 8, 11):
            data = load(out, f'gpt2_{kind}{block}.json')
            r = next(r for r in data['records'] if r['group'] ==
                     'research_three' and r['rule'] == 'full')
            m = r['meet']
            rows.append(f'{kind.upper()} & {block} & {m["positive"]} & '
                f'{m["count"]} & {r["extra_count"]} & '
                f'{m["closed_positive"]} \\\\')
            arrays = np.load(out / f'gpt2_{kind}{block}_members.npz')
            sets.append(set(arrays[m['array_key']].tolist()))
            summary['conditions'][f'{kind}{block}'] = dict(
                cases=len(data['records']),
                zero_meets=sum(r['meet']['positive'] == 0
                               for r in data['records']),
                empty_joins=sum(r['join']['count'] == 0
                                for r in data['records']),
                universal_closure=data['baseline']['zero']['closed_positive'])
        for i, j in ((0, 1), (1, 2)):
            a, b = sets[i], sets[j]
            summary['coincidence'].append(dict(
                surrogate=kind, blocks=[[0, 8, 11][i], [0, 8, 11][j]],
                both=len(a & b), lost=len(a - b), added=len(b - a),
                neither=25600-len(a | b), jaccard=len(a & b)/len(a | b)))
    table(tables / 'reading-depth.tex', 'tab:reading-depth',
          'Research descriptions across GPT-2 blocks', 'llrrrr',
          r'Surrogate & Block & $|\mathrm{supp}(u)|$ & $|G(u)|$ & '
          r'$|E|$ & $|\mathrm{supp}(FG(u))|$', rows,
          'The full meet uses experiments (3457@49), measurements '
          '(5411@94), and analysis (5411@115). Each dictionary is separate. '
          'SAEs read residual-pre; transcoders read normalized MLP input '
          'and approximate MLP output. E excludes the union of all three '
          'constituent extents. Closure strengthens the description but '
          'preserves its extent exactly. The full join is empty in all '
          'six conditions. Counts use all 25,600 cached rows.')
    rows = []
    for r in summary['coincidence']:
        rows.append(r['surrogate'].upper() + ' & $' +
                    str(r['blocks'][0]) + r'\to ' + str(r['blocks'][1]) +
                    '$ & ' + ' & '.join(str(r[k]) for k in
                    ('both', 'lost', 'added', 'neither')) +
                    f' & {r["jaccard"]:.3f} \\\\')
    table(tables / 'reading-coincidence.tex', 'tab:reading-coincidence',
          'Turnover of shared research extents', 'llrrrrr',
          'Surrogate & Blocks & Both & Earlier only & Later only & '
          'Neither & Jaccard', rows,
          'Rows compare the same three source occurrences in separate '
          'dictionaries. Both is the intersection; Earlier only and Later '
          'only are directional differences. Neither is relative to all '
          '25,600 rows. Jaccard divides intersection by union. These are '
          'description comparisons, not transported queries or causal '
          'trajectories.')
    x = load(out, 'gpt2_tc11.json')
    r = next(r for r in x['records'] if r['group'] == 'biological_four'
             and r['rule'] == 'rank1')
    arrays = np.load(out / 'gpt2_tc11_members.npz')
    sets = [set(arrays[c['array_key']]) for c in r['components']]
    rows = []
    for i, (name, c) in enumerate(zip(r['sources'], r['components'])):
        n = len(set.intersection(*(s for j, s in enumerate(sets) if j != i)))
        q = c['query']
        rows.append(tex(name) + f' & {q["coordinates"][0]} & '
            f'{q["values"][0]:.5f} & {c["count"]} & {n} & '
            f'{100 * 3 / c["count"]:.2f} \\\\')
    table(tables / 'reading-components.tex', 'tab:reading-components',
          'Individual contributions to a four-component biological join',
          'lrrrrr', 'Occurrence & Coordinate & Amplitude & Constituent & '
          r'Without component & Retained (\%)', rows,
          model + 'Rank 1 from monkeys (3457@55), mice (3457@57), '
          'specimens (5411@31), and species (5411@74). The joint extent '
          'has three rows. Without component counts the intersection of '
          'the other three extents; retaining three means redundancy. '
          'Retained divides three by the individual constituent count. '
          'The meet of these four distinct single-coordinate vectors is '
          'zero, hence has all 25,600 members. Amplitudes are rounded '
          'only for display.')
    a = witness(out, 'tc11', 'biological_four', 'rank1', 'join', 11388)
    b = witness(out, 'tc11', 'biological_four', 'rank1', 'join', 23609)
    pair_table(paper, 'reading-biological',
        'Biological evidence and a spliced item in the same joined extent',
        'Tissue development', 'Longevity research followed by commentary',
        [a, b], model + r'The four requirements in '
        r'\cref{tab:reading-components} return rows 11388, 23609, and 24255; '
        'all three have distributed witnesses. Full 128-token rows are '
        'shown here to expose the second item\'s topic boundary. ' + legend)
    a = witness(out, 'tc11', 'cat_dog', 'full', 'meet', 12087)
    b = witness(out, 'tc11', 'cat_dog', 'full', 'meet', 17545)
    pair_table(paper, 'reading-ambiguity',
        'Entertainment associations and an unresolved Cat/dog meet member',
        'Film catalogue', 'Literary passage and political splice', [a, b],
        model + 'The full Cat/dog meet uses row 4042 at positions 8 and 82 '
        'and has 37 positive coordinates and 25 members; 24 are additional '
        'to its one-row constituent union. Cat occurs in a film title; '
        'dog occurs after a visible text splice. The right-hand relation '
        'remains unresolved despite exact numerical membership. ' + legend)
    a = witness(out, 'sae11', 'new_de', 'rank23', 'join', 7799)
    b = witness(out, 'sae11', 'new_de', 'rank23', 'join', 21747)
    pair_table(paper, 'reading-newde',
        'Coherent and spliced witnesses of the New/de component join',
        'Boxing promotion and personal names', 'Geopolitics and film news',
        [a, b], 'GPT-2-small / ReLU SAE, block 11, residual-pre. '
        'New rank 2 (3457@1) and de rank 3 (5411@16) require '
        '$7275:26.57727$ and $22858:12.59291$. Constituent extents contain '
        '489 and 418 rows; their join has 13, all distributed. Every '
        'member retains a literal New witness. These full-row examples '
        'contrast within-topic and visibly spliced co-occurrence. ' + legend)
    data = load(out, 'prefix_contrast.json')
    rows = []
    for r in data['records']:
        rows.append(f'{r["row"]}@{r["target"]} & '
            f'{r["original"]:.3f} & ' + tex(r['after'].strip()) +
            ' & ' + tex(r['family_replacement'].strip()) +
            f' & {r["family_activation"]:.3f} \\\\')
    table(tables / 'reading-prefix.tex', 'tab:reading-prefix',
          'Verb substitutions at a contextual component', 'lrl lr'.replace(
              ' ', ''),
          'Row@target & Original & Other verb & Family verb & Family value',
          rows, model + 'Coordinate 4355 at the same target token. '
          'Other-verb replacements give exactly zero in all seven rows. '
          'Family replacements preserve positive activation, but only '
          'putting in row 9486 exceeds the original threshold 11.57799. '
          'Each edit replaces one earlier token and preserves length. '
          'Suffix controls preserve the complete target code exactly. '
          'Replacements and examples were chosen after observing the '
          'pattern; this is input sensitivity, not latent ablation or '
          'downstream behavioral validation.')
    data = load(out, 'structure_probe.json')
    chosen = []
    for family, row in [('pythia_topk', 1723), ('smol_topk', 1810)]:
        record = dict(next(r for r in data['records'] if
                           r['family'] == family and r['row'] == row))
        record.update(partial=record['whole'], status='S')
        chosen.append(record)
    pair_table(paper, 'reading-structure',
        'Structural relationships beyond the source category',
        'Pythia-70M / TopK SAE; block 3, residual-post',
        'SmolLM2-135M / TopK SAE; block 15, MLP output', chosen,
        'The historical source summaries are DBpedia rows 709 and 743. '
        'Their rank-1 joins require 1512:57.30409 in Pythia and '
        '3328:31.25120 in SmolLM2. The left witness is the abbreviation '
        'period within St., not the title separator; the right witnesses '
        'are a space and digits in 2010, not a proper-name fragment. '
        'They qualify hypotheses formed from other sampled members. '
        'The probe sampled 16 members and 16 nonmembers per model before '
        'its label comparison. BOS/padding are excluded. ' + legend)
    save(out / 'summary.json', summary)


def main() -> None:
    """Regenerate the paper tables from completed local study records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    args = parser.parse_args()
    generate(args.out, args.paper)


if __name__ == '__main__':
    main()
