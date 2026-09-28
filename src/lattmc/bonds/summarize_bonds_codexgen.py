"""Generate manuscript tables and check illustrated corpus memberships."""

import csv
import json
import re

import numpy as np

from .paths_codexgen import OUT, PAPER
report = json.loads((OUT / 'results.json').read_text())
rows = report['comparisons']
flat = [{k: v for k, v in row.items() if k != 'representation'}
        for row in rows]
with (OUT / 'comparisons.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)

header = r'''\begin{table}[htbp]
  \centering
  \small
  \begin{tabular}{lrrr}
    \toprule
    Direction & Graded failures & Support failures & Mean graded $J$ \\
    \midrule
'''
lines = []
summary = []
for left, right, label in [
    ('sae', 'sae', r'SAE $\to$ SAE'),
    ('tc', 'tc', r'TC $\to$ TC'),
    ('sae', 'tc', r'SAE $\to$ TC'),
    ('tc', 'sae', r'TC $\to$ SAE'),
]:
    selected = [
        r for r in rows if r['seed'] > 0
        and r['source'].startswith(left) and r['target'].startswith(right)
    ]
    graded = [r for r in selected if r['mode'] == 'graded']
    support = [r for r in selected if r['mode'] == 'support']
    gf = sum(r['direct_violation'] for r in graded)
    sf = sum(r['direct_violation'] for r in support)
    mean_j = float(np.mean([r['jaccard'] for r in graded]))
    lines.append(f'    {label} & {gf}/192 & {sf}/192 & {mean_j:.3f}'
                 + r' \\' + '\n')
    summary.append(dict(
        source=left, target=right, graded_failures=gf,
        support_failures=sf, mean_graded_jaccard=mean_j,
    ))
footer = r'''    \bottomrule
  \end{tabular}
  \caption{Direct-incidence failures for 32 random seeds. Each direction
    contains six ordered distinct-layer pairs and hence 192 comparisons
    per mode. TC denotes transcoder; $J$ is extent Jaccard similarity.}
  \label{tab:random-bonds}
\end{table}
'''
(PAPER / 'tables/random-bonds.tex').write_text(header+''.join(lines)+footer)

header = r'''\begin{table}[htbp]
  \centering
  \small
  \begin{tabular}{lrrrr}
    \toprule
    Context & \multicolumn{2}{c}{Notebook seed}
      & \multicolumn{2}{c}{Random-seed median} \\
    & Graded & Support & Graded & Support \\
    \midrule
'''
lines = []
for name in ('sae0', 'sae8', 'sae11', 'tc0', 'tc8', 'tc11'):
    selected = [r for r in report['closures'] if r['context'] == name]
    values = []
    for seed_type in ('notebook', 'random'):
        for mode in ('graded', 'support'):
            sizes = [r['size'] for r in selected if r['mode'] == mode
                     and (r['seed'] == 0 if seed_type == 'notebook'
                          else r['seed'] > 0)]
            values.append(float(np.median(sizes)))
    label = name.replace('sae', 'SAE ').replace('tc', 'TC ')
    numbers = [f'{v:,.1f}' if v % 1 else f'{v:,.0f}' for v in values]
    numbers = [v.replace(',', '{,}') for v in numbers]
    lines.append('    '+label+' & '+' & '.join(numbers)+r' \\'+'\n')
footer = r'''    \bottomrule
  \end{tabular}
  \caption{Extent sizes on the full 25{,}600-block corpus. Each seed has
    17 members. Random-seed medians summarize 32 shared seed groups;
    fractional medians average the two middle counts.}
  \label{tab:computed-extents}
\end{table}
'''
(PAPER / 'tables/computed-extents.tex').write_text(
    header+''.join(lines)+footer
)

checks = []
for file, left, right in [
    ('transcoder-layer-3-to-layer-9.tex', 'tc3', 'tc9'),
    ('transcoder-layer-1-to-sae-layer-11.tex', 'tc1', 'sae11'),
]:
    text = (PAPER / 'tables' / file).read_text()
    ids = [int(s) for s in re.findall(r'ID\s+(\d+)', text)]
    source = np.load(OUT / f'{left}_closures.npz')['graded_0_extent']
    target = np.load(OUT / f'{right}_closures.npz')['graded_0_extent']
    seed = report['protocol']['seeds'][0]
    common = set(source) & set(target) - set(seed)
    assert set(ids[:3]) <= set(seed)
    assert set(ids[3:]) <= common
    checks.append(dict(file=file, seed_ids=ids[:3], nonseed_ids=ids[3:]))

selected = [r for r in rows if r['seed'] == 0]
assert all(r['direct_violation'] for r in selected)
assert not any(r['trivial_bond'] for r in rows)
for expected, actual in zip(
    [(5394, 100, 30), (4974, 10428, 2081)], report['historical']
):
    assert expected == (
        actual['source_size'], actual['target_size'],
        actual['nonseed_common'],
    )
(OUT / 'summary.json').write_text(json.dumps(dict(
    families=summary, illustrated_memberships=checks,
    notebook_seed_violates_all_48=True, trivial_bonds=0,
    total_bonds=len(rows), historical_counts_reproduced=True,
), indent=2)+'\n')
print(json.dumps(summary, indent=2))
print('All table membership and historical-count checks passed.')
