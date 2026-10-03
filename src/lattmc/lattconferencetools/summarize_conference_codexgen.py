"""Generate tables and bootstrap summaries from the fixed retrieval runs."""

from __future__ import annotations
from collections.abc import Sequence

from collections import Counter
import json

from .paths_codexgen import RELEASE, PROTOCOL, ORGANIZATION
from pathlib import Path
import zipfile

import numpy as np

from .conference_experiments_codexgen import HERE, OUT, METHODS, sha256

NAMES = {
    'graded': 'Graded (selected)', 'support': 'Support (selected)',
    'matched_support': 'Support (matched)', 'full_graded': 'Graded (all)',
    'cosine': 'Activation cosine', 'tfidf': 'TF-IDF',
    'random_source': 'Random-source graded',
}


def interval(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    """Compute a mean and bootstrap interval from shared resampling indices."""
    replicates = values[indices].mean(axis=1)
    low, high = np.quantile(replicates, [.025, .975])
    return dict(mean=float(values.mean()), low=float(low), high=float(high))


def cell(value: dict[str, float], ci: bool = True) -> str:
    """Format a percentage with an optional confidence interval."""
    mean = f"{100 * value['mean']:.2f}"
    if not ci:
        return mean
    return (mean + f" [{100 * value['low']:.2f}, "
            f"{100 * value['high']:.2f}]")


def table(
    name: str,
    caption: str,
    label: str,
    columns: str,
    header: str,
    rows: Sequence[str],
) -> None:
    """Write a labeled summary table in LaTeX."""
    lines = [r'\begin{table}[tbp]', r'\centering', r'\small',
             r'\setlength{\tabcolsep}{4pt}',
             '\\caption{' + caption + '}', '\\label{' + label + '}',
             '\\begin{tabular}{' + columns + '}', r'\toprule',
             header + r' \\', r'\midrule']
    lines += [row + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    # Break source rows at a column separator; rendered layout is unchanged.
    wrapped = []
    for line in lines:
        while len(line) > 79:
            pos = line.rfind(' & ', 0, 75)
            if pos < 0:
                pos = line.rfind(' ', 0, 75)
            if pos < 0:
                raise ValueError(f'Unbreakable table line: {line}')
            wrapped.append(line[:pos])
            line = line[pos:].lstrip()
        wrapped.append(line)
    from lattmc.latex.naming_codexgen import annotate, wrap
    content = wrap(annotate('\n'.join(wrapped)))
    (HERE / 'tables' / name).write_text(content)


def verify_source_revision(recorded: str) -> None:
    """Accept the current source or the documented path-only relocation."""
    folder = Path(__file__).resolve().parent
    current = sha256(folder / 'conference_experiments_codexgen.py')
    if recorded == current:
        return
    record = ORGANIZATION / 'source_relocation_v2.json'
    relocation = json.loads(record.read_text())
    assert current == relocation['relocated_sha256']
    assert recorded in relocation['accepted_prior_sha256']
    print('Using archived results from the source before relocation.')


def main() -> None:
    """Generate tables and bootstrap summaries from the fixed retrieval runs.
    """
    design = json.loads((OUT / 'design.json').read_text())
    phrases = design['chosen']
    n = len(phrases)
    indices = np.random.default_rng(20260925).integers(0, n, (10000, n))
    summary, main_rows, coincidence_rows, selection_rows = {}, [], [], []
    for kind in ('sae', 'tc'):
        detail_rows = []
        for layer in (0, 8, 11):
            stem = f'{kind}_{layer}'
            report = json.loads((OUT / f'{stem}.json').read_text())
            verify_source_revision(report['source_sha256'])
            assert report['design_sha256'] == sha256(OUT / 'design.json')
            values = {}
            stats = {}
            for method in METHODS:
                values[method] = {}
                stats[method] = {}
                for metric in ('ap', 'p10'):
                    array = np.array([
                        np.mean([r['metrics'][method][metric]
                                 for r in report['runs'] if r['phrase'] == p])
                        for p in phrases
                    ])
                    values[method][metric] = array
                    stats[method][metric] = interval(array, indices)
                detail_rows.append(
                    f'{layer} & {NAMES[method]} & '
                    + cell(stats[method]['ap']) + ' & '
                    + cell(stats[method]['p10']))
            differences = {}
            for method in METHODS[1:]:
                delta = values['graded']['ap'] - values[method]['ap']
                differences[method] = interval(delta, indices)
            confusion = {
                key: sum(r['metrics']['graded'][key] for r in report['runs'])
                for key in ('tp', 'fp', 'fn', 'tn')
            }
            mean_f1 = float(np.mean([
                r['metrics']['graded']['f1'] for r in report['runs']]))
            budgets = Counter(str(r['selected_budget'])
                              for r in report['runs'])
            degeneracies = dict(
                zero_queries=sum(r['full_size'] == 0 for r in report['runs']),
                full_test_empty=sum(
                    r['metrics']['full_graded']['tp']
                    + r['metrics']['full_graded']['fp'] == 0
                    for r in report['runs']),
                selected_test_empty=sum(
                    r['metrics']['graded']['tp']
                    + r['metrics']['graded']['fp'] == 0
                    for r in report['runs']),
            )
            summary[stem] = dict(metrics=stats, paired_ap=differences,
                                 confusion=confusion, mean_f1=mean_f1,
                                 budgets=dict(budgets), **degeneracies)
            model = ('SAE' if kind == 'sae' else 'TC') + f' {layer}'
            main_rows.append(' & '.join([
                model, cell(stats['graded']['ap']),
                cell(stats['support']['ap'], False),
                cell(stats['cosine']['ap'], False),
                cell(stats['tfidf']['ap'], False),
            ]))
            coincidence_rows.append(' & '.join([
                model, *[f'{confusion[key]:,}'
                         for key in ('tp', 'fp', 'fn', 'tn')],
                f'{100 * mean_f1:.2f}',
            ]))
            selection_rows.append(' & '.join([
                model, *[str(budgets[str(k)]) for k in (1, 4, 16, 64, None)],
                str(degeneracies['full_test_empty']),
            ]))
        table(
            f'conference-{kind}-details.tex',
            ('Held-out retrieval for ' + ('SAEs' if kind == 'sae' else
             'transcoders') + '. Values are percentages; brackets give '
             r'95\% phrase-bootstrap intervals. Matched support uses the '
             'graded method\'s selected coordinates.'),
            f'tab:conference-{kind}-details', 'llrr',
            r'Layer & Method & AP [95\% interval] & P@10 [95\% interval]',
            detail_rows,
        )
    table(
        'conference-main.tex',
        ('Test average precision (percent) over 32 phrases, averaging five '
         r'source draws per phrase. Brackets show 95\% phrase-bootstrap '
         'intervals for graded retrieval. All method intervals and P@10 '
         'appear in \\appcref{app:retrieval-benchmark}.'),
        'tab:conference-main', 'lrrrr',
        r'Model/layer & Graded [95\% interval] & Support & Cosine & TF-IDF',
        main_rows,
    )
    table(
        'conference-coincidence.tex',
        ('Test coincidence counts for calibrated graded retrieval, summed '
         'over 160 phrase--source tasks. Each task evaluates the same '
         '4,984 test rows; these are repeated decisions, not independent '
         'documents. F1 is a macro mean, not computed from pooled counts.'),
        'tab:conference-coincidence', 'lrrrrr',
        r'Model/layer & TP & FP & FN & TN & Mean F1 (\%)', coincidence_rows,
    )
    table(
        'conference-selection.tex',
        ('Coordinate budgets selected on calibration AP (160 draws per '
         'model/layer). The last column counts empty full-coordinate '
         'test extents at their calibrated thresholds.'),
        'tab:conference-selection', 'lrrrrrr',
        r'Model/layer & 1 & 4 & 16 & 64 & All & Full empty', selection_rows,
    )
    rows = []
    for stem, value in summary.items():
        name = stem.upper().replace('_', ' ')
        rows.append(' & '.join([
            name, *[cell(value['paired_ap'][method])
                    for method in ('support', 'matched_support', 'tfidf')],
        ]))
    table(
        'conference-paired.tex',
        ('Paired differences in test AP, graded minus comparator, in '
         r'percentage points with 95\% phrase-bootstrap intervals. These '
         'are descriptive intervals without a multiple-comparison test.'),
        'tab:conference-paired', 'lrrr',
        r'Model/layer & Selected support & Matched support & TF-IDF', rows,
    )
    rows = []
    for phrase in phrases:
        counts = design['eligible'][phrase]
        rows.append(' & '.join([phrase, *[str(counts[key]) for key in
                                          ('train', 'calibration', 'test')]]))
    table(
        'conference-tasks.tex',
        ('All 32 sampled literal phrase tasks and their positive row counts. '
         'A row may be positive for more than one task.'),
        'tab:conference-tasks', 'lrrr',
        'Phrase & Training & Calibration & Test', rows,
    )
    # Post hoc label dependence diagnostic; no score-dependent task removal.
    positive_sets = [set(design['tasks'][i * 5]['positives'])
                     & set(design['splits']['test']) for i in range(n)]
    overlaps = []
    for i in range(n):
        for j in range(i):
            jac = len(positive_sets[i] & positive_sets[j]) / len(
                positive_sets[i] | positive_sets[j])
            if jac >= .5:
                overlaps.append([phrases[j], phrases[i], jac])
    ntest = len(design['splits']['test'])
    summary['design'] = dict(
        test_prevalence=float(np.mean([
            len(x) / ntest for x in positive_sets])),
        large_label_overlaps=overlaps,
    )
    ablation_rows = []
    for kind in ('sae', 'tc'):
        for layer in (0, 8, 11):
            stem = f'{kind}_{layer}'
            path = OUT / f'ablation_{stem}.json'
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            stats = {}
            primary = json.loads((OUT / f'{stem}.json').read_text())
            chosen_values = np.array([
                np.mean([r['metrics']['graded']['ap']
                         for r in primary['runs'] if r['phrase'] == p])
                for p in phrases
            ])
            for budget in ('1', '4', '16', '64'):
                values = np.array([
                    np.mean([r['metrics'][budget]['ap']
                             for r in report['runs'] if r['phrase'] == p])
                    for p in phrases
                ])
                stats[budget] = interval(values, indices)
                if budget == '1':
                    delta = interval(chosen_values - values, indices)
            summary[stem]['posthoc_budgets'] = stats
            summary[stem]['posthoc_vs_single'] = delta
            ablation_rows.append(' & '.join([
                stem.upper().replace('_', ' '),
                *[cell(stats[k], False) for k in ('1', '4', '16', '64')],
                cell(delta),
            ]))
    if ablation_rows:
        table(
            'conference-ablation.tex',
            ('Post hoc fixed-budget ablation. AP is in percent; the last '
             'column gives selected-budget minus one-coordinate AP in '
             r'percentage points, with a paired 95\% phrase interval.'),
            'tab:conference-ablation', 'lrrrrr',
            r'Model/layer & 1 & 4 & 16 & 64 & Selected minus 1', ablation_rows,
        )
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    RELEASE.mkdir(parents=True, exist_ok=True)
    archive = RELEASE / 'conference_retrieval_results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and not path.name.endswith('_scores.npz'):
                stream.write(path, path.name)
        stream.write(PROTOCOL, 'CONFERENCE_PROTOCOL.md')
        for path in sorted(Path(__file__).parent.glob('*')):
            if path.suffix in ('.py', '.json'):
                dest = Path("src/lattmc/lattconferencetools") / path.name
                stream.write(path, dest)
    # Shard score arrays so no repository artifact exceeds hosting limits.
    for stem in ('sae_0', 'sae_8', 'sae_11', 'tc_0', 'tc_8', 'tc_11',
                 'tfidf'):
        archive = RELEASE / f'conference_scores_{stem}.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as stream:
            for name in (f'{stem}_scores.npz', f'ablation_{stem}_scores.npz'):
                if (OUT / name).exists():
                    stream.write(OUT / name, name)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
