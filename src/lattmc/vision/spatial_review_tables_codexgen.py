"""Export the cached review audit and complete supporting measurements."""

from __future__ import annotations
from typing import Any
from collections.abc import Sequence

import argparse
import json
from pathlib import Path

import numpy as np

from lattmc.vision.spatial_review_codexgen import DATA, MODELS, OUT


LABELS = ['TopK', 'BatchTopK', 'JumpReLU', 'RA-TopK', 'ReLU (fixed)',
          'Pretrained RA', 'Transcoder']
NAMES = dict(zip(MODELS, LABELS))
FAMILIES = dict(topk='TopK', batchtopk='BatchTopK', jump='JumpReLU',
                archetypal='RA-TopK', relu_fixed='ReLU (fixed)',
                relu='ReLU (adaptive)')


def read(path: Path) -> dict[str, Any]:
    """Load a cached JSON result file."""
    return json.loads(path.read_text())


def totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum query-satisfaction counts and calculate the common-site discrepancy.
    """
    keys = ['pooled', 'common', 'independent_expected',
            'separate_query_intersection']
    r = {k: sum(x[k] for x in rows) for k in keys}
    r['discrepancy'] = (1 - r['common'] / r['pooled']
                        if r['pooled'] else None)
    return r


def table(
    folder: Path,
    stem: str,
    caption: str,
    columns: str,
    header: str,
    rows: Sequence[Sequence[object]],
    long: bool = False,
) -> None:
    """Write a regular or long LaTeX table with experiment notes."""
    env = 'longtable' if long else 'tabular'
    lines = ([r'\begingroup\small'] if long else
             [r'\begin{table}[tb]', r'\centering\small'])
    if long:
        lines.append(r'\begin{longtable}{' + columns + '}')
    lines += [r'\caption{' + caption + '}', r'\label{tab:' + stem + '}']
    if long:
        lines += [r'\\\toprule', header + r' \\\midrule',
                  r'\endfirsthead', r'\toprule',
                  header + r' \\\midrule', r'\endhead',
                  r'\bottomrule\endfoot']
    else:
        lines += [r'\begin{tabular}{' + columns + r'}\toprule',
                  header + r' \\\midrule']
    for row in rows:
        # Whitespace between cells is immaterial; keep sources navigable.
        lines.append(' & '.join(map(str, row)) + r' \\')
    if not long:
        lines.append(r'\bottomrule')
    lines += [r'\end{' + env + '}',
              r'\endgroup' if long else r'\end{table}']
    wrapped = []
    for line in lines:
        while len(line) > 77:
            split = line.rfind(' ', 0, 77)
            if split < 1:
                raise ValueError(line)
            wrapped.append(line[:split])
            line = line[split + 1:]
        wrapped.append(line)
    (folder / (stem.replace('-', '_') + '_codexgen.tex')).write_text(
        '\n'.join(wrapped) + '\n')


def main(folder: Path) -> None:
    """Generate spatial-review tables from the cached experiment results."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    audit = read(OUT / 'spatial_audit_codexgen.json')['results']
    aggregate, main_rows, detailed, sensitivity = [], [], [], []
    for model, label in NAMES.items():
        for beta in [.5, 1., 1.5]:
            r = totals([x for x in audit if x['model'] == model
                        and x['threshold_multiplier'] == beta])
            aggregate.append(dict(model=model, multiplier=beta, **r))
            gap = (f"{100 * r['discrepancy']:.1f}"
                   if r['discrepancy'] is not None else '--')
            vals = [r['pooled'], r['common'],
                    f"{r['independent_expected']:.1f}", gap]
            sensitivity.append([label, beta, *vals])
            if beta == 1:
                main_rows.append([label, r['pooled'], r['common'],
                                  r['separate_query_intersection'],
                                  vals[2], gap])
        for dataset in ['imagenette', 'imagewoof', 'pets', 'parts', 'dtd']:
            r = totals([x for x in audit if x['model'] == model
                        and x['dataset'] == dataset
                        and x['threshold_multiplier'] == 1])
            detailed.append([label, dataset, r['pooled'], r['common'],
                             r['separate_query_intersection'],
                             f"{r['independent_expected']:.1f}"])
    (OUT / 'summary_codexgen.json').write_text(
        json.dumps(aggregate, indent=2) + '\n')
    table(folder, 'review-spatial-main',
          'Frozen join queries across five test collections. '
          '$G$ denotes pooled matches, $H$ common join witnesses, '
          r'$C=|H(u)\cap H(v)|$, and $E$ conditional expected common '
          r'matches. $\Delta$ is the pooled match-weighted discrepancy. '
          'Each model has four queries on 547 records; counts overlap '
          'across queries and are not independent sample sizes.',
          'lrrrrr', r'Model & $G$ & $H$ & $C$ & $E$ & $\Delta$ (\%)',
          main_rows)
    table(folder, 'review-spatial-datasets',
          'Unscaled joins by dataset; four queries per row. '
          'Definitions of $G,H,C,E$ agree with the main table.',
          'llrrrr', r'Model & Dataset & $G$ & $H$ & $C$ & $E$',
          detailed, long=True)
    table(folder, 'review-thresholds',
          r'Sensitivity to a common threshold multiplier $\beta$. '
          'All multipliers are reported; none is selected on test data. '
          'The discrepancy is undefined when $G=0$.',
          'lrrrrr', r'Model & $\beta$ & $G$ & $H$ & $E$ & '
          r'$\Delta$ (\%)', sensitivity, long=True)
    rows = []
    for family, label in FAMILIES.items():
        for budget in [16, 32]:
            for seed in range(3):
                name = f'{family}_k{budget}_s{seed}'
                r = read(DATA / f'results/{name}/imagenette_codexgen.json')
                m = r['metrics']['test']
                rows.append([label, budget, seed, f"{m['r2']:.4f}",
                             f"{m['l0']:.2f}"])
    table(folder, 'review-all-fits',
          'Every local Imagenette test fit, including the failed '
          'adaptive-penalty diagnostic. Setting is a training parameter, '
          'not measured sparsity.', 'lrrrr',
          r'Family & Setting & Seed & $R^{2}_{\mathrm{tr}}$ & Active',
          rows, long=True)
    rows = []
    for r in read(DATA / 'results/summary_codexgen.json')['transfer']:
        lo, hi = r['image_bootstrap_95']
        skip = r.get('skip_only_r2')
        rows.append([NAMES[r['model']], r['dataset'], r['images'],
                     f"{r['r2']:.3f}", f'[{lo:.3f}, {hi:.3f}]',
                     f"{r['l0']:.1f}",
                     '--' if skip is None else f'{skip:.3f}'])
    table(folder, 'review-transfer',
          'Complete reconstruction diagnostics: local setting-32 seed-0 '
          r'models and pretrained checkpoints. Intervals are 95\% '
          'percentile intervals from 2,000 image bootstrap replicates '
          'within each fixed test subset. Skip refers only to the '
          'transcoder; its prediction target differs from the SAE target.',
          'llrrrrr', r'Model & Dataset & $n$ & $R^{2}$ & Interval & '
          r'Active & Skip', rows, long=True)
    rows = []
    for r in read(DATA / 'results/stability_codexgen.json'):
        rows.append([FAMILIES[r['family']], r['budget'],
                     '/'.join(map(str, r['seed_pair'])),
                     f"{r['decoder_cosine']:.3f}",
                     f"{r['extent_jaccard']:.3f}",
                     r['nonempty_comparisons']])
    table(folder, 'review-stability',
          'All seed-0 decoder assignments and test-extent comparisons. '
          '$n$ counts eligible nonempty extent unions, not independent '
          'images. Both-empty comparisons are omitted.', 'lrr rrr',
          r'Family & Setting & Seeds & Cosine & Jaccard & $n$',
          rows, long=True)
    rows = []
    for model, label in NAMES.items():
        for dataset in ['pets', 'parts']:
            f = DATA / f'results/{model}/{dataset}_annotations_codexgen.json'
            values = read(f)['features']
            rows.append([label, dataset, len(values), *[
                f'{np.mean([r[k] for r in values]):.3f}' for k in
                ['annotation_coverage', 'random_mean',
                 'norm_matched_coverage']]])
    table(folder, 'review-annotations',
          'Annotation diagnostics with both references. Means use only '
          'coordinates with positive test responses; $n$ is that count. '
          'References use the same selected images. Foreground and part '
          'occupancy do not certify semantic correctness.', 'llrrrr',
          r'Model & Dataset & $n$ & Peak & Random & Norm', rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-tables', required=True)
    main(parser.parse_args().paper_tables)
