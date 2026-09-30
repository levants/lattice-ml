"""Aggregate scores, image bootstrap intervals, and optional tables."""

import argparse
import json
from pathlib import Path

import numpy as np

from lattmc.vision.overcomplete_data_codexgen import load
from lattmc.vision.overcomplete_evaluate_codexgen import read_codes
from lattmc.vision.overcomplete_fetch_codexgen import ROOT


FAMILIES = ['topk', 'batchtopk', 'jump', 'archetypal', 'relu_fixed']
LABELS = ['TopK', 'BatchTopK', 'JumpReLU', 'RA-TopK', 'ReLU (fixed)']
DATASETS = ['imagenette', 'imagewoof', 'pets', 'parts', 'dtd']


def aggregate():
    result = {'local': [], 'transfer': [], 'stability': json.loads(
        (ROOT / 'results/stability_codexgen.json').read_text())}
    for family, label in zip(FAMILIES, LABELS):
        for budget in [16, 32]:
            rows = []
            for seed in range(3):
                name = f'{family}_k{budget}_s{seed}'
                path = ROOT / f'results/{name}/imagenette_codexgen.json'
                rows.append(json.loads(path.read_text())['metrics']['test'])
            result['local'].append({'family': family, 'label': label,
                                    'budget': budget, **{
                k: {'mean': float(np.mean([r[k] for r in rows])),
                    'sd': float(np.std([r[k] for r in rows], ddof=1))}
                for k in ['r2', 'l0']}})
    names = [f'{f}_k32_s0' for f in FAMILIES]
    names += ['pretrained_ra', 'prisma_transcoder']
    for name in names:
        for dataset in DATASETS:
            path = ROOT / f'results/{name}/{dataset}_codexgen.json'
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            matrix, stats = read_codes(name, dataset)
            _, records = load(dataset)
            test = np.array([v['split'] == 'test' for v in records])
            errors, baseline = stats['error'][test], stats['baseline'][test]
            rng = np.random.default_rng(530)
            ids = rng.integers(len(errors), size=(2000, len(errors)))
            scores = 1 - errors[ids].sum(1) / baseline[ids].sum(1)
            row = {'model': name, 'dataset': dataset,
                   **r['metrics']['test'], 'image_bootstrap_95':
                   np.quantile(scores, [.025, .975]).tolist()}
            if name == 'prisma_transcoder':
                z = np.load(ROOT / f'codes/{name}/{dataset}_codexgen.npz')
                row['skip_only_r2'] = float(
                    1 - z['skip_error'][test].sum() / baseline.sum())
            result['transfer'].append(row)
    path = ROOT / 'results/summary_codexgen.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    return result


def tables(summary, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = [r'\begin{table}[tb]', r'\centering\small',
            r'\caption{Imagenette test reconstruction and active coordinates.',
            r'Mean $\pm$ sample standard deviation over three seeds.',
            r'The nominal setting is not the measured sparsity.}',
            r'\label{tab:overcomplete-local}',
            r'\begin{tabular}{lrrr}\toprule',
            r'Family & Setting & $R^{2}$ & Active coordinates \\\midrule']
    for row in summary['local']:
        values = [row['label'], str(row['budget'])]
        for key, digits in [('r2', 4), ('l0', 1)]:
            stat = row[key]
            values.append(f"${stat['mean']:.{digits}f}"
                          r'\pm' + f"{stat['sd']:.{digits}f}$")
        rows.append(' & '.join(values) + r' \\')
    rows.extend([r'\bottomrule\end{tabular}', r'\end{table}'])
    (folder / 'overcomplete_local_codexgen.tex').write_text(
        '\n'.join(rows) + '\n')
    rows = [r'\begin{table}[tb]', r'\centering\small',
            r'\caption{Transfer reconstruction $R^{2}$ for local setting-$32$',
            r'seed-$0$ SAEs and the pretrained RA-SAE. All use DINOv2 final',
            r'states; the pretrained dictionary has a different width and',
            r'training history. Full sparsities and image-bootstrap intervals',
            r'are in the released result tables.}',
            r'\label{tab:overcomplete-transfer}',
            r'\begin{tabular}{lrrrr}\toprule',
            r'Model & Imagewoof & Pets & Parts & DTD \\\midrule']
    names = [f'{f}_k32_s0' for f in FAMILIES] + ['pretrained_ra']
    for name, label in zip(names, LABELS + ['Pretrained RA']):
        values = [label]
        for dataset in DATASETS[1:]:
            found = [r for r in summary['transfer'] if
                     r['model'] == name and r['dataset'] == dataset]
            values.append(f"{found[0]['r2']:.3f}" if found else '--')
        rows.append(' & '.join(values) + r' \\')
    rows.extend([r'\bottomrule\end{tabular}', r'\end{table}'])
    (folder / 'overcomplete_transfer_codexgen.tex').write_text(
        '\n'.join(rows) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-tables')
    args = parser.parse_args()
    summary = aggregate()
    if args.paper_tables:
        tables(summary, args.paper_tables)
