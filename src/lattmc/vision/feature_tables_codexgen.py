"""Produce manuscript tables directly from the CNN/ViT result records."""

from __future__ import annotations
from collections.abc import Sequence

import json
import textwrap

from lattmc.vision.paths_codexgen import experiment_root, repository_root


def write_table(
    name: str,
    columns: str,
    header: str,
    rows: Sequence[str],
    caption: str,
    label: str,
) -> None:
    """Write a captioned feature-analysis table to the paper directory."""
    paper = repository_root() / 'texs/sparsesurrs/visionlattices'
    lines = [r'\begin{table}[tbp]', r'\centering',
             r'\begin{tabular}{' + columns + '}', r'\toprule',
             header + r' \\', r'\midrule']
    lines += [row + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}', r'\caption{' + caption + '}',
              r'\label{' + label + '}', r'\end{table}']
    lines = [part for line in lines for part in textwrap.wrap(
        line, width=79, break_long_words=False, break_on_hyphens=False)]
    (paper / f'tables/{name}.tex').write_text('\n'.join(lines) + '\n')


def tables() -> None:
    """Generate the paper tables from cached feature-analysis results."""
    metrics, features, queries = [], [], []
    for name, title in [('resnet34', 'ResNet34'), ('dinov2', 'DINOv2')]:
        root = experiment_root('imagenette_' + name) / 'results'
        result = json.loads((root / 'experiment_codexgen.json').read_text())
        records = json.loads((root / 'featureviz_codexgen.json').read_text())
        test = result['metrics']['test']['r2']
        transfer = result['metrics']['transfer']['r2']
        alive = f"{result['training_alive']}/{result['latents']}"
        metrics.append(f'{title} & {test:.3f} & {transfer:.3f} & {alive}')
        for i, record in enumerate(records['features']):
            maxima = records['optimized_actual_codes'][2 * i:2 * i + 2]
            features.append(
                f'{title} & {record["class"]} & {record["feature"]} & '
                f'{record["test_ap"]:.3f} & '
                f'{maxima[0]:.1f}, {maxima[1]:.1f}')
        for q in records['lattice_queries']:
            op = q['operation']
            queries.append(f'{title} & {op} & {q["pooled"]} & '
                           f'{q["same_site"]}')
    write_table('imagenette_metrics', 'lrrr',
                r'Model & Test $R^{2}_{\mathrm{tr}}$ & Transfer '
                r'$R^{2}_{\mathrm{tr}}$ & Active', metrics,
                'Reconstruction in original channel units and active\n'
                'training coordinates. Test denotes Imagenette; transfer\n'
                'denotes Imagewoof. All models use the training mean\n'
                'as the reconstruction baseline.', 'tab:imagenette-metrics')
    write_table('imagenette_features', 'llrrr',
                'Model & Selection & Coordinate & Test AP & Optimized codes',
                features, 'Training-selected coordinates, descriptive\n'
                'Imagenette test AP, and actual optimized-stimulus codes\n'
                'for both fixed initializations. Class prevalence is 0.1.',
                'tab:imagenette-features')
    write_table('imagenette_queries', 'llrr',
                'Model & Query & Pooled matches & Same-site matches', queries,
                'Query extents on the 200 evaluation photographs.\n'
                'The meet and join combine source activation requirements.',
                'tab:imagenette-queries')


if __name__ == '__main__':
    tables()
