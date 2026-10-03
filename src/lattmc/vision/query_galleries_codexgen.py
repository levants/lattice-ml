"""Training-selected Imagenette queries with measured meet/join galleries."""

from __future__ import annotations
from typing import Any
from pathlib import Path

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lattmc.vision.backbones_codexgen import dataset, load_codes
from lattmc.vision.contexts_codexgen import graded_score
from lattmc.vision.paths_codexgen import experiment_root, repository_root


CASES = ((1, 'springer'), (6, 'truck'))
OPERATIONS = ('u', 'v', 'meet', 'join')


def compute(
    sample: dict[str, np.ndarray],
    codes: np.ndarray,
    target: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Choose on training data; evaluate every held-out Imagenette image."""
    pooled = codes.max(1)
    train = np.flatnonzero(sample['splits'] == 'train')
    rows = np.flatnonzero(sample['splits'] == 'test')
    labels = sample['labels'][train]
    values = pooled[train]
    contrast = (values[labels == target].mean(0)
                - values[labels != target].mean(0))
    contrast /= values.std(0).clip(1e-12)
    features = np.argsort(-contrast, kind='stable')[:2]
    candidates = train[labels == target]
    sources = []
    for feature in features:
        order = candidates[np.argsort(-pooled[candidates, feature],
                                       kind='stable')]
        sources.append(next(int(i) for i in order if int(i) not in sources))
    base = 0.5 * pooled[sources][:, features]
    vectors = np.stack([base[0], base[1], base.min(0), base.max(0)])
    selected = codes[rows][:, :, features]
    maxima = selected.max(1)
    pooled_masks = (maxima[None] >= vectors[:, None]).all(2)
    site_masks = (selected[None] >= vectors[:, None, None]).all(3).any(2)
    scores = np.stack([graded_score(maxima, q) for q in vectors])
    displayed = np.full((4, 3), -1, dtype=np.int64)
    for i in range(4):
        ranked = np.argsort(-scores[i], kind='stable')
        chosen = rows[ranked[pooled_masks[i, ranked]][:3]]
        displayed[i, :len(chosen)] = chosen
    assert np.array_equal(pooled_masks[3], pooled_masks[0] & pooled_masks[1])
    assert np.all(~(pooled_masks[0] | pooled_masks[1]) | pooled_masks[2])
    assert np.all(~site_masks | pooled_masks)
    extras = pooled_masks[2] & ~(pooled_masks[0] | pooled_masks[1])
    arrays = dict(features=features, sources=np.array(sources), rows=rows,
                  vectors=vectors, pooled=pooled_masks, same_site=site_masks,
                  scores=scores, display_rows=displayed, meet_extra=extras)
    operations = []
    for i, op in enumerate(OPERATIONS):
        operations.append({
            'operation': op, 'thresholds': vectors[i].tolist(),
            'pooled': int(pooled_masks[i].sum()),
            'same_site': int(site_masks[i].sum()),
            'display_rows': displayed[i].tolist(),
            'class_counts': np.bincount(
                sample['labels'][rows[pooled_masks[i]]],
                minlength=len(sample['classes'])).tolist(),
        })
    record = {
        'class': str(sample['classes'][target]), 'class_id': target,
        'features': features.tolist(), 'sources': sources,
        'train_contrasts': contrast[features].tolist(),
        'source_ids': sample['source_ids'][sources].tolist(),
        'evaluation': 'all 100 Imagenette test images; no Imagewoof',
        'meet_extra_beyond_union': int(extras.sum()),
        'operations': operations,
    }
    return record, arrays


def render(
    sample: dict[str, np.ndarray],
    record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    name: str,
    slug: str,
    paper: Path,
) -> None:
    """Draw sources and ranked matches, with empty slots kept explicit."""
    fig, axes = plt.subplots(5, 4, figsize=(9.8, 11.8),
                             layout='constrained')
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    features = record['features']
    axes[0, 0].axis('off')
    title = 'ResNet34' if name == 'resnet34' else 'DINOv2 ViT-S/14'
    axes[0, 0].text(
        0, .95, f'{title}\n{record["class"]}\n\nCoordinates\n'
        f'({features[0]}, {features[1]})\n\nTraining sources →',
        va='top', fontsize=12)
    for i, row in enumerate(record['sources']):
        axes[0, i + 1].imshow(sample['images'][row])
        axes[0, i + 1].set_title(f'Source {OPERATIONS[i]} · row {row}',
                                fontsize=12)
    axes[0, 3].axis('off')
    axes[0, 3].text(
        0, .9, 'meet = coordinate min\njoin = coordinate max\n\n'
        '100 test photographs\n3 ranked matches / row\n\n'
        's = satisfaction ratio\nsite = common witness',
        va='top', fontsize=11)
    if arrays['meet_extra'].any():
        ax = axes[0, 3]
        ax.clear()
        ax.set_axis_on()
        position = int(np.flatnonzero(arrays['meet_extra'])[0])
        row = arrays['rows'][position]
        ax.imshow(sample['images'][row])
        ax.set_xticks([])
        ax.set_yticks([])
        label = sample['classes'][sample['labels'][row]]
        ax.set_title(f'Meet-only: {label}\nrow {row}', fontsize=11)
        a, b, m = arrays['scores'][:3, position]
        ax.set_xlabel(f'u: {a:.2f}, v: {b:.2f}, meet: {m:.2f}',
                      fontsize=10)
    for i, operation in enumerate(record['operations']):
        info = axes[i + 1, 0]
        info.axis('off')
        thresholds = ', '.join(f'{v:.2f}' for v in arrays['vectors'][i])
        info.text(0, .75, f'{operation["operation"]}: ({thresholds})\n\n'
                  f'pooled = {operation["pooled"]}\n'
                  f'same site = {operation["same_site"]}', fontsize=12)
        for col, row in enumerate(arrays['display_rows'][i]):
            ax = axes[i + 1, col + 1]
            if row < 0:
                ax.axis('off')
                ax.text(.5, .5, 'No further match', ha='center',
                        va='center', fontsize=11, color='0.4')
                continue
            position = int(np.flatnonzero(arrays['rows'] == row)[0])
            ax.imshow(sample['images'][row])
            label = sample['classes'][sample['labels'][row]]
            score = arrays['scores'][i, position]
            site = 'yes' if arrays['same_site'][i, position] else 'no'
            ax.set_title(f'{label} · row {row}', fontsize=11)
            ax.set_xlabel(f's = {score:.2f} · site: {site}', fontsize=11)
    path = paper / f'figures/query_{slug}_{name}_codexgen.pdf'
    fig.savefig(path)
    plt.close(fig)


def run() -> dict[str, Any]:
    """Build and save galleries for the registered exemplar queries."""
    sample = dataset()
    paper = repository_root() / 'texs/sparsesurrs/visionlattices'
    report = {}
    for name in ['resnet34', 'dinov2']:
        codes = load_codes(name)['codes']
        root = experiment_root('imagenette_' + name)
        report[name] = []
        for target, slug in CASES:
            record, arrays = compute(sample, codes, target)
            np.savez_compressed(
                root / f'retrieval/query_{slug}_codexgen.npz', **arrays)
            render(sample, record, arrays, name, slug, paper)
            report[name].append(record)
        (root / 'results/query_galleries_codexgen.json').write_text(
            json.dumps(report[name], indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run()
