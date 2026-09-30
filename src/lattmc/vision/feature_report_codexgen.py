"""Render measured CNN/ViT feature evidence without relabeling mismatches."""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lattmc.vision.backbones_codexgen import dataset, load_codes
from lattmc.vision.paths_codexgen import experiment_root, repository_root


def plots():
    sample = dataset()
    paper = repository_root() / 'texs/sparsesurrs/visionlattices'
    records_by_model = {}
    for name in ['resnet34', 'dinov2']:
        root = experiment_root('imagenette_' + name)
        records = json.loads((root / 'results/featureviz_codexgen.json')
                             .read_text())['features']
        records_by_model[name] = records
        cache = load_codes(name)
        grid = int(np.sqrt(cache['codes'].shape[1]))
        pooled = cache['codes'].max(1)
        transfer = np.flatnonzero(sample['splits'] == 'transfer')
        groups = [(r, np.array(r['test_rows']), r['class']) for r in records]
        feature = records[0]['feature']
        top = transfer[np.argsort(-pooled[transfer, feature],
                                   kind='stable')[:3]]
        groups.append((records[0], top, 'Imagewoof transfer'))
        fig, axes = plt.subplots(3, 6, figsize=(12, 7), layout='constrained')
        for row, (record, ids, label) in enumerate(groups):
            feature = record['feature']
            vmax = pooled[ids, feature].max()
            for col, index in enumerate(ids):
                raw, heat = axes[row, 2 * col:2 * col + 2]
                raw.imshow(sample['images'][index])
                classes = (sample['transfer_classes'] if row == 2
                           else sample['classes'])
                actual = classes[sample['labels'][index]]
                raw.set_title(f'{actual}\nrow {index}', fontsize=11)
                heat.imshow(cache['codes'][index, :, feature].reshape(
                    grid, grid), cmap='magma', vmin=0, vmax=vmax,
                    interpolation='nearest')
                heat.set_title(f'max {pooled[index, feature]:.2f}',
                               fontsize=11)
                for ax in [raw, heat]:
                    ax.set_xticks([])
                    ax.set_yticks([])
                if col == 0:
                    raw.set_ylabel(f'F{feature}\n{label}', fontsize=11)
        fig.savefig(paper / f'figures/imagenette_{name}_codexgen.pdf')
        plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(8, 10), layout='constrained')
    for model_index, name in enumerate(['resnet34', 'dinov2']):
        root = experiment_root('imagenette_' + name)
        with np.load(root / 'visualization/optimized_codexgen.npz') as data:
            for offset, record in enumerate(records_by_model[name]):
                row = 2 * model_index + offset
                axes[row, 0].imshow(sample['images'][record['source_row']])
                axes[row, 0].set_title('Training reference', fontsize=11)
                axes[row, 0].set_ylabel(
                    f'{name} F{record["feature"]}\n{record["class"]}',
                    fontsize=11)
                for seed in range(2):
                    index = 2 * offset + seed
                    axes[row, seed + 1].imshow(
                        data['optimized'][index].transpose(1, 2, 0))
                    before = data['initial_codes'][index]
                    after = data['optimized_codes'][index]
                    axes[row, seed + 1].set_title(
                        f'Init {seed + 1}: {before:.1f} → {after:.1f}',
                        fontsize=11)
                for ax in axes[row]:
                    ax.set_xticks([])
                    ax.set_yticks([])
    fig.savefig(paper / 'figures/optimized_features_codexgen.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout='constrained')
    natural, nat_axes = plt.subplots(1, 2, figsize=(9, 3.2),
                                     layout='constrained')
    for row, name in enumerate(['resnet34', 'dinov2']):
        root = experiment_root('imagenette_' + name)
        with np.load(root / 'visualization/probes_codexgen.npz') as data:
            for col, record in enumerate(records_by_model[name]):
                ax = axes[row, col]
                for kind in ['curve', 'line', 'corner']:
                    ids = data['kinds'] == kind
                    ax.plot(data['angles'][ids],
                            data['synthetic_scores'][ids, col],
                            marker='.', label=kind)
                ax.set_ylim(bottom=0)
                if data['synthetic_scores'][:, col].max() == 0:
                    ax.set_ylim(0, 1)
                    ax.text(0.5, 0.45, 'No active sparse code',
                            transform=ax.transAxes, ha='center', color='0.4')
                ax.set_title(f'{name} F{record["feature"]}')
                ax.set_xlabel('Orientation (degrees)')
                ax.set_ylabel('Maximum sparse code')
                ax.legend(fontsize=8)
                nat_axes[row].plot(range(0, 360, 30),
                                   data['rotation_scores'][col], marker='.',
                                   label=f'F{record["feature"]}')
            nat_axes[row].set_title(name)
            nat_axes[row].set_xlabel('Training image rotation (degrees)')
            nat_axes[row].set_ylabel('Maximum sparse code')
            nat_axes[row].legend()
    fig.savefig(paper / 'figures/synthetic_tuning_codexgen.pdf')
    natural.savefig(paper / 'figures/rotation_tuning_codexgen.pdf')
    plt.close(fig)
    plt.close(natural)
    # Show the exact synthetic inputs at four orientations.
    root = experiment_root('imagenette_resnet34')
    with np.load(root / 'visualization/probes_codexgen.npz') as data:
        fig, axes = plt.subplots(3, 4, figsize=(7, 5), layout='constrained')
        for row, kind in enumerate(['curve', 'line', 'corner']):
            for col, angle in enumerate([0, 90, 180, 270]):
                index = np.flatnonzero((data['kinds'] == kind)
                                       & (data['angles'] == angle))[0]
                axes[row, col].imshow(data['synthetic'][index]
                                      .transpose(1, 2, 0))
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if row == 0:
                    axes[row, col].set_title(f'{angle} degrees')
                if col == 0:
                    axes[row, col].set_ylabel(kind)
        fig.savefig(paper / 'figures/synthetic_stimuli_codexgen.pdf')
        plt.close(fig)
    print('Rendered six CNN/ViT evidence figures')


if __name__ == '__main__':
    plots()
