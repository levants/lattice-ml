"""Measured, dataset-balanced galleries and query witnesses from caches."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from lattmc.vision.overcomplete_data_codexgen import load
from lattmc.vision.overcomplete_evaluate_codexgen import pool, read_codes
from lattmc.vision.overcomplete_fetch_codexgen import ROOT


DATASETS = ['imagenette', 'imagewoof', 'pets', 'parts', 'dtd']
TITLES = ['Imagenette', 'Imagewoof', 'Pets', 'PartImageNet', 'DTD']
OUT = ROOT / 'figures'


def panel(ax, image, position, grid, title):
    ax.imshow(image)
    y, x = divmod(int(position), grid)
    size = 224 // grid
    ax.add_patch(Rectangle((x * size, y * size), size, size,
                           fill=False, edgecolor='#ff3322', linewidth=1.7))
    # A larger neighborhood is shown for context, not as a receptive field.
    radius = 1 if grid == 16 else 0
    x0, x1 = max(0, x - radius) * size, min(grid, x + radius + 1) * size
    y0, y1 = max(0, y - radius) * size, min(grid, y + radius + 1) * size
    inset = ax.inset_axes([0.64, 0.01, 0.35, 0.35])
    inset.imshow(image[y0:y1, x0:x1], interpolation='nearest')
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color('#ff3322')
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def foreground_features(name):
    from lattmc.vision.overcomplete_annotations_codexgen import occupancy
    frequencies = []
    for dataset in ['pets', 'imagewoof', 'parts']:
        matrix, _ = read_codes(name, dataset)
        _, records = load(dataset)
        test = np.array([r['split'] == 'test' for r in records])
        frequencies.append((pool(matrix)[test] > 0).sum(0))
    eligible = ((frequencies[0] >= 10) & (frequencies[1] >= 5)
                & (frequencies[2] >= 5))
    features = np.flatnonzero(eligible)
    matrix, _ = read_codes(name, 'pets')
    data, records = load('pets')
    codes = matrix[:, features].toarray().reshape(len(records), 256, -1)
    coverage = occupancy(data['masks'], 16, 1)
    scores = []
    for j, feature in enumerate(features):
        response = codes[:, :, j]
        active = response.max(1) > 0
        positions = response.argmax(1)[active]
        foreground = coverage[np.flatnonzero(active), positions].mean()
        counts = np.bincount(positions, minlength=256)
        probability = counts[counts > 0] / len(positions)
        entropy = -(probability * np.log(probability)).sum() / np.log(256)
        scores.append((float(foreground * entropy), int(feature)))
    return [feature for _, feature in sorted(scores, reverse=True)[:3]]


def gallery(name, transfer=False, foreground=False):
    OUT.mkdir(parents=True, exist_ok=True)
    definition = json.loads((ROOT / f'results/{name}/queries_codexgen.json')
                            .read_text())
    features = definition['selected_features'][:3]
    suffix = 'gallery'
    if transfer:
        frequencies = []
        for dataset in DATASETS:
            _, records = load(dataset)
            matrix, _ = read_codes(name, dataset)
            sites = matrix.shape[0] // len(records)
            maxima = pool(matrix, sites)
            test = np.array([r['split'] == 'test' for r in records])
            frequencies.append((maxima[test] > 0).mean(0))
        score = np.prod(frequencies, axis=0) ** (1 / len(DATASETS))
        features = np.argsort(-score, kind='stable')[:3].tolist()
        suffix = 'transfer_gallery'
    if foreground:
        features = foreground_features(name)
        suffix = 'foreground_gallery'
    fig, axes = plt.subplots(3, 5, figsize=(11, 7))
    provenance = []
    for col, dataset in enumerate(DATASETS):
        data, records = load(dataset)
        matrix, _ = read_codes(name, dataset)
        sites = matrix.shape[0] // len(records)
        grid = round(sites ** 0.5)
        codes = matrix[:, features].toarray().reshape(len(records), sites, -1)
        ids = np.array([i for i, r in enumerate(records)
                        if r['split'] == 'test'])
        for row, feature in enumerate(features):
            maxima = codes[ids, :, row].max(1)
            image_id = int(ids[maxima.argmax()])
            position = int(codes[image_id, :, row].argmax())
            response = float(codes[image_id, position, row])
            record = records[image_id]
            title = f'{TITLES[col]} | {response:.2f}'
            if response == 0:
                axes[row, col].text(.5, .5, 'No positive test response',
                                    ha='center', fontsize=8)
                axes[row, col].set_title(TITLES[col], fontsize=7)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            else:
                panel(axes[row, col], data['images'][image_id], position,
                      grid, title)
            if col == 0:
                axes[row, col].set_ylabel(f'Feature {feature}', fontsize=9)
            provenance.append({'model': name, 'feature': feature,
                               'dataset': dataset, 'row': image_id,
                               'position': position, 'response': response,
                               **record})
    fig.suptitle(name.replace('_', ' ') +
                 ': strongest test response within each dataset', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f'{name}_{suffix}_codexgen.png', dpi=180)
    plt.close(fig)
    (OUT / f'{name}_{suffix}_codexgen.json').write_text(json.dumps({
        'selection': ('Exploratory: pet foreground occupancy times spatial '
                      'entropy, with transfer support requirements; not an '
                      'unbiased evaluation' if foreground else
                      'Exploratory: test-set activation frequency across '
                      'datasets; not an unbiased evaluation' if transfer else
                      'First three training-selected features'),
        'panels': provenance}, indent=2) + '\n')


def queries(name='topk_k32_s0', pair=3):
    definition = json.loads((ROOT / f'results/{name}/queries_codexgen.json')
                            .read_text())['queries'][pair]
    features = definition['features']
    images, codes, metadata = [], [], []
    for dataset in DATASETS:
        data, records = load(dataset)
        matrix, _ = read_codes(name, dataset)
        sites = matrix.shape[0] // len(records)
        z = matrix[:, features].toarray().reshape(len(records), sites, 2)
        ids = [i for i, r in enumerate(records) if r['split'] == 'test']
        images.extend(data['images'][ids])
        codes.extend(z[ids])
        metadata.extend(dict(dataset=dataset, row=i, record=records[i])
                        for i in ids)
    codes = np.array(codes)
    u, v = np.array(definition['u']), np.array(definition['v'])
    operations = {'meet': np.minimum(u, v), 'u': u, 'v': v,
                  'join': np.maximum(u, v)}
    fig, axes = plt.subplots(4, 5, figsize=(11, 9))
    provenance = []
    for row, (operation, query) in enumerate(operations.items()):
        margin = (codes / query).min(2)
        strength = margin.max(1)
        selected = np.argsort(-strength, kind='stable')
        selected = selected[strength[selected] >= 1][:5]
        for col, ax in enumerate(axes[row]):
            if col >= len(selected):
                ax.axis('off')
                continue
            index = int(selected[col])
            position = int(margin[index].argmax())
            info = metadata[index]
            panel(ax, images[index], position, 16,
                  f"{info['dataset']} | margin {strength[index]:.2f}")
            if col == 0:
                ax.set_ylabel(f'{operation}: {int((strength >= 1).sum())}'
                              ' images', fontsize=9)
            provenance.append(dict(operation=operation,
                                   query=query.tolist(), position=position,
                                   codes=codes[index, position].tolist(),
                                   **info))
    fig.suptitle(f'Features {features}: frozen queries across five test sets',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'meet_join_codexgen.png', dpi=180)
    plt.close(fig)
    (OUT / 'meet_join_codexgen.json').write_text(
        json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', nargs='?', default='topk_k32_s0')
    parser.add_argument('--queries', action='store_true')
    parser.add_argument('--transfer-gallery', action='store_true')
    parser.add_argument('--foreground-gallery', action='store_true')
    args = parser.parse_args()
    gallery(args.name, args.transfer_gallery, args.foreground_gallery)
    if args.queries:
        queries()
