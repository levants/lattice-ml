"""Annotation diagnostics with unconstrained classes and patch positions."""

import argparse
import json

import numpy as np

from lattmc.vision.overcomplete_data_codexgen import load
from lattmc.vision.overcomplete_evaluate_codexgen import read_codes
from lattmc.vision.overcomplete_extract_codexgen import load_dense
from lattmc.vision.overcomplete_fetch_codexgen import ROOT


def occupancy(masks, grid, label):
    block = 224 // grid
    values = (masks == label).reshape(-1, grid, block, grid, block)
    return values.mean((2, 4)).reshape(-1, grid * grid)


def analyze(name, dataset):
    definition = json.loads((ROOT / f'results/{name}/queries_codexgen.json')
                            .read_text())
    features = definition['selected_features']
    data, records = load(dataset)
    test = np.array([r['split'] == 'test' for r in records])
    matrix, _ = read_codes(name, dataset)
    sites = matrix.shape[0] // len(records)
    grid = round(sites ** 0.5)
    codes = matrix[:, features].toarray().reshape(len(records), sites, -1)
    codes, masks = codes[test], data['masks'][test]
    labels = np.array([r['label'] for r in records])[test]
    # The published segmentation palette has part IDs 0..39; 40 is background.
    # Preserve IDs, avoiding an unverified semantic relabeling.
    categories = [1] if dataset == 'pets' else list(range(40))
    coverage = np.stack([occupancy(masks, grid, c) for c in categories], -1)
    dense = load_dense(dataset)[test]
    norm = np.linalg.norm(dense, axis=2)
    if sites != 256:
        # Norm-matching must use the actual model's cached input states.
        paths = sorted((ROOT / f'activations/{name}/{dataset}').glob('*.npz'))
        dense = np.concatenate([np.load(p)['input'] for p in paths])[test]
        norm = np.linalg.norm(dense, axis=2)
    rng = np.random.default_rng(137)
    results = []
    for j, feature in enumerate(features):
        response = codes[:, :, j]
        maxima = response.max(1)
        images = np.argsort(-maxima, kind='stable')[:20]
        images = images[maxima[images] > 0]
        if not len(images):
            continue
        positions = response[images].argmax(1)
        observed = coverage[images, positions]
        random_values = []
        for _ in range(200):
            random_sites = rng.integers(sites, size=len(images))
            random_values.append(coverage[images, random_sites].mean(0))
        delta = np.abs(norm[images] - norm[images, positions, None])
        delta[np.arange(len(images)), positions] = np.inf
        matched = delta.argmin(1)
        # Part ID is selected for descriptive scoring on this test collection.
        # Apply that same maximum operation to every randomized reference.
        actual = float(observed.mean(0).max())
        random_scores = np.array(random_values).max(1)
        matched_score = float(coverage[images, matched].mean(0).max())
        results.append({'feature': feature, 'images': len(images),
                        'distinct_labels': len(set(labels[images])),
                        'dominant_annotation': int(
                            categories[observed.mean(0).argmax()]),
                        'annotation_coverage': actual,
                        'random_mean': float(random_scores.mean()),
                        'random_q025_q975':
                        np.quantile(random_scores, [.025, .975]).tolist(),
                        'norm_matched_coverage': matched_score})
    result = {'model': name, 'dataset': dataset, 'features': results,
              'note': 'Annotation association is not a universal feature '
              'meaning. Different classes and positions are allowed.',
              'mean_coverage': float(np.mean(
                  [r['annotation_coverage'] for r in results])),
              'mean_random': float(np.mean(
                  [r['random_mean'] for r in results])),
              'mean_norm_matched': float(np.mean(
                  [r['norm_matched_coverage'] for r in results]))}
    path = ROOT / f'results/{name}/{dataset}_annotations_codexgen.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(name, dataset, result['mean_coverage'], result['mean_random'],
          result['mean_norm_matched'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('dataset', choices=['pets', 'parts'])
    args = parser.parse_args()
    analyze(args.name, args.dataset)
