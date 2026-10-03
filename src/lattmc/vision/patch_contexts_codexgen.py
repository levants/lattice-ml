"""Exact projected patch concepts, common witnesses, and finite downsets."""

from __future__ import annotations
from typing import Any

import argparse
import json

import numpy as np

from lattmc.vision.contexts_codexgen import graded_score
from lattmc.vision.patch_models_codexgen import ROOT


def skyline(points: np.ndarray) -> np.ndarray:
    """Maximal generators of a nonempty two-dimensional finite downset."""
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    answer, largest_y = [], -np.inf
    for row in points[order]:
        if row[1] > largest_y:
            answer.append(row)
            largest_y = row[1]
    return np.array(answer)


def intersect(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compute maximal generators for the intersection of two downsets."""
    values = np.minimum(first[:, None, :], second[None, :, :])
    return skyline(values.reshape(-1, 2))


def downset_extent(sites: np.ndarray, generators: np.ndarray) -> np.ndarray:
    """Test whether every generator is dominated at some image site."""
    return np.array([all(np.any(np.all(image >= g, axis=1))
                         for g in generators) for image in sites])


def read(
    name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    """Load image metadata, selected patch codes, and feature definitions."""
    root = ROOT / name
    with np.load(root / 'dataset/images_codexgen.npz') as data:
        sample = {k: data[k] for k in data.files}
    with np.load(root / 'activations/selected_codexgen.npz') as data:
        codes, features = data['codes'], data['features']
    path = root / 'results/extraction_codexgen.json'
    metadata = json.loads(path.read_text())
    return sample, codes, features, metadata


def analyze(name: str) -> dict[str, Any]:
    """Compare pooled, same-site, and downset query satisfaction."""
    sample, codes, features, metadata = read(name)
    root = ROOT / name
    (root / 'contexts').mkdir(exist_ok=True)
    train = sample['splits'] == 'train'
    test = sample['splits'] == 'test'
    rms = np.sqrt((codes[train].max(1).astype(float) ** 2).mean(0))
    rms = rms.clip(1e-12)
    flat = codes.reshape(-1, len(features))
    top = flat.max(0)
    all_cases = []
    for case_number, original in enumerate(metadata['cases']):
        pair = [2 * case_number, 2 * case_number + 1]
        candidates = np.flatnonzero(train & (
            sample['labels'] == original['class_id']))
        pair_codes = codes[:, :, pair]
        common = np.min(pair_codes / rms[pair], axis=2)
        ranked = candidates[np.argsort(-common[candidates].max(1),
                                       kind='stable')]
        sources = ranked[:2]
        sites = common[sources].argmax(1)
        base = np.zeros((2, len(features)), dtype=np.float32)
        for i in range(2):
            base[i, pair] = .5 * pair_codes[sources[i], sites[i]]
        queries = np.stack([base[0], base[1], np.minimum(*base),
                            np.maximum(*base)])
        extents = (flat[None] >= queries[:, None]).all(2)
        site_masks = extents.reshape(4, len(codes), codes.shape[1])
        common_images = site_masks.any(2)
        pooled = (codes.max(1)[None] >= queries[:, None]).all(2)
        assert np.array_equal(extents[3], extents[0] & extents[1])
        assert np.array_equal(pooled[3], pooled[0] & pooled[1])
        assert np.all(~common_images | pooled)
        closed = np.stack([flat[a].min(0) if a.any() else top
                           for a in extents])
        np.testing.assert_array_equal((flat[None] >= closed[:, None]).all(2),
                                       extents)
        strengths = np.stack([graded_score(flat, q).reshape(
            len(codes), codes.shape[1]) for q in queries])
        generators = intersect(skyline(pair_codes[sources[0]]),
                               skyline(pair_codes[sources[1]]))
        extent = downset_extent(pair_codes, generators)
        closed_generators = top[pair][None]
        for row in np.flatnonzero(extent):
            closed_generators = intersect(closed_generators,
                                           skyline(pair_codes[row]))
        np.testing.assert_array_equal(generators, closed_generators)
        np.testing.assert_array_equal(
            downset_extent(pair_codes, closed_generators), extent)
        path = root / f'contexts/class_{case_number:02d}_codexgen.npz'
        np.savez_compressed(
            path, queries=queries, closed=closed, patch_extents=site_masks,
            pooled=pooled, common_images=common_images, scores=strengths,
            sources=sources, source_sites=sites, pair=np.array(pair),
            generators=generators, downset_extent=extent)
        result = {**original, 'source_rows': sample['rows'][sources].tolist(),
                  'source_sites': sites.tolist(),
                  'source_joint_scores': common[sources, sites].tolist(),
                  'query_positive_counts': (queries > 0).sum(1).tolist(),
                  'pooled_test': pooled[:, test].sum(1).tolist(),
                  'same_site_test': common_images[:, test].sum(1).tolist(),
                  'patches_test': site_masks[:, test].sum((1, 2)).tolist(),
                  'closed_positive_counts': (closed > 0).sum(1).tolist(),
                  'downset_generators': len(generators),
                  'downset_test_images': int(extent[test].sum())}
        all_cases.append(result)
    results = {'name': name, 'selected_dimensions': len(features),
               'context_images': len(codes),
               'patches_per_image': codes.shape[1],
               'operations': ['u', 'v', 'meet', 'join'],
               'concept_scope': 'all 200 training and 100 test images',
               'cases': all_cases}
    (root / 'results/contexts_codexgen.json').write_text(
        json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=['prisma', 'saev'])
    analyze(parser.parse_args().name)
