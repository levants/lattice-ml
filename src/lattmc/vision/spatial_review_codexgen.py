"""Audit frozen joins against a conditional spatial-overlap reference."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

import hashlib
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import gammaln


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'vision_tokens/overcomplete'
OUT = DATA / 'results/spatial_review'
MODELS = ['topk_k32_s0', 'batchtopk_k32_s0', 'jump_k32_s0',
          'archetypal_k32_s0', 'relu_fixed_k32_s0', 'pretrained_ra',
          'prisma_transcoder']
DATASETS = ['imagenette', 'imagewoof', 'pets', 'parts', 'dtd']


def overlap_probability(
    a: ArrayLike,
    b: ArrayLike,
    sites: int,
) -> np.ndarray | np.floating:
    """Probability of nonempty overlap of independent uniform site sets."""
    a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    result = np.ones(a.shape, dtype=float)
    valid = a + b <= sites
    n, k = sites - a[valid], b[valid]
    log_zero = (gammaln(n + 1) - gammaln(n - k + 1)
                - gammaln(sites + 1) + gammaln(sites - k + 1))
    result[valid] = -np.expm1(log_zero)
    return np.clip(result, 0, 1)


def verify_reference() -> int:
    """Check the overlap formula against exhaustive finite-set enumeration."""
    cases = 0
    for sites in range(1, 9):
        for a in range(sites + 1):
            fixed = set(range(a))
            for b in range(sites + 1):
                hits = sum(bool(fixed.intersection(s)) for s in
                           itertools.combinations(range(sites), b))
                exact = hits / comb(sites, b)
                assert abs(float(overlap_probability(a, b, sites))
                           - exact) < 1e-12
                cases += 1
    return cases


def main() -> None:
    """Audit frozen joins against a conditional spatial-overlap reference."""
    OUT.mkdir(parents=True, exist_ok=True)
    results, arrays, provenance = [], {}, []
    checks = verify_reference()
    for model in MODELS:
        query_path = DATA / f'results/{model}/queries_codexgen.json'
        definitions = json.loads(query_path.read_text())['queries']
        assert len(definitions) == 4
        for dataset in DATASETS:
            records_path = DATA / f'dataset/{dataset}_codexgen.json'
            records = json.loads(records_path.read_text())
            test = np.array([r['split'] == 'test' for r in records])
            code_path = DATA / f'codes/{model}/{dataset}_codexgen.npz'
            with np.load(code_path) as f:
                matrix = sparse.csr_matrix(
                    (f['data'], f['indices'], f['indptr']),
                    shape=f['shape'])
            sites = matrix.shape[0] // len(records)
            old_path = DATA / f'results/{model}/{dataset}_codexgen.json'
            old = json.loads(old_path.read_text())['queries']
            for q, definition in enumerate(definitions):
                z = matrix[:, definition['features']].toarray()
                z = z.reshape(len(records), sites, 2)[test]
                u, v = np.array(definition['u']), np.array(definition['v'])
                assert np.all(u > 0) and np.all(v > 0)
                previous = None
                for multiplier in [.5, 1., 1.5]:
                    threshold = np.maximum(u, v) * multiplier
                    mask = z >= threshold
                    pooled = mask.any(1).all(1)
                    common = mask.all(2).any(1)
                    intersection = ((z >= u * multiplier).all(2).any(1)
                                    & (z >= v * multiplier).all(2).any(1))
                    expected = overlap_probability(
                        mask[:, :, 0].sum(1), mask[:, :, 1].sum(1), sites)
                    assert np.all(common <= intersection)
                    assert np.all(intersection <= pooled)
                    if previous is not None:
                        assert np.all(pooled <= previous[0])
                        assert np.all(common <= previous[1])
                    previous = pooled, common
                    if multiplier == 1:
                        reference = old[q]['operations']['join']
                        assert int(pooled.sum()) == reference['pooled_test']
                        assert int(common.sum()) == reference['common_test']
                    key = f'{model}_{dataset}_{q}_{multiplier}'
                    arrays[key] = np.column_stack(
                        [pooled, common, expected, intersection])
                    results.append({
                        'model': model, 'dataset': dataset, 'pair': q,
                        'features': definition['features'],
                        'threshold_multiplier': multiplier,
                        'images': int(test.sum()), 'sites': sites,
                        'pooled': int(pooled.sum()),
                        'common': int(common.sum()),
                        'independent_expected': float(expected.sum()),
                        'separate_query_intersection':
                            int(intersection.sum())})
            provenance.append({
                'model': model, 'dataset': dataset,
                'queries_sha256': hashlib.sha256(
                    query_path.read_bytes()).hexdigest(),
                'records_sha256': hashlib.sha256(
                    records_path.read_bytes()).hexdigest(),
                'selected_codes_sha256': hashlib.sha256(
                    matrix[:, sorted({j for d in definitions
                                      for j in d['features']})]
                    .toarray().tobytes()).hexdigest()})
            print(model, dataset, 'verified', flush=True)
    output = {'protocol': 'Frozen positive training queries; test images '
              'only; no refitting. Independent uniform permutation of '
              'one threshold mask, conditional on both marginal counts.',
              'reference_exhaustive_cases': checks, 'results': results,
              'provenance': provenance}
    (OUT / 'spatial_audit_codexgen.json').write_text(
        json.dumps(output, indent=2) + '\n')
    np.savez_compressed(OUT / 'per_image_codexgen.npz', **arrays)
    print(len(results), 'query-dataset-threshold cases;', checks,
          'exhaustive reference cases', flush=True)


if __name__ == '__main__':
    main()
