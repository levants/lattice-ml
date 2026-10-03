"""Independently audit saved scores, split isolation, and provenance."""

from __future__ import annotations
from typing import Any

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .common_codexgen import save_json, sha256


def audit(output: Path, protocol: Path) -> dict[str, Any]:
    """Verify protocol hashes, cached artifacts, and evaluation records."""
    checks, tasks, predictions = 0, 0, 0
    for target in sorted(output.iterdir()):
        if not (target / 'design.json').exists():
            continue
        design = json.loads((target / 'design.json').read_text())
        assert design['protocol_sha256'] == sha256(protocol)
        rows = design['rows']
        labels = np.array([r['label'] for r in rows])
        train = {i for i, r in enumerate(rows) if r['split'] == 'train'}
        tests = [i for i, r in enumerate(rows) if r['split'] == 'test']
        for path in sorted(target.glob('*_results.json')):
            data = json.loads(path.read_text())
            name, pooling = data['checkpoint'], data['pooling']
            score_path = target / f'{name}_{pooling}_scores.npz'
            assert data['scores_sha256'] == sha256(score_path)
            assert data['design_sha256'] == sha256(target / 'design.json')
            assert data['protocol_sha256'] == sha256(protocol)
            assert data['source_sha256'] == sha256(
                Path(__file__).with_name('families_evaluate_codexgen.py'))
            assert data['test_ids'] == tests
            extraction_path = target / f'{name}_extraction.json'
            assert data['extraction_sha256'] == sha256(extraction_path)
            extraction = json.loads(extraction_path.read_text())
            assert extraction['source_sha256'] == sha256(
                Path(__file__).with_name('families_extract_codexgen.py'))
            for name, value in extraction['companion_hashes'].items():
                assert sha256(Path(__file__).with_name(name)) == value
            for item in extraction['files'].values():
                assert sha256(target / item['name']) == item['sha256']
            scores = np.load(score_path)
            for index, record in enumerate(data['records']):
                tasks += 1
                assert set(record['positive']).issubset(train)
                assert set(record['negative']).issubset(train)
                assert set(record['random']).issubset(train)
                assert all(labels[i] == record['label']
                           for i in record['positive'])
                assert all(labels[i] != record['label']
                           for i in record['negative'])
                y = labels[tests] == record['label']
                for method in data['methods']:
                    score = scores[method][index]
                    assert np.isfinite(score).all()
                    # Independent AP: sum precision at each distinct score,
                    # weighted by its number of positive examples.
                    order = np.argsort(-score, kind='stable')
                    ranked = y[order]
                    ends = np.r_[np.flatnonzero(
                        np.diff(score[order]) != 0), len(score) - 1]
                    cumulative = np.cumsum(ranked)[ends]
                    increments = np.diff(np.r_[0, cumulative])
                    ap = np.sum(increments * cumulative / (ends + 1))
                    ap /= y.sum()
                    expected = record['metrics'][method]
                    assert np.isclose(ap, expected['ap'], atol=1e-12)
                    pred = score >= record['thresholds'][method]
                    observed = dict(tp=int(sum(y & pred)),
                                    fp=int(sum(~y & pred)),
                                    fn=int(sum(y & ~pred)),
                                    tn=int(sum(~y & ~pred)))
                    assert all(expected[k] == v
                               for k, v in observed.items())
                    checks += 1
                    predictions += len(y)
            for shot in (3, 10):
                subset = [r for r in data['records'] if r['shot'] == shot]
                for method in data['methods']:
                    mean = np.mean([r['metrics'][method]['ap']
                                    for r in subset])
                    reported = data['summary'][str(shot)]['methods'][method]
                    assert np.isclose(mean, reported['mean'], atol=1e-12)
    prefix_checks = 0
    for dataset in ('ag_news', 'dbpedia_14'):
        target = output / dataset
        for pooling in ('max', 'mean'):
            full = sparse.load_npz(
                target / f'gemma_matryoshka_{pooling}.npz')
            for width in (512, 2048):
                prefix = sparse.load_npz(
                    target / f'gemma_matryoshka_{width}_{pooling}.npz')
                assert (prefix != full[:, :width]).nnz == 0
                prefix_checks += 1
    assert tasks == 4320, tasks
    result = dict(query_configurations=tasks, metric_vectors=checks,
                  test_score_comparisons=predictions, status='passed',
                  exact_prefix_checks=prefix_checks,
                  protocol_sha256=sha256(protocol))
    save_json(output / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.protocol), indent=2))
