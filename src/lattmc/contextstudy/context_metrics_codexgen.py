"""Audit category, contextual-granularity, and matched-depth measurements."""

from __future__ import annotations
from typing import Any
from collections.abc import Callable
from collections.abc import Mapping

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from lattmc.activationstudy.common_codexgen import save_json, sha256
from lattmc.activationstudy.families_evaluate_codexgen import interval
from .depth_codexgen import OLD, OUT

FAMILIES = ('gemma_matryoshka', 'pythia_topk', 'smol_topk', 'qwen_transcoder')
METHODS = ('graded', 'support', 'best_single', 'sae_cosine',
           'dense_cosine', 'tfidf_cosine')


def safe_precision(record: dict[str, int]) -> float:
    """Compute precision with a denominator floor for empty predictions."""
    return record['tp'] / max(1, record['tp'] + record['fp'])


def repeated(
    records: list[dict[str, Any]],
    fn: Callable[[dict[str, Any]], float],
) -> np.ndarray:
    """Compute one mean metric value per experimental repeat."""
    return np.array([np.mean([fn(r) for r in records if r['repeat'] == rep])
                     for rep in range(10)])


def independent_ap(y: np.ndarray, score: np.ndarray) -> float:
    """Compute average precision independently with grouped score ties."""
    order = np.argsort(-score, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(score) - 1]
    cumulative = np.cumsum(y[order])[ends]
    increments = np.diff(np.r_[0, cumulative])
    return float(np.sum(increments * cumulative / (ends + 1)) / y.sum())


def granular(
    records: list[dict[str, Any]],
    scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Compare strict, broad, and sibling-label retrieval interpretations."""
    results = []
    for label, sibling, title in ((7, 8, 'Geographic'), (9, 10, 'Biological')):
        values = {k: [] for k in ('strict', 'broad', 'sibling')}
        for i, task in enumerate(records):
            if task['label'] != label:
                continue
            score = scores['graded'][i]
            masks = dict(strict=np.ones(len(labels), bool),
                         broad=np.ones(len(labels), bool),
                         sibling=labels != label)
            for kind, mask in masks.items():
                y = (labels == label if kind == 'strict' else
                     np.isin(labels, (label, sibling)) if kind == 'broad'
                     else labels == sibling)
                value = average_precision_score(y[mask], score[mask])
                assert np.isclose(value, independent_ap(y[mask], score[mask]))
                values[kind].append(float(value))
        assert all(len(v) == 10 for v in values.values())
        results.append(dict(
            context=title, source_label=label, sibling_label=sibling,
            methods={k: dict(**interval(v), prevalence=(50 / 650 if
                k == 'sibling' else 100 / 700 if k == 'broad' else 50 / 700),
                lift=float(np.mean(v)) / (50 / 650 if k == 'sibling'
                     else 100 / 700 if k == 'broad' else 50 / 700))
                     for k, v in values.items()},
        ))
    return results


def main() -> dict[str, Any]:
    """Audit cached metrics and save category and depth comparisons."""
    depth, family, granularity, checks = [], [], [], 0
    hashes = {}
    for dataset in ('ag_news', 'dbpedia_14'):
        designs = json.loads((OLD / dataset / 'design.json').read_text())
        labels = np.array([r['label'] for r in designs['rows']])
        test = np.array([i for i, r in enumerate(designs['rows'])
                         if r['split'] == 'test'])
        first_scores = first_records = first_ap = None
        first_vectors = None
        for layer in (3, 15, 27):
            p = OUT / dataset / f'smol{layer}_results.json'
            report = json.loads(p.read_text())
            score_path = OUT / dataset / f'smol{layer}_scores.npz'
            assert sha256(score_path) == report['scores_sha256']
            scores = np.load(score_path)
            records = report['records']
            assert report['test_ids'] == test.tolist()
            hashes[str(p.relative_to(OUT))] = sha256(p)
            for i, row in enumerate(records):
                y = labels[test] == row['label']
                for method in METHODS:
                    value = independent_ap(y, scores[method][i])
                    assert np.isclose(value, row['metrics'][method]['ap'])
                    threshold = row['thresholds'][method]
                    pred = scores[method][i] >= threshold
                    for key, count in dict(tp=sum(pred & y), fp=sum(pred & ~y),
                                          fn=sum(~pred & y),
                                          tn=sum(~pred & ~y)).items():
                        assert row['metrics'][method][key] == count
                    checks += 1
            methods = {}
            for method in METHODS:
                methods[method] = interval(repeated(
                    records, lambda r: r['metrics'][method]['ap']))
            vectors = {m: repeated(records, lambda r: r['metrics'][m]['ap'])
                       for m in METHODS}
            if first_vectors is None:
                first_vectors = vectors
            aps = vectors['graded']
            if first_ap is None:
                first_ap, first_scores, first_records = aps, scores, records
            jaccards = []
            for i, row in enumerate(records):
                a = scores['graded'][i] >= row['thresholds']['graded']
                b = first_scores['graded'][i] >= (
                    first_records[i]['thresholds']['graded'])
                union = int((a | b).sum())
                jaccards.append(float((a & b).sum() / union) if union else 1.)
                assert row['low_ids'] == first_records[i]['low_ids']
                assert row['positive'] == first_records[i]['positive']
            low = [r for r in records if 'graded' in r['low']]
            result = dict(
                dataset=dataset, layer=layer, methods=methods,
                delta_vs_3=interval(aps - first_ap),
                method_delta_vs_3={m: interval(vectors[m] - first_vectors[m])
                                   for m in METHODS},
                low_ap=float(np.mean([r['low']['graded']['ap'] for r in low])),
                low_tfidf=float(np.mean([r['low']['tfidf_cosine']['ap']
                                        for r in low])),
                low_prevalence=float(np.mean([r['low']['graded']['prevalence']
                                              for r in low])),
                low_valid_tasks=len(low),
                precision=float(np.mean([safe_precision(r['metrics']['graded'])
                                         for r in records])),
                recall=float(np.mean([r['metrics']['graded']['tp'] /
                    sum(r['metrics']['graded'][k] for k in ('tp', 'fn'))
                    for r in records])),
                extent=float(np.mean([r['metrics']['graded']['tp'] +
                                      r['metrics']['graded']['fp']
                                      for r in records])),
                extent_jaccard_vs_3=float(np.mean(jaccards)),
            )
            depth.append(result)
            if dataset == 'dbpedia_14':
                granularity.append(dict(model=f'SmolLM2/{layer}',
                    values=granular(records, scores, labels[test])))
        for name in FAMILIES:
            p = OLD / dataset / f'{name}_max_results.json'
            report = json.loads(p.read_text())
            ids = [i for i, r in enumerate(report['records'])
                   if r['shot'] == 3]
            records = [report['records'][i] for i in ids]
            scores = np.load(OLD / dataset / f'{name}_max_scores.npz')
            low = [r for r in records if 'graded' in r['low_overlap']]
            family.append(dict(
                dataset=dataset, model=name,
                ap=interval(repeated(records, lambda r:
                                     r['metrics']['graded']['ap'])),
                low_ap=float(np.mean([r['low_overlap']['graded']['ap']
                                      for r in low])),
                low_tfidf=float(np.mean([r['low_overlap']['tfidf_cosine']['ap']
                                         for r in low])),
                low_prevalence=float(np.mean([
                    r['low_overlap']['graded']['prevalence'] for r in low])),
                low_valid_tasks=len(low),
            ))
            if dataset == 'dbpedia_14':
                granularity.append(dict(model=name, values=granular(
                    records, {k: scores[k][ids] for k in scores.files},
                    labels[test])))
    result = dict(depth=depth, family=family, granularity=granularity,
                  independent_metric_checks=checks, hashes=hashes,
                  source_sha256=sha256(__file__))
    save_json(OUT / 'context_metrics.json', result)
    print('Independent score checks:', checks)
    for d in depth:
        print(d['dataset'], d['layer'], 'AP', d['methods']['graded'],
              'delta', d['delta_vs_3'], 'low', d['low_ap'])
    return result


if __name__ == '__main__':
    main()
