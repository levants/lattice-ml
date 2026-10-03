"""Evaluate every frozen retrieval condition without fitting to test rows."""

from __future__ import annotations
from typing import Any
from collections.abc import Sequence

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .common_codexgen import (
    BUDGETS, CHECKPOINTS, SEED, metrics, queries, save_json, sha256,
    threshold_for,
)


def probe(
    features: np.ndarray | sparse.spmatrix,
    positive: np.ndarray,
    negative: np.ndarray,
    calibration: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Use fixed few-shot labels; choose regularization on calibration AP."""
    sources = np.r_[positive, negative]
    models, aps = [], []
    for c in (.01, .1, 1., 10.):
        model = LogisticRegression(
            C=c, solver='liblinear', dual=True, max_iter=1000,
            random_state=SEED,
        ).fit(features[sources], y[sources])
        models.append(model)
        aps.append(average_precision_score(
            y[calibration], model.decision_function(features[calibration])))
    chosen = int(np.argmax(aps))
    return models[chosen].decision_function(features), models[chosen].C


def cosine(
    features: np.ndarray | sparse.spmatrix,
    positive: np.ndarray,
) -> np.ndarray:
    """Score rows against the normalized positive-example centroid."""
    centroid = np.asarray(features[positive].mean(axis=0)).reshape(1, -1)
    centroid = normalize(centroid).ravel()
    return np.asarray(features @ centroid).ravel()


def interval(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Estimate the mean and a seeded bootstrap confidence interval."""
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, len(values), (2000, len(values)))
    low, high = np.quantile(np.asarray(values)[ids].mean(axis=1), [.025, .975])
    return dict(mean=float(np.mean(values)), low=float(low), high=float(high))


def low_overlap_summary(
    records: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    """Summarize AP and prevalence for valid low-overlap tasks."""
    values = [r['low_overlap'][method] for r in records
              if method in r['low_overlap']]
    return dict(valid_tasks=len(values),
                ap=(float(np.mean([v['ap'] for v in values]))
                    if values else None),
                prevalence=float(np.mean([v['prevalence'] for v in values]))
                if values else None)


def summarize(
    records: list[dict[str, Any]],
    methods: Sequence[str],
) -> dict[str, Any]:
    """Aggregate repeated retrieval results by shot count and method."""
    result = {}
    for shot in (3, 10):
        selected = [r for r in records if r['shot'] == shot]
        values = {}
        for method in methods:
            repeated = np.array([
                np.mean([r['metrics'][method]['ap'] for r in selected
                         if r['repeat'] == repeat]) for repeat in range(10)])
            values[method] = repeated
        result[str(shot)] = dict(
            methods={method: interval(value)
                     for method, value in values.items()},
            paired_graded={method: interval(values['graded'] - value)
                           for method, value in values.items()
                           if method != 'graded'},
            confusion={method: {key: sum(r['metrics'][method][key]
                         for r in selected)
                         for key in ('tp', 'fp', 'fn', 'tn')}
                       for method in methods},
            full_empty=sum(r['metrics']['full']['tp']
                           + r['metrics']['full']['fp'] == 0
                           for r in selected),
            zero_queries=sum(r['active_coordinates'] == 0 for r in selected),
            budgets=dict(Counter(str(r['budget']) for r in selected)),
            half_test_ap={method: [float(np.mean([
                r['half_test_ap'][method][i] for r in selected]))
                for i in range(3)] for method in methods},
            low_overlap={method: low_overlap_summary(selected, method)
                         for method in methods},
        )
    return result


def evaluate(output: Path, protocol: Path) -> None:
    """Evaluate frozen retrieval conditions and save scores and summaries."""
    started = time.time()
    for dataset in ('ag_news', 'dbpedia_14'):
        target = output / dataset
        design = json.loads((target / 'design.json').read_text())
        assert design['protocol_sha256'] == sha256(protocol)
        texts = json.loads((target / 'texts.local.json').read_text())
        labels = np.array([row['label'] for row in design['rows']])
        train, cal, test = [np.array([i for i, r in enumerate(design['rows'])
                           if r['split'] == split])
                            for split in ('train', 'calibration', 'test')]
        classes = sorted(set(labels))
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            max_features=30000)
        vectorizer.fit([texts[i] for i in train])
        text_features = vectorizer.transform(texts)
        # Saved IDF and vocabulary permit independent preprocessing audits.
        save_json(target / 'tfidf_vocabulary.json',
                  {k: int(v) for k, v in vectorizer.vocabulary_.items()})
        np.save(target / 'tfidf_idf.npy', vectorizer.idf_)
        subsets = []
        for rep in range(3):
            rng = np.random.default_rng(SEED + rep)
            subsets.append(np.sort(np.concatenate([
                rng.choice(np.flatnonzero(labels[test] == c),
                           sum(labels[test] == c) // 2, replace=False)
                for c in classes])))
        tasks = []
        for label in classes:
            y = labels == label
            for repeat in range(10):
                rng = np.random.default_rng(SEED + 1000 * label + repeat)
                positive = rng.choice(train[y[train]], 10, replace=False)
                negative = rng.choice(train[~y[train]], 10, replace=False)
                random = rng.choice(train, 10, replace=False)
                for shot in (3, 10):
                    pos, neg = positive[:shot], negative[:shot]
                    scores = dict(tfidf_cosine=cosine(text_features, pos))
                    scores['tfidf_probe'], c = probe(
                        text_features, pos, neg, cal, y)
                    tasks.append(dict(
                        label=int(label), repeat=repeat, shot=shot,
                        positive=pos, negative=neg, random=random[:shot],
                        base_scores=scores, tfidf_c=c,
                    ))
        for name in CHECKPOINTS:
            extraction = json.loads(
                (target / f'{name}_extraction.json').read_text())
            assert extraction['design_sha256'] == sha256(
                target / 'design.json')
            dense = normalize(np.load(target / f'{name}_dense.npz')['values'])
            dense_scores = []
            for task in tasks:
                scores = dict(dense_cosine=cosine(dense, task['positive']))
                scores['dense_probe'], c = probe(
                    dense, task['positive'], task['negative'], cal,
                    labels == task['label'])
                dense_scores.append((scores, c))
            for pooling in ('max', 'mean'):
                result_path = target / f'{name}_{pooling}_results.json'
                feature_path = target / f'{name}_{pooling}.npz'
                if result_path.exists():
                    old = json.loads(result_path.read_text())
                    assert old['source_sha256'] == sha256(__file__)
                    print('Already evaluated', dataset, name, pooling,
                          flush=True)
                    continue
                assert sha256(feature_path) == (
                    extraction['files'][pooling]['sha256'])
                matrix = sparse.load_npz(feature_path).astype(np.float64)
                csc = matrix.tocsc()
                csc.sort_indices()
                normalized = normalize(matrix)
                maxima = matrix[train].max(axis=0).toarray().ravel()
                frequency = np.asarray((matrix[train] > 0).sum(axis=0)).ravel()
                records, saved_scores = [], {}
                for index, task in enumerate(tasks):
                    y = labels == task['label']
                    query = matrix[task['positive']].toarray().min(axis=0)
                    candidates, ordered = queries(
                        csc, query, maxima, frequency, len(train))
                    aps = [average_precision_score(y[cal], score[cal])
                           for score in candidates]
                    chosen = int(np.argmax(aps))
                    binary_aps = [average_precision_score(
                        y[cal], score[cal] > 0)
                                  for score in candidates]
                    support_choice = int(np.argmax(binary_aps))
                    scores = dict(
                        graded=candidates[chosen],
                        support=(candidates[support_choice] > 0).astype(float),
                        matched=(candidates[chosen] > 0).astype(float),
                        single=candidates[0], full=candidates[-1],
                        sae_cosine=cosine(normalized, task['positive']),
                        **task['base_scores'], **dense_scores[index][0],
                    )
                    if len(ordered):
                        first = ordered[:64]
                        single_scores = csc[:, first].toarray() / query[first]
                        single_aps = [average_precision_score(y[cal], values)
                                      for values in single_scores[cal].T]
                        best = int(np.argmax(single_aps))
                        scores['best_single'] = single_scores[:, best]
                        best_feature = int(first[best])
                    else:
                        scores['best_single'] = candidates[0]
                        best_feature = None
                    scores['sae_probe'], sae_c = probe(
                        normalized, task['positive'], task['negative'], cal, y)
                    random_query = matrix[task['random']].toarray().min(axis=0)
                    random_scores, _ = queries(
                        csc, random_query, maxima, frequency, len(train))
                    random_aps = [average_precision_score(y[cal], score[cal])
                                  for score in random_scores]
                    scores['random'] = random_scores[
                        int(np.argmax(random_aps))]
                    low = scores['tfidf_cosine'][test] <= np.median(
                        scores['tfidf_cosine'][test])
                    record = {key: task[key]
                              for key in ('label', 'repeat', 'shot')}
                    record.update(
                        positive=task['positive'].tolist(),
                        negative=task['negative'].tolist(),
                        random=task['random'].tolist(),
                        active_coordinates=len(ordered),
                        budget=BUDGETS[chosen],
                        selected_coordinates=(
                            ordered[:BUDGETS[chosen]].tolist()),
                        best_single_feature=best_feature,
                        sae_c=sae_c, dense_c=dense_scores[index][1],
                        tfidf_c=task['tfidf_c'], metrics={},
                        thresholds={}, half_test_ap={}, low_overlap={},
                    )
                    for method, score in scores.items():
                        threshold = threshold_for(
                            y[cal], score[cal],
                            graded=method in ('graded', 'single',
                                              'best_single',
                                              'full', 'random'),
                            binary=method in ('support', 'matched'))
                        test_scores = score[test]
                        record['thresholds'][method] = threshold
                        record['metrics'][method] = metrics(
                            y[test], test_scores, threshold)
                        record['half_test_ap'][method] = [float(
                            average_precision_score(y[test][ids],
                                                    test_scores[ids]))
                            for ids in subsets]
                        if y[test][low].any() and (~y[test][low]).any():
                            record['low_overlap'][method] = metrics(
                                y[test][low], test_scores[low])
                        saved_scores.setdefault(method, []).append(test_scores)
                    records.append(record)
                methods = list(saved_scores)
                score_path = target / f'{name}_{pooling}_scores.npz'
                np.savez_compressed(score_path,
                                    **{k: np.array(v)
                                       for k, v in saved_scores.items()})
                report = dict(
                    dataset=dataset, checkpoint=name, pooling=pooling,
                    methods=methods, records=records,
                    summary=summarize(records, methods),
                    test_ids=test.tolist(),
                    half_test_ids=[s.tolist() for s in subsets],
                    design_sha256=sha256(target / 'design.json'),
                    extraction_sha256=sha256(
                        target / f'{name}_extraction.json'),
                    feature_sha256=sha256(feature_path),
                    scores_sha256=sha256(score_path),
                    protocol_sha256=sha256(protocol),
                    source_sha256=sha256(__file__),
                )
                save_json(result_path, report)
                print(dataset, name, pooling, 'tasks', len(records),
                      'seconds', round(time.time() - started), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        evaluate(args.output, args.protocol)
