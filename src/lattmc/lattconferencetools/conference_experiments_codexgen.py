"""Execute CONFERENCE_PROTOCOL.md using only existing local caches."""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

from .paths_codexgen import (
    PAPER, repository, CACHE, RELEASE, PROTOCOL, NOTEBOOKS, TEMPLATES,
    ORGANIZATION,
)

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

HERE = PAPER
ROOT = repository()
OUT = CACHE / 'conference_results'
SEED = 20260924
BUDGETS = (1, 4, 16, 64, None)
ALPHAS = (1., .75, .5, .25, .1, .05)
TOKENS = ROOT / ('notebooks/transcoders/data/transcoders/gpt2/'
                 'owt_tokens/owt_tokens_torch.pt')
METHODS = ('graded', 'support', 'matched_support', 'full_graded',
           'cosine', 'tfidf', 'random_source')


def sha256(path: Path | str) -> str:
    """Compute the SHA-256 digest of an experiment input."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 23), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    """Write an indented JSON artifact with finite numeric values."""
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def split_tokens(tokens: np.ndarray) -> tuple[dict[str, list[int]], int]:
    """Split by hash intervals, keeping identical token rows together."""
    names = ('train', 'calibration', 'test')
    groups = defaultdict(list)
    for row, values in enumerate(tokens):
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        groups[digest].append(row)
    splits = {name: [] for name in names}
    for digest, rows in groups.items():
        salt = f'{SEED}:{digest}'.encode()
        number = int(hashlib.sha256(salt).hexdigest()[:16], 16)
        fraction = number / 2 ** 64
        name = names[0 if fraction < .6 else 1 if fraction < .8 else 2]
        splits[name].extend(rows)
    return {key: sorted(value) for key, value in splits.items()}, len(groups)


def phrases(text: str) -> set[str]:
    """Whitespace-adjacent full words; no joining across punctuation."""
    text = text.lower()
    words = list(re.finditer(r'\b[a-z]{3,}\b', text))
    found = set()
    for left, right in zip(words, words[1:]):
        if not text[left.end():right.start()].isspace():
            continue
        pair = (left.group(), right.group())
        if not any(word in ENGLISH_STOP_WORDS for word in pair):
            found.add(' '.join(pair))
    return found


def prepare() -> None:
    """Prepare deterministic corpus splits and experiment inputs."""
    import torch
    from transformers import AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    tokens = torch.load(TOKENS, weights_only=True,
                        map_location='cpu').numpy()
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    splits, unique = split_tokens(tokens)
    postings = defaultdict(list)
    for row, text in enumerate(texts):
        for phrase in phrases(text):
            postings[phrase].append(row)
    split_sets = {key: set(value) for key, value in splits.items()}
    eligible = {}
    for phrase in sorted(postings):
        rows = postings[phrase]
        if not 50 <= len(rows) <= 400:
            continue
        counts = {key: len(set(rows) & ids)
                  for key, ids in split_sets.items()}
        if (counts['train'] >= 15 and counts['calibration'] >= 5
                and counts['test'] >= 5):
            eligible[phrase] = counts
    if not eligible:
        raise ValueError('No eligible tasks: protocol cannot be executed.')
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(sorted(eligible), min(32, len(eligible)),
                               replace=False).tolist())
    tasks = []
    for phrase in chosen:
        positive = sorted(set(postings[phrase]) & split_sets['train'])
        for repeat in range(5):
            tasks.append(dict(
                phrase=phrase, repeat=repeat, positives=postings[phrase],
                sources=rng.choice(positive, 3, replace=False).tolist(),
                random_sources=rng.choice(splits['train'], 3,
                                          replace=False).tolist(),
            ))
    manifest = dict(
        protocol_sha256=sha256(PROTOCOL),
        source_sha256=sha256(__file__), token_sha256=sha256(TOKENS),
        seed=SEED, rows=len(tokens), unique_token_blocks=unique,
        splits=splits, eligible=eligible, chosen=chosen, tasks=tasks,
        versions={name: metadata.version(name) for name in
                  ('numpy', 'scipy', 'scikit-learn', 'torch',
                   'transformers', 'tokenizers')},
    )
    save_json(OUT / 'design.json', manifest)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=100000,
        sublinear_tf=True, dtype=np.float64,
    )
    train_texts = [texts[row] for row in splits['train']]
    vectorizer.fit(train_texts)
    features = vectorizer.transform(texts)
    scores = []
    for task in tasks:
        centroid = np.asarray(features[task['sources']].mean(axis=0))
        centroid = normalize(centroid).ravel()
        scores.append(np.asarray(features @ centroid).ravel())
    np.savez_compressed(OUT / 'tfidf_scores.npz', scores=np.array(scores))
    save_json(OUT / 'tfidf_vocabulary.json',
              {word: int(col) for word, col in vectorizer.vocabulary_.items()})
    np.savez_compressed(OUT / 'tfidf_idf.npz', idf=vectorizer.idf_)
    print(json.dumps(dict(
        rows=len(tokens), unique=unique, eligible=len(eligible),
        phrases=chosen, splits={k: len(v) for k, v in splits.items()},
        vocabulary=len(vectorizer.vocabulary_),
    )), flush=True)


def expected_precision(
    y: np.ndarray,
    scores: np.ndarray,
    k: int = 10,
) -> float:
    """Average P@k over uniform permutations within the cutoff tie group."""
    k = min(k, len(y))
    boundary = np.partition(scores, len(scores) - k)[-k]
    above = scores > boundary
    tied = scores == boundary
    remainder = k - int(above.sum())
    hits = float(y[above].sum()) + remainder * float(y[tied].mean())
    return hits / k


def statistics(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute AP, tie-aware precision, and optional confusion counts."""
    result = dict(ap=float(average_precision_score(y, scores)),
                  p10=expected_precision(y, scores),
                  prevalence=float(y.mean()))
    if threshold is not None:
        pred = scores >= threshold
        tp = int(np.sum(y & pred))
        fp = int(np.sum(~y & pred))
        fn = int(np.sum(y & ~pred))
        tn = int(np.sum(~y & ~pred))
        denom = 2 * tp + fp + fn
        result.update(tp=tp, fp=fp, fn=fn, tn=tn,
                      f1=2 * tp / denom if denom else 0.)
    return result


def full_scores(csc: sparse.csc_matrix, query: np.ndarray) -> np.ndarray:
    """Exact dominance scores, intersecting sparse postings first."""
    active = np.flatnonzero(query > 0)
    if not len(active):
        return np.ones(csc.shape[0])
    sizes = np.diff(csc.indptr)[active]
    ordered = active[np.argsort(sizes, kind='stable')]
    candidates = np.arange(csc.shape[0])
    values = np.full(len(candidates), np.inf)
    for col in ordered:
        begin, end = csc.indptr[col:col + 2]
        rows = csc.indices[begin:end]
        pos = np.searchsorted(rows, candidates)
        good = pos < len(rows)
        good[good] &= rows[pos[good]] == candidates[good]
        candidates = candidates[good]
        pos = pos[good]
        values = np.minimum(values[good], csc.data[begin:end][pos]
                            .astype(np.float64) / query[col])
        if not len(candidates):
            break
    result = np.zeros(csc.shape[0])
    result[candidates] = values
    return result


def query_scores(
    csc: sparse.csc_matrix,
    query: np.ndarray,
    maxima: np.ndarray,
    frequency: np.ndarray,
    ntrain: int,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    """Build budgeted scores, query sizes, and coordinate rankings."""
    active = np.flatnonzero(query > 0)
    weight = query[active] / maxima[active]
    weight *= np.log((ntrain + 1) / (frequency[active] + 1))
    ordered = active[np.argsort(-weight, kind='stable')]
    full = full_scores(csc, query)
    if not len(active):
        return [full] * len(BUDGETS), [0] * len(BUDGETS), ordered
    first = ordered[:64]
    ratios = csc[:, first].toarray().astype(np.float64)
    ratios /= query[first]
    np.minimum.accumulate(ratios, axis=1, out=ratios)
    scores = [ratios[:, min(k, len(first)) - 1] for k in BUDGETS[:-1]]
    sizes = [min(k, len(active)) for k in BUDGETS[:-1]]
    return scores + [full], sizes + [len(active)], ordered


def choose(
    scores: Sequence[np.ndarray],
    ids: np.ndarray,
    y: np.ndarray,
    binary: bool = False,
) -> int:
    """Choose the candidate with the highest calibration AP."""
    values = [s > 0 if binary else s for s in scores]
    aps = [average_precision_score(y[ids], s[ids]) for s in values]
    return int(np.argmax(aps))


def alpha_for(scores: np.ndarray, ids: np.ndarray, y: np.ndarray) -> float:
    """Select the calibrated dominance threshold by F1."""
    values = [statistics(y[ids], scores[ids], alpha)['f1']
              for alpha in ALPHAS]
    return ALPHAS[int(np.argmax(values))]


def run(kind: str, layer: int) -> None:
    """Evaluate and save retrieval results for one checkpoint layer."""
    started = time.time()
    design = json.loads((OUT / 'design.json').read_text())
    if design['protocol_sha256'] != sha256(PROTOCOL):
        raise ValueError('Protocol changed after dataset preparation.')
    folder = 'sae' if kind == 'sae' else 'transcoders'
    cache = ROOT / f'notebooks/{folder}/data/{folder}/gpt2/V{layer}.npz'
    matrix = sparse.load_npz(cache).tocsr()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError('Expected finite, nonnegative activations.')
    csc = matrix.tocsc()
    train, cal, test = (np.array(design['splits'][key])
                        for key in ('train', 'calibration', 'test'))
    training = matrix[train]
    maxima = training.max(axis=0).toarray().ravel().astype(np.float64)
    frequency = training.getnnz(axis=0)
    del training
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
    tfidf = np.load(OUT / 'tfidf_scores.npz')['scores']
    output, saved_scores, saved_queries = [], {}, {}
    for number, task in enumerate(design['tasks']):
        y = np.zeros(matrix.shape[0], dtype=bool)
        y[task['positives']] = True
        seed_rows = matrix[task['sources']].toarray().astype(np.float64)
        query = seed_rows.min(axis=0)
        scores, sizes, ordered = query_scores(
            csc, query, maxima, frequency, len(train))
        selected = choose(scores, cal, y)
        binary = choose(scores, cal, y, binary=True)
        alpha = alpha_for(scores[selected], cal, y)
        centroid = seed_rows.mean(axis=0)
        cosine = np.asarray(matrix @ centroid).ravel()
        denom = norms * np.linalg.norm(centroid)
        cosine = np.divide(cosine, denom, out=np.zeros_like(cosine),
                           where=denom > 0)
        random_query = matrix[task['random_sources']].toarray().min(axis=0)
        random_scores, random_sizes, random_ordered = query_scores(
            csc, random_query.astype(np.float64), maxima, frequency,
            len(train))
        random_selected = choose(random_scores, cal, y)
        random_alpha = alpha_for(random_scores[random_selected], cal, y)
        candidates = dict(
            graded=(scores[selected], alpha),
            support=((scores[binary] > 0).astype(float), .5),
            matched_support=((scores[selected] > 0).astype(float), .5),
            full_graded=(scores[-1], alpha_for(scores[-1], cal, y)),
            cosine=(cosine, None), tfidf=(tfidf[number], None),
            random_source=(random_scores[random_selected], random_alpha),
        )
        results = {}
        for method, (score, threshold) in candidates.items():
            results[method] = statistics(y[test], score[test], threshold)
            saved_scores[f'{number}_{method}'] = score[test]
        saved_queries[f'{number}_full'] = query
        saved_queries[f'{number}_order'] = ordered
        saved_queries[f'{number}_random'] = random_query
        saved_queries[f'{number}_random_order'] = random_ordered
        output.append(dict(
            phrase=task['phrase'], repeat=task['repeat'],
            selected_budget=BUDGETS[selected], selected_size=sizes[selected],
            support_budget=BUDGETS[binary], support_size=sizes[binary],
            alpha=alpha, full_alpha=candidates['full_graded'][1],
            random_budget=BUDGETS[random_selected],
            random_size=random_sizes[random_selected],
            random_alpha=random_alpha, full_size=int(np.sum(query > 0)),
            metrics=results,
        ))
        if number % 20 == 0:
            print(kind, layer, number, f'{time.time()-started:.1f}s',
                  flush=True)
    stem = f'{kind}_{layer}'
    np.savez_compressed(OUT / f'{stem}_scores.npz', **saved_scores)
    np.savez_compressed(OUT / f'{stem}_queries.npz', **saved_queries)
    save_json(OUT / f'{stem}.json', dict(
        model=kind, layer=layer, cache_sha256=sha256(cache),
        source_sha256=sha256(__file__),
        design_sha256=sha256(OUT / 'design.json'),
        elapsed_seconds=time.time() - started, runs=output,
    ))
    print(stem, 'complete', time.time() - started, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--kind', choices=('sae', 'tc'))
    parser.add_argument('--layer', type=int, choices=(0, 8, 11))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.action == 'prepare':
            prepare()
        elif args.kind is None or args.layer is None:
            parser.error('run requires --kind and --layer')
        else:
            run(args.kind, args.layer)
