"""Shared provenance, metrics and sparse dominance operations."""

from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scipy import sparse

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

SEED = 20260930
BUDGETS = (1, 4, 16, 64, None)
ALPHAS = (1., .75, .5, .25, .1, .05)
CHECKPOINTS = {
    'gpt2_res8': ('gpt2-small-res-jb', 'blocks.8.hook_resid_pre'),
    'gpt2_mlp8': ('gpt2-small-mlp-tm', 'blocks.8.hook_mlp_out'),
    'gemma2_l0_37': ('gemma-scope-2b-pt-res',
                     'layer_8/width_16k/average_l0_37'),
    'gemma2_l0_301': ('gemma-scope-2b-pt-res',
                      'layer_8/width_16k/average_l0_301'),
}


def sha256(path: Path | str) -> str:
    """Compute the SHA-256 digest of a file in bounded chunks."""
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 22), b''):
            value.update(chunk)
    return value.hexdigest()


def save_json(path: Path, value: Any) -> None:
    """Write a JSON value with indentation and reject nonfinite numbers."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metrics(
    y: np.ndarray,
    score: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute tie-aware retrieval metrics and optional confusion counts."""
    y = np.asarray(y, dtype=bool)
    k = min(10, len(y))
    boundary = np.partition(score, len(score) - k)[-k]
    above, tied = score > boundary, score == boundary
    p10 = (y[above].sum() + (k - above.sum()) * y[tied].mean()) / k
    result = dict(ap=float(average_precision_score(y, score)),
                  p10=float(p10), prevalence=float(y.mean()))
    if threshold is not None:
        pred = score >= threshold
        for key, mask in dict(tp=y & pred, fp=~y & pred,
                              fn=y & ~pred, tn=~y & ~pred).items():
            result[key] = int(mask.sum())
        denom = 2 * result['tp'] + result['fp'] + result['fn']
        result['f1'] = 2 * result['tp'] / denom if denom else 0.
    return result


def threshold_for(
    y: np.ndarray,
    score: np.ndarray,
    graded: bool = False,
    binary: bool = False,
) -> float:
    """Select a threshold by calibration F1 with deterministic tie breaking."""
    if binary:
        return .5
    candidates = ALPHAS if graded else np.unique(
        np.quantile(score, np.linspace(0, 1, 11)))[::-1]
    values = [metrics(y, score, t)['f1'] for t in candidates]
    return float(candidates[int(np.argmax(values))])


def full_scores(csc: sparse.csc_matrix, query: np.ndarray) -> np.ndarray:
    """Exact nonnegative dominance ratios, including the empty query."""
    active = np.flatnonzero(query > 0)
    if not len(active):
        return np.ones(csc.shape[0])
    ordered = active[np.argsort(np.diff(csc.indptr)[active], kind='stable')]
    rows = np.arange(csc.shape[0])
    values = np.full(len(rows), np.inf)
    for col in ordered:
        begin, end = csc.indptr[col:col + 2]
        found = csc.indices[begin:end]
        pos = np.searchsorted(found, rows)
        good = pos < len(found)
        good[good] &= found[pos[good]] == rows[good]
        rows, pos = rows[good], pos[good]
        values = np.minimum(values[good],
                            csc.data[begin:end][pos] / query[col])
        if not len(rows):
            break
    score = np.zeros(csc.shape[0])
    score[rows] = values
    return score


def queries(
    csc: sparse.csc_matrix,
    query: np.ndarray,
    maxima: np.ndarray,
    frequency: np.ndarray,
    ntrain: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build budgeted dominance scores and return ranked active coordinates."""
    active = np.flatnonzero(query > 0)
    full = full_scores(csc, query)
    if not len(active):
        return [full] * 5, active
    weight = query[active] / maxima[active]
    weight *= np.log((ntrain + 1) / (frequency[active] + 1))
    ordered = active[np.argsort(-weight, kind='stable')]
    first = ordered[:64]
    ratios = csc[:, first].toarray().astype(np.float64)
    ratios /= query[first]
    np.minimum.accumulate(ratios, axis=1, out=ratios)
    candidates = [ratios[:, min(k, len(first)) - 1] for k in BUDGETS[:-1]]
    return candidates + [full], ordered
