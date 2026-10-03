"""Reproducible graded activation queries against cached GPT-2 summaries.

Run one model/layer at a time using the repository's existing uv Python.
The source notebooks supply the corpus rows, positions, models, and hooks.
This extension uses full token codes, without top-feature or floor reduction.
"""

from __future__ import annotations

from typing import Any

from .paths_codexgen import (
    PAPER, repository, CACHE, RELEASE, PROTOCOL, NOTEBOOKS, TEMPLATES,
    ORGANIZATION,
)

import argparse
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/lattcontexts-matplotlib')

import numpy as np
from scipy import sparse
import torch

ROOT = repository()
OUT = CACHE / 'activation_results'
sys.path.insert(0, str(ROOT))

CASES = {
    'nyc': (3457, [1, 2, 3], [r'\bnew\s+york\s+city\b']),
    'animals': (4042, [8, 82], [r'\bcats?\b', r'\bdogs?\b']),
    'rio': (5411, [15, 16, 17], [r'\brio\s+de\s+janeiro\b']),
    'richmond': (117, [30, 31], [r'\brichmond\s+hill\b']),
    'park': (21481, [24, 25, 26], [r'\bnational\s+park\s+service\b']),
    'sports': (1924, [15, 39, 94],
               [r'\bcleveland\b', r'\bclippers\b', r'\bcavaliers\b']),
}
ALPHAS = [0.25, 0.5, 0.75, 1.0]


def file_hash(path: Path | str) -> str:
    """Compute the SHA-256 digest of a cached artifact."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def coincidence(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, int | float | None]:
    """A 2 by 2 coincidence table on the shared corpus row universe."""
    both = int(np.sum(left & right))
    only_left = int(np.sum(left & ~right))
    only_right = int(np.sum(~left & right))
    neither = int(np.sum(~left & ~right))
    union = both + only_left + only_right
    return dict(
        both=both, only_left=only_left, only_right=only_right,
        neither=neither, jaccard=both / union if union else None,
    )


def extent(
    matrix: sparse.csc_matrix,
    query: np.ndarray,
    alpha: float | None = 1.0,
) -> np.ndarray:
    """Exact coordinatewise dominance; None means positive-support only.

    CSC postings avoid densifying the complete 25,600 by 24,576 matrix.
    Query thresholds are evaluated in float64, without a comparison epsilon.
    """
    active = np.flatnonzero(query > 0)
    candidates = np.ones(matrix.shape[0], dtype=bool)
    sizes = np.diff(matrix.indptr)[active]
    for col in active[np.argsort(sizes, kind='stable')]:
        start, stop = matrix.indptr[col:col + 2]
        rows = matrix.indices[start:stop]
        values = matrix.data[start:stop].astype(np.float64)
        threshold = 0.0 if alpha is None else float(query[col]) * alpha
        keep = values > 0 if alpha is None else values >= threshold
        column_mask = np.zeros(matrix.shape[0], dtype=bool)
        column_mask[rows[keep]] = True
        candidates &= column_mask
        if not candidates.any():
            break
    return candidates


def intent(
    matrix: sparse.csr_matrix,
    mask: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
) -> np.ndarray:
    """The exact sequence-summary meet, with the declared empty meet."""
    rows = np.flatnonzero(mask)
    if not len(rows):
        return upper.copy()
    if len(rows) == matrix.shape[0]:
        return lower.copy()
    result = upper.copy()
    for start in range(0, len(rows), 128):
        block = matrix[rows[start:start + 128]].toarray()
        result = np.minimum(result, block.min(axis=0))
    return result


def run(kind: str, layer: int, output: Path = OUT) -> dict[str, Any]:
    """Execute all six preselected cases for one model and layer."""
    from src.lattmc.tc.transcoder_analyzers_codexgen import (
        init_transcoder_or_sae,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    token_path = ROOT / 'notebooks/transcoders/data/transcoders/gpt2'
    token_path /= 'owt_tokens/owt_tokens_torch.pt'
    tokens = torch.load(token_path, map_location='cpu', weights_only=True)
    folder = ('sae/data/sae' if kind == 'sae'
              else 'transcoders/data/transcoders')
    matrix_path = ROOT / 'notebooks' / folder / 'gpt2' / f'V{layer}.npz'
    model = init_transcoder_or_sae(
        model_name='gpt2-small', layers=[layer], device=torch.device('cpu'),
        tr_or_sae=kind == 'tc',
    )
    tokenizer = model.tokenizer
    targets, pooled, labels = {}, {}, {}
    for name, (row, positions, _) in CASES.items():
        values = model.run_layers(tokens[row], [layer])[layer]
        assert values.shape == (128, 24576), values.shape
        assert np.isfinite(values).all() and (values >= 0).all()
        targets[name] = values[positions].copy()
        pooled[row] = values.max(axis=0)
        labels[name] = [tokenizer.decode([int(tokens[row, p])])
                        for p in positions]
        print(name, row, positions, labels[name], flush=True)
    del values, model
    gc.collect()
    texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    del tokenizer
    gc.collect()

    cached = sparse.load_npz(matrix_path).tocsr()
    assert cached.shape == (25600, 24576)
    assert np.isfinite(cached.data).all() and (cached.data >= 0).all()
    corrections = []
    parts = []
    start = 0
    # Refresh only the six source summaries in memory. This prevents tiny
    # cross-version roundoff from making a source fail its own exact query.
    # Cached files are never overwritten; every change is measured below.
    for row in sorted(pooled):
        old = cached.getrow(row).toarray().ravel()
        fresh = pooled[row]
        assert np.allclose(old, fresh, atol=1e-3, rtol=1e-3)
        corrections.append(dict(
            row=row, max_abs_change=float(np.max(np.abs(old - fresh))),
            changed_coordinates=int(np.sum(old != fresh)),
            changed_support=int(np.sum((old > 0) != (fresh > 0))),
        ))
        parts.extend([cached[start:row], sparse.csr_matrix(fresh[None, :])])
        start = row + 1
    parts.append(cached[start:])
    matrix = sparse.vstack(parts, format='csc')
    del cached, parts, pooled
    gc.collect()
    matrix.eliminate_zeros()
    row_matrix = matrix.tocsr()
    upper = matrix.max(axis=0).toarray().ravel()
    lower = matrix.min(axis=0).toarray().ravel()
    records, arrays, case_info = [], {}, []
    for name, (row, positions, patterns) in CASES.items():
        lexical = np.array([
            all(re.search(p, text, re.IGNORECASE) for p in patterns)
            for text in texts
        ], dtype=bool)
        arrays[f'{name}_lexical'] = np.flatnonzero(lexical)
        case_info.append(dict(
            name=name, row=row, positions=positions, tokens=labels[name],
            lexical_patterns=patterns, lexical_count=int(lexical.sum()),
        ))
        source = targets[name]
        queries = {'meet': source.min(axis=0), 'join': source.max(axis=0)}
        if name == 'rio':
            queries['pair_meet'] = source[:2].min(axis=0)
        singles = [extent(matrix, u, 0.5) for u in source]
        for operation, query in queries.items():
            support = extent(matrix, query, None)
            n_positive = int(np.sum(query > 0))
            arrays[f'{name}_{operation}_query'] = query
            previous = support
            for alpha in [None, *ALPHAS]:
                mask = support if alpha is None else extent(
                    matrix, query, alpha,
                )
                assert mask[row]
                assert np.all(~mask | previous)
                previous = mask
                key = f'{name}_{operation}_{alpha}'
                ids = np.flatnonzero(mask)
                arrays[key] = ids
                cross = coincidence(mask, lexical)
                record = dict(
                    case=name, operation=operation, alpha=alpha,
                    active_coordinates=n_positive, count=len(ids),
                    non_source_count=len(ids) - 1,
                    prevalence=len(ids) / len(tokens),
                    retained_support=len(ids) / int(support.sum()),
                    lexical=cross,
                    lexical_hit_fraction=cross['both'] / len(ids),
                    lexical_coverage=(cross['both'] / int(lexical.sum())
                                      if lexical.any() else None),
                )
                if alpha in [0.5, 1.0]:
                    closed = intent(row_matrix, mask, upper, lower)
                    assert np.array_equal(extent(matrix, closed), mask)
                    assert np.all(closed >= query.astype(float) * alpha)
                    arrays[f'{key}_closed'] = closed
                    record['closed_active_coordinates'] = int(
                        np.sum(closed > 0),
                    )
                    record['closure_added_coordinates'] = int(
                        np.sum((closed > 0) & (query == 0)),
                    )
                    if alpha == 0.5 and operation == 'join':
                        assert np.array_equal(
                            mask, np.logical_and.reduce(singles),
                        )
                    if alpha == 0.5 and operation == 'meet':
                        assert np.all(~np.logical_or.reduce(singles) | mask)
                records.append(record)
            print(kind, layer, name, operation, n_positive,
                  [r['count'] for r in records[-5:]], flush=True)
    report = dict(
        kind=kind, layer=layer, corpus_size=len(tokens), dimension=24576,
        versions={p: metadata.version(p) for p in [
            'torch', 'transformer-lens', 'transformers', 'sae-lens',
            'numpy', 'scipy',
        ]},
        token_sha256=file_hash(token_path),
        matrix_sha256=file_hash(matrix_path),
        code_sha256=file_hash(__file__),
        source_refresh=corrections, cases=case_info, records=records,
        thresholds=ALPHAS, comparison_tolerance=0,
        closure_checks=True, lattice_identity_checks=True,
    )
    stem = f'{kind}_layer{layer}'
    (output / f'{stem}.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(output / f'{stem}.npz', **arrays)
    print('Saved', stem, len(records), 'queries', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['sae', 'tc'])
    parser.add_argument('layer', type=int, choices=[0, 8, 11])
    args = parser.parse_args()
    run(args.kind, args.layer)
