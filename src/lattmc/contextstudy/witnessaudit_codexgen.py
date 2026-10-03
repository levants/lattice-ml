"""Independently audit complete token predicates and FCA caller agreement."""

from __future__ import annotations
from typing import Any

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from lattmc.fca.lattice_utils import meet_all, join_all
from lattmc.fca.fca_utils import find_G_x as original_extent
from lattmc.fca.fca_utils_codexgen import find_G_x
from lattmc.activationstudy.common_codexgen import full_scores, sha256
from lattmc.lattconferencetools.activation_experiments_codexgen import extent
from .witnesses_codexgen import DEST, ROOT, classify


def main() -> dict[str, Any]:
    # A change of alpha can turn a distributed query into a single-token
    # query. Special/padding positions must never leak into valid masks.
    """Verify the cached evidence and save a structured audit report."""
    x = np.array([[2., .5], [.5, 2.]])
    assert classify(x, [1, 1])['status'] == 'D'
    assert classify(x, [1, 1], .5)['status'] == 'S'
    assert classify([[2, 0], [0, .9]], [1, 1])['status'] == 'R'
    assert classify([[0, 0]], [0, 0])['status'] == 'N'
    assert classify(x, [1, 1], positions=[4, 8])['whole'] == []
    with_special = np.vstack([[9, 9], x])
    assert classify(with_special[1:], [1, 1])['status'] == 'D'
    rng = np.random.default_rng(20261002)
    trials = 0
    for _ in range(64):
        a = rng.integers(0, 5, size=(8, 5)).astype(float)
        q = rng.integers(0, 5, size=5).astype(float)
        for alpha in (.25, .5, 1.):
            expected = np.flatnonzero(np.all(a >= alpha*q, axis=1))
            assert np.array_equal(find_G_x(a, alpha*q,
                disable_progress=True).astype(int), expected)
            assert np.array_equal(original_extent(a, alpha*q,
                disable_progress=True).astype(int), expected)
            assert np.array_equal(np.flatnonzero(extent(
                sparse.csc_matrix(a), q, alpha)), expected)
            assert np.array_equal(np.flatnonzero(full_scores(
                sparse.csc_matrix(a), q) >= alpha), expected)
            trials += 1
        assert np.array_equal(meet_all(a), a.min(0))
        assert np.array_equal(join_all(a), a.max(0))
    data = json.loads((DEST / 'records.json').read_text())
    rows, coordinates = 0, 0
    for r in data['records']:
        path = ROOT / r['trace_file']
        assert sha256(path) == r['trace_sha256']
        trace = np.load(path)[r['trace_key']][r['valid_positions']]
        trace = trace.astype(float)
        q = np.array(r['query']) * r['alpha']
        mask = trace >= q
        whole = np.all(mask, axis=1)
        member = bool(np.all(np.any(mask, axis=0)))
        assert r['member'] == member
        assert r['status'] == ('S' if whole.any() else 'D' if member else 'R')
        pos = np.array(r['valid_positions'])
        assert np.array_equal(pos[whole], r['whole'])
        assert np.array_equal(pos[mask.any(axis=1)], r['partial'])
        assert r['matches'] == [np.flatnonzero(x).tolist() for x in mask]
        ratios = trace.max(0) / np.array(r['query'])
        assert np.isclose(r['score'], ratios.min())
        for j,w in enumerate(r['witnesses']):
            p = int(np.argmax(trace[:,j]))
            assert w['position'] == pos[p] and w['value'] == trace[p,j]
            assert w['requirement'] == q[j]
            assert w['meets'] == bool(trace[p,j] >= q[j])
            coordinates += 1
        rows += 1
    assert rows == 72 and coordinates == 306
    legacy = json.loads((DEST / 'legacy.json').read_text())
    assert legacy['source_sha256'] == sha256(
        Path(__file__).with_name('legacyreplay_codexgen.py'))
    assert legacy['trace_sha256'] == sha256(DEST / 'legacy.npz')
    saved = np.load(DEST / 'legacy.npz')
    legacy_rows, unresolved = 0, 0
    for r in legacy['records']:
        if r['status'] == 'U':
            assert len(r['candidates']) > 1
            unresolved += 1
            continue
        trace = saved[r['trace_key']].astype(float)
        q = np.array(r['query'])
        whole = np.all(trace >= q, axis=1)
        member = bool(np.all(trace.max(0) >= q))
        assert r['member'] == member
        assert r['status'] == ('S' if whole.any() else 'D' if member else 'R')
        assert np.array_equal(np.flatnonzero(whole), r['whole'])
        assert np.array_equal(np.flatnonzero((trace >= q).any(1)),
                              r['partial'])
        for j, w in enumerate(r['witnesses']):
            pos = int(np.argmax(trace[:, j]))
            assert w['position'] == pos and w['value'] == trace[pos, j]
            assert w['requirement'] == q[j]
            assert w['meets'] == bool(trace[pos, j] >= q[j])
        legacy_rows += 1
    assert legacy_rows == 22 and unresolved == 4
    result = dict(status='passed', rows=rows, coordinates=coordinates,
                  sparse_dense_library_comparisons=trials,
                  boundary_cases=6, counts=data['counts'],
                  source_sha256=sha256(__file__), legacy_rows=legacy_rows,
                  unresolved_historical_rows=unresolved)
    (DEST / 'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
    return result


if __name__ == '__main__':
    main()
