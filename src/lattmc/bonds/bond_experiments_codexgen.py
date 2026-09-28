"""Exact cached-corpus one-pair bonds and a finite saturation reference.

Run with the repository uv environment. Input caches are read-only.
All numerical comparisons use stored values without an epsilon.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy import sparse

from .paths_codexgen import OUT, ROOT
NOTEBOOK_SEED = [
    25314, 7705, 25413, 1757, 9276, 19655, 23050, 18379, 19094,
    8099, 17518, 13766, 20164, 2949, 15098, 11772, 23708,
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def saturate(seed, xs, vs, le_x, le_v, join_x, join_v, cx, dv):
    """Whole-fiber algorithm; closures and joins return lattice elements."""
    relation = set(seed)
    history = [len(relation)]
    while True:
        added = set(relation)
        for a, v in relation:
            added.update(((cx(a), v), (a, dv(v))))
        for a in xs:
            added.add((a, join_v([v for b, v in relation if b == a])))
        for v in vs:
            added.add((join_x([a for a, w in relation if w == v]), v))
        updated = {
            (b, w) for a, v in added for b in xs for w in vs
            if le_x(b, a) and le_v(w, v)
        }
        if updated == relation:
            return relation, history
        relation = updated
        history.append(len(relation))


def one_pair(seed, value, xs, vs, le_x, le_v, cx, dv, bx, bv):
    """Explicit small-lattice evaluation of the three rectangles."""
    a0, a1, v0, v1 = cx(bx), cx(seed), dv(bv), dv(value)
    return {
        (a, v) for a in xs for v in vs
        if le_x(a, a0) or le_v(v, v0)
        or (le_x(a, a1) and le_v(v, v1))
    }


def extent(csc, query):
    candidates = np.ones(csc.shape[0], dtype=bool)
    active = np.flatnonzero(query > 0)
    sizes = np.diff(csc.indptr)[active]
    for col in active[np.argsort(sizes, kind='stable')]:
        start, stop = csc.indptr[col:col + 2]
        keep = csc.data[start:stop] >= query[col]
        column = np.zeros(csc.shape[0], dtype=bool)
        column[csc.indices[start:stop][keep]] = True
        candidates &= column
        if not candidates.any():
            break
    return candidates


def intent(csr, rows, upper):
    rows = np.asarray(rows, dtype=int)
    result = upper.copy()
    for start in range(0, len(rows), 128):
        block = csr[rows[start:start + 128]].toarray()
        result = np.minimum(result, block.min(axis=0))
    return result


def context(kind, layer, seeds):
    start_time = time.perf_counter()
    folder = 'sae' if kind == 'sae' else 'transcoders'
    path = ROOT / f'notebooks/{folder}/data/{folder}/gpt2/V{layer}.npz'
    csr = sparse.load_npz(path).tocsr()
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    assert csr.shape == (25600, 24576)
    assert np.isfinite(csr.data).all() and (csr.data > 0).all()
    csc = csr.tocsc()
    upper = csr.max(axis=0).toarray().ravel()
    lower = csr.min(axis=0).toarray().ravel()
    floor = np.zeros(csr.shape[1], dtype=csr.dtype)
    for col in range(csc.shape[1]):
        values = csc.data[csc.indptr[col]:csc.indptr[col + 1]]
        if len(values):
            floor[col] = values.min()
    name = f'{kind}{layer}'
    arrays = {'upper': upper, 'lower': lower, 'floor': floor}
    records = []
    for mode in ('graded', 'support'):
        top = upper if mode == 'graded' else floor
        bottom_intent = lower if mode == 'graded' else np.minimum(
            lower, floor
        )
        empty_closed = extent(csc, top)
        arrays[f'{mode}_empty_extent'] = np.flatnonzero(empty_closed)
        arrays[f'{mode}_bottom_intent'] = bottom_intent
        assert np.array_equal(extent(csc, bottom_intent),
                              np.ones(25600, dtype=bool))
        for index, seed in enumerate(seeds):
            query = intent(csr, seed, upper)
            if mode == 'support':
                query = np.minimum(query, floor)
            closed = extent(csc, query)
            ids = np.flatnonzero(closed)
            roundtrip = intent(csr, ids, upper)
            if mode == 'support':
                roundtrip = np.minimum(roundtrip, floor)
            assert closed[seed].all()
            assert np.array_equal(roundtrip, query)
            assert not np.any(empty_closed & ~closed)
            arrays[f'{mode}_{index}_intent'] = query
            arrays[f'{mode}_{index}_extent'] = ids
            records.append(dict(
                context=name, mode=mode, seed=index, size=len(ids),
                active=int(np.count_nonzero(query)),
                universal=bool(closed.all()),
                empty_extent_size=int(empty_closed.sum()),
            ))
    for index in range(len(seeds)):
        assert set(arrays[f'graded_{index}_extent']).issubset(
            arrays[f'support_{index}_extent']
        )
    np.savez_compressed(OUT / f'{name}_closures.npz', **arrays)
    manifest = dict(
        context=name, cache=str(path.relative_to(ROOT)),
        sha256=sha256(path), shape=list(csr.shape), dtype=str(csr.dtype),
        nnz=int(csr.nnz), seconds=time.perf_counter() - start_time,
        output_sha256=sha256(OUT / f'{name}_closures.npz'),
    )
    del csr, csc
    gc.collect()
    print(name, 'done', round(manifest['seconds'], 2), flush=True)
    return manifest, records


def comparisons(seeds, primary):
    arrays = {
        name: np.load(OUT / f'{name}_closures.npz') for name in primary
    }
    records = []
    for source, target in itertools.permutations(primary, 2):
        if source.lstrip('saetc') == target.lstrip('saetc'):
            continue
        for mode in ('graded', 'support'):
            for index, seed in enumerate(seeds):
                left = arrays[source][f'{mode}_{index}_extent']
                right = arrays[target][f'{mode}_{index}_extent']
                common = np.intersect1d(left, right)
                only_left = np.setdiff1d(left, right)
                only_right = np.setdiff1d(right, left)
                query = arrays[target][f'{mode}_{index}_intent']
                bottom = arrays[target][f'{mode}_bottom_intent']
                records.append(dict(
                    source=source, target=target, mode=mode, seed=index,
                    source_size=len(left), target_size=len(right),
                    intersection=len(common), nonseed_common=len(common)-17,
                    source_only=len(only_left), target_only=len(only_right),
                    jaccard=len(common) / (len(left)+len(right)-len(common)),
                    direct_violation=bool(len(only_left)),
                    trivial_bond=bool(np.array_equal(query, bottom)),
                    added_singletons=int(len(left)-17),
                    representation=dict(
                        source=f'{source}_closures.npz',
                        target=f'{target}_closures.npz',
                        source_extent=f'{mode}_{index}_extent',
                        source_bottom=f'{mode}_empty_extent',
                        target_intent=f'{mode}_{index}_intent',
                        target_bottom=f'{mode}_bottom_intent',
                    ),
                ))
    return records


def run():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(20260928)
    seeds = [NOTEBOOK_SEED] + [
        sorted(rng.choice(25600, size=17, replace=False).tolist())
        for _ in range(32)
    ]
    protocol = dict(
        random_seed=20260928, seeds=seeds,
        primary_layers=[0, 8, 11], modes=['graded', 'support'],
        selection='Notebook seed plus 32 uniform 17-row samples',
        comparison='All ordered distinct-layer pairs in the six contexts',
        input_policy='Stored unmodified caches; no fresh model inference',
    )
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    configurations = list(itertools.product(('sae', 'tc'), (0, 8, 11)))
    configurations += [('tc', 1), ('tc', 3), ('tc', 9)]
    manifests, closures = [], []
    for kind, layer in configurations:
        manifest, records = context(kind, layer, seeds)
        manifests.append(manifest)
        closures.extend(records)
    primary = [f'{kind}{layer}' for kind, layer in configurations[:6]]
    pairs = comparisons(seeds, primary)
    old = []
    for source, target in [('tc3', 'tc9'), ('tc1', 'sae11')]:
        for row in comparisons(seeds[:1], [source, target]):
            if row['source'] == source and row['mode'] == 'graded':
                old.append(row)
    tokens = ROOT / 'notebooks/transcoders/data/transcoders/gpt2'
    tokens /= 'owt_tokens/owt_tokens_torch.pt'
    notebooks = []
    for folder in ('sae', 'transcoders', 'bonds'):
        for path in sorted((ROOT / 'notebooks' / folder).glob('*.ipynb')):
            if 'tokens_places' in path.name or path.name in (
                'bonds_sae_words_to_embeddings.ipynb',
                'bonds_tc_words_to_embeddings.ipynb',
            ):
                if 'gemma' not in path.name:
                    notebooks.append(dict(
                        path=str(path.relative_to(ROOT)), sha256=sha256(path)
                    ))
    report = dict(
        protocol=protocol, contexts=manifests, closures=closures,
        comparisons=pairs, historical=old, notebook_inputs=notebooks,
        tokens=dict(path=str(tokens.relative_to(ROOT)), sha256=sha256(tokens)),
        environment=dict(
            python=platform.python_version(),
            numpy=importlib.metadata.version('numpy'),
            scipy=importlib.metadata.version('scipy'),
            revision=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
            ).strip(),
            script_sha256=sha256(Path(__file__)),
            script_path=str(Path(__file__).resolve().relative_to(ROOT)),
        ),
    )
    (OUT / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Completed', len(pairs), 'exact minimal bonds', flush=True)
    print('Historical:', json.dumps(old), flush=True)


if __name__ == '__main__':
    run()
