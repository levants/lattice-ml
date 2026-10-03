"""Isolate and evaluate coordinates introduced by description closure.

Use every saved meet/join in contextreading_v1, preserving its dictionary,
source-row refresh and exact amplitude convention. Save all extents and
separate empty-source conventions from empirical co-occurrence evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from transformers import AutoTokenizer

from lattmc.contextstudy.exploration_codexgen import SOURCES
from lattmc.contextstudy.operations_codexgen import (
    Record, Vector, digest, encoded, extent, save,
)


def decode(record: Record, width: int) -> Vector:
    """Expand a serialized nonnegative description without rounding."""
    result = np.zeros(width, dtype=np.float64)
    result[record['coordinates']] = record['values']
    return result


def sample(ids: Vector, excluded: set[int]) -> list[int]:
    """Choose two initial and up to six seeded additional non-source rows."""
    valid = np.array([i for i in ids if i not in excluded], dtype=int)
    if len(valid) <= 8:
        return valid.tolist()
    rng = np.random.default_rng(20261005)
    return valid[:2].tolist() + sorted(rng.choice(
        valid[2:], 6, replace=False).tolist())


def evaluate(root: Path, out: Path, kind: str, block: int,
             protocol: Path) -> None:
    """Compute h extents and closure diagnostics for one fixed dictionary.

    Reads immutable cached codes; refreshes only the four established
    source rows in memory. Writes JSON records, membership arrays and a
    label-free packet of sampled original text. Comparisons use no epsilon.
    """
    previous = root / 'data/activation_studies/contextreading_v1'
    source = root / 'data/activation_studies/latticemethods_v1'
    name = f'gpt2_{kind}{block}'
    path = previous / f'{name}.json'
    data = json.loads(path.read_text())
    old = data['previous_record']
    trace_path = source / f'{name}_traces.npz'
    assert digest(trace_path) == old['traces_sha256']
    traces = np.load(trace_path)
    matrix_path = root / 'notebooks' / (
        'sae/data/sae' if kind == 'sae' else 'transcoders/data/transcoders')
    matrix_path /= f'gpt2/V{block}.npz'
    assert digest(matrix_path) == old['matrix_sha256']
    matrix = sparse.load_npz(matrix_path).tolil()
    excluded = {r for r, _ in SOURCES.values()}
    for row in sorted(excluded):
        matrix[row] = traces[f'trace_{row}'].max(axis=0)
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    csc = matrix.tocsc()
    width = matrix.shape[1]
    top = matrix.max(axis=0).toarray().ravel().astype(float)
    floor = decode(data['closures'][data['baseline']['zero']['closure_key']],
                   width)
    old_members = np.load(previous / f'{name}_members.npz')
    arrays, records, closures = {}, [], {}
    for case in data['records']:
        for operation in ('meet', 'join'):
            original = case[operation]
            u = decode(original['query'], width)
            c = decode(data['closures'][original['closure_key']], width)
            h = np.where(u == 0, c, 0)
            mask = extent(csc, h)
            ids = np.flatnonzero(mask)
            before = old_members[original['array_key']]
            original_mask = np.zeros(matrix.shape[0], dtype=bool)
            original_mask[before] = True
            assert (mask[before]).all()
            assert np.array_equal(extent(csc, np.maximum(u, h)),
                                  original_mask)
            effective = np.where(h > floor, h, 0)
            assert np.array_equal(extent(csc, effective), mask)
            signature = hashlib.sha256(mask.tobytes()).hexdigest()
            if signature not in closures:
                closed = (matrix[ids].min(axis=0).toarray().ravel()
                          if len(ids) else top.copy())
                assert np.array_equal(extent(csc, closed), mask)
                closures[signature] = encoded(closed)
            closed = decode(closures[signature], width)
            equal = np.array_equal(mask, original_mask)
            assert np.all(closed >= h)
            assert bool(np.all(u <= closed)) == equal
            key = f'{case["group"]}_{case["rule"]}_{operation}'
            additional = np.flatnonzero(mask & ~original_mask)
            arrays[key] = ids
            arrays[key + '_extra'] = additional
            records.append(dict(
                group=case['group'], rule=case['rule'], operation=operation,
                sources=case['sources'], query=original['query'],
                closure=original['closure_key'], h=encoded(h),
                effective_h=encoded(effective), h_closure=signature,
                original_count=len(before), h_count=len(ids),
                additional_count=len(additional), same_extent=equal,
                original_positive=int(np.count_nonzero(u)),
                closure_positive=int(np.count_nonzero(c)),
                added_positive=int(np.count_nonzero(h)),
                floor_only_positive=int(np.count_nonzero((h > 0) &
                                                        (h <= floor))),
                strengthened_existing=int(np.count_nonzero((u > 0) &
                                                           (c > u))),
                empty_original=not len(before), zero_h=not np.any(h),
                retention=len(before)/len(ids) if len(ids) else None,
                additional_sample=sample(additional, excluded),
                member_sample=sample(ids, excluded), members_key=key))
    token_path = root / 'notebooks/transcoders/data/transcoders/gpt2'
    token_path /= 'owt_tokens/owt_tokens_torch.pt'
    assert digest(token_path) == old['token_sha256']
    tokens = torch.load(token_path, map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    ids = sorted({i for r in records for i in r['additional_sample']})
    save(out / f'{name}_reading.json', {str(i): tokenizer.decode(
        tokens[i], clean_up_tokenization_spaces=False) for i in ids})
    save(out / f'{name}.json', dict(
        records=records, h_closures=closures,
        parent_sha256=digest(path), matrix_sha256=digest(matrix_path),
        source_sha256=digest(Path(__file__)), seed=20261005,
        protocol_sha256=digest(protocol), membership_tolerance=0))
    np.savez_compressed(out / f'{name}_members.npz', **arrays)
    informative = [r for r in records if not r['empty_original']]
    print(name, 'nonempty', len(informative), 'same',
          sum(r['same_extent'] for r in informative), 'expanded',
          sum(r['additional_count'] > 0 for r in informative), flush=True)


def main() -> None:
    """Run one condition using the existing project environment and caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--kind', choices=('sae', 'tc'), required=True)
    parser.add_argument('--block', type=int, choices=(0, 8, 11), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluate(args.root, args.out, args.kind, args.block, args.protocol)


if __name__ == '__main__':
    main()
