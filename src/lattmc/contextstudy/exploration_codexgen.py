"""Explore exact contextual lattice operations without semantic labels.

Reuse saved token codes; record constituent masks, closures and fixed
reading samples. No new inference, threshold fitting or label lookup occurs.
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

from lattmc.contextstudy.operations_codexgen import (
    Record, Vector, digest, encoded, extent, project, save,
)

SOURCES = dict(
    New=(3457, 1), City=(3457, 3), findings=(3457, 21),
    experiments=(3457, 49), monkeys=(3457, 55), mice=(3457, 57),
    virus=(3457, 65), Rio=(5411, 15), de=(5411, 16),
    Janeiro=(5411, 17), specimens=(5411, 31), lab=(5411, 45),
    Paulo=(5411, 49), species=(5411, 74), measurements=(5411, 94),
    analysis=(5411, 115), Cat=(4042, 8), version=(4042, 27),
    city=(4042, 79), dog=(4042, 82), park=(4042, 83),
    Cleveland=(1924, 15), Clippers=(1924, 39), rating=(1924, 72),
    Cavaliers=(1924, 94))
GROUPS = dict(
    biological_pair=['monkeys', 'specimens'],
    research_three=['experiments', 'measurements', 'analysis'],
    urban_four=['City', 'Cleveland', 'Paulo', 'city'],
    mixed_pair=['virus', 'Clippers'],
    mixed_three=['dog', 'lab', 'rating'],
    mixed_six=['New', 'de', 'Cat', 'Cavaliers', 'monkeys', 'specimens'],
    biological_four=['monkeys', 'mice', 'specimens', 'species'],
    assessment_four=['findings', 'analysis', 'rating', 'version'],
    new_de=['New', 'de'], rio_janeiro=['Rio', 'Janeiro'],
    cat_dog=['Cat', 'dog'])
RULES = ('full', 'rank1', 'rank2', 'rank3', 'rank23')


def selected(vector: Vector, rule: str, index: int) -> Vector | None:
    """Apply a fixed projection rule, retaining the original amplitude."""
    if rule == 'full':
        return vector
    rank = 2 + index % 2 if rule == 'rank23' else int(rule[-1])
    return project(vector, (rank,))


def sample(ids: Vector, excluded: set[int]) -> list[int]:
    """Select two earliest and up to six seeded non-source members."""
    valid = np.array([i for i in ids if i not in excluded], dtype=int)
    if len(valid) <= 8:
        return valid.tolist()
    rng = np.random.default_rng(20261004)
    return valid[:2].tolist() + sorted(rng.choice(
        valid[2:], 6, replace=False).tolist())


def evaluate(root: Path, out: Path, kind: str, block: int,
             protocol: Path) -> None:
    """Evaluate a condition against exact original summaries and closures.

    Saves JSON provenance/queries and compressed member arrays. All source
    traces and cache hashes must match the previous executed study.
    """
    previous = root / 'data/activation_studies/latticemethods_v1'
    name = f'gpt2_{kind}{block}'
    old = json.loads((previous / f'{name}.json').read_text())
    trace_path = previous / f'{name}_traces.npz'
    assert digest(trace_path) == old['traces_sha256']
    traces = np.load(trace_path)
    matrix_path = root / 'notebooks' / (
        'sae/data/sae' if kind == 'sae' else 'transcoders/data/transcoders')
    matrix_path /= f'gpt2/V{block}.npz'
    assert digest(matrix_path) == old['matrix_sha256']
    matrix = sparse.load_npz(matrix_path).tolil()
    for row in sorted({r for r, _ in SOURCES.values()}):
        matrix[row] = traces[f'trace_{row}'].max(axis=0)
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    csc = matrix.tocsc()
    top = matrix.max(axis=0).toarray().ravel()
    token_path = root / 'notebooks/transcoders/data/transcoders/gpt2'
    token_path /= 'owt_tokens/owt_tokens_torch.pt'
    assert digest(token_path) == old['token_sha256']
    tokens = torch.load(token_path, map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    vectors = {n: traces[f'trace_{r}'][p] for n, (r, p) in SOURCES.items()}
    excluded = {r for r, _ in SOURCES.values()}
    masks, closures, arrays, records = {}, {}, {}, []

    def measure(query: Vector, key: str) -> Record:
        """Save exact extent and verify its invariant under intent closure."""
        signature = query.tobytes()
        if signature not in masks:
            masks[signature] = extent(csc, query)
        mask = masks[signature]
        ids = np.flatnonzero(mask)
        arrays[key] = ids
        mask_key = hashlib.sha256(mask.tobytes()).hexdigest()
        if mask_key not in closures:
            closed = (matrix[ids].min(axis=0).toarray().ravel()
                      if len(ids) else top.copy())
            assert np.array_equal(extent(csc, closed), mask)
            closures[mask_key] = closed
        closed = closures[mask_key]
        assert np.all(closed.astype(float) >= query.astype(float))
        return dict(query=encoded(query), count=len(ids),
                    positive=int(np.count_nonzero(query)),
                    closure_key=mask_key,
                    closed_positive=int(np.count_nonzero(closed)),
                    added_closure_coordinates=int(np.count_nonzero(
                        (closed > 0) & (query == 0))),
                    sampled_ids=sample(ids, excluded), array_key=key)

    for group, names in GROUPS.items():
        for rule in RULES:
            values = [selected(vectors[n], rule, i)
                      for i, n in enumerate(names)]
            if any(v is None for v in values):
                records.append(dict(group=group, rule=rule,
                                    skipped='insufficient_positive_ranks'))
                continue
            key = f'{group}_{rule}'
            components = [measure(v, f'{key}_c{i}')
                          for i, v in enumerate(values)]
            meet = measure(np.minimum.reduce(values), f'{key}_meet')
            join = measure(np.maximum.reduce(values), f'{key}_join')
            union = set().union(*(set(arrays[c['array_key']])
                                  for c in components))
            intersection = set(arrays[components[0]['array_key']])
            for c in components[1:]:
                intersection.intersection_update(arrays[c['array_key']])
            assert intersection == set(arrays[join['array_key']])
            assert union <= set(arrays[meet['array_key']])
            extra = np.array(sorted(set(arrays[meet['array_key']])-union))
            arrays[f'{key}_extra'] = extra.astype(int)
            records.append(dict(
                group=group, rule=rule, sources=names, components=components,
                meet=meet, join=join, extra_count=len(extra),
                extra_sample=sample(extra, excluded),
                retention=[join['count']/c['count'] if c['count'] else None
                           for c in components],
                union_count=len(union)))
    zero = measure(np.zeros(matrix.shape[1], dtype=np.float32), 'zero')
    bos = traces['trace_3457'][0]
    baseline = dict(
        zero=zero, bos=encoded(bos),
        all_initial_tokens_equal=bool(torch.all(tokens[:, 0] == tokens[0, 0])),
        source_bos_exactly_equal=all(np.array_equal(
            traces[f'trace_{row}'][0], bos) for row in excluded),
        bos_extent=int(extent(csc, bos).sum()))
    save(out / f'{name}.json', dict(
        records=records, baseline=baseline,
        closures={k: encoded(v) for k, v in closures.items()},
        previous_record=old,
        source_positions=SOURCES, source_sha256=digest(Path(__file__)),
        protocol_sha256=digest(protocol), seed=20261004,
        interpretation='exploratory; no labels used'))
    np.savez_compressed(out / f'{name}_members.npz', **arrays)
    ids = sorted({row for record in records for op in ('meet', 'join')
                  for row in record.get(op, {}).get('sampled_ids', [])})
    save(out / f'{name}_reading.json', {
        str(i): tokenizer.decode(tokens[i],
            clean_up_tokenization_spaces=False) for i in ids})
    print(name, len(records), 'groups/rules; closures', len(closures),
          'universal closure coordinates', zero['closed_positive'],
          flush=True)


def main() -> None:
    """Run one offline condition from verified cached token traces."""
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
