"""Independently audit closure-added coordinate queries and witness traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from lattmc.contextstudy.operations_codexgen import digest, save


def dense_members(matrix: sparse.csr_matrix, ids: list[int],
                  values: list[float]) -> np.ndarray:
    """Check requirements using bounded dense blocks on surviving rows.

    This row-oriented implementation is independent of the CSC intersection
    routine used to generate the saved extents. No tolerance is applied.
    """
    candidates = np.arange(matrix.shape[0])
    for start in range(0, len(ids), 32):
        if not len(candidates):
            break
        columns = ids[start:start + 32]
        data = matrix[candidates][:, columns].toarray().astype(float)
        keep = (data >= np.array(values[start:start + 32])).all(axis=1)
        candidates = candidates[keep]
    return candidates


def audit(root: Path, out: Path) -> dict[str, int]:
    """Verify all 660 h definitions, exact extents and recorded replays."""
    total = changed = witness_count = 0
    for kind in ('sae', 'tc'):
        for block in (0, 8, 11):
            name = f'gpt2_{kind}{block}'
            data = json.loads((out / f'{name}.json').read_text())
            old_path = root / 'data/activation_studies/contextreading_v1'
            old = json.loads((old_path / f'{name}.json').read_text())
            assert digest(old_path / f'{name}.json') == data['parent_sha256']
            path = root / 'notebooks' / (
                'sae/data/sae' if kind == 'sae' else
                'transcoders/data/transcoders') / f'gpt2/V{block}.npz'
            assert digest(path) == data['matrix_sha256']
            matrix = sparse.load_npz(path).tolil()
            previous = root / 'data/activation_studies/latticemethods_v1'
            traces = np.load(previous / f'{name}_traces.npz')
            for row in (1924, 3457, 4042, 5411):
                matrix[row] = traces[f'trace_{row}'].max(axis=0)
            matrix = matrix.tocsr()
            arrays = np.load(out / f'{name}_members.npz')
            old_arrays = np.load(old_path / f'{name}_members.npz')
            checked = {}
            for r in data['records']:
                u = dict(zip(r['query']['coordinates'], r['query']['values']))
                c = old['closures'][r['closure']]
                expected = {i: v for i, v in zip(c['coordinates'],
                            c['values']) if i not in u and v > 0}
                h = dict(zip(r['h']['coordinates'], r['h']['values']))
                assert h == expected and not (set(h) & set(u))
                signature = tuple(h.items())
                if signature not in checked:
                    checked[signature] = dense_members(
                        matrix, list(h), list(h.values()))
                ids = checked[signature]
                assert np.array_equal(ids, arrays[r['members_key']])
                original = next(x[r['operation']] for x in old['records']
                    if x['group'] == r['group'] and x['rule'] == r['rule'])
                before = old_arrays[original['array_key']]
                assert set(before) <= set(ids)
                extra = np.setdiff1d(ids, before)
                assert np.array_equal(extra, arrays[r['members_key']+'_extra'])
                assert len(extra) == r['additional_count']
                assert len(ids) == r['h_count']
                total += 1
    data = json.loads((out / 'tc11_witnesses.json').read_text())
    path = out / 'tc11_witnesses.npz'
    assert digest(path) == data['traces_sha256']
    arrays = np.load(path)
    for r in data['records']:
        values = arrays[r['trace_key']].astype(float)
        hits = values >= np.array(r['query']['values'])
        whole = np.flatnonzero(hits.all(axis=1)).tolist()
        partial = np.flatnonzero(hits.any(axis=1)).tolist()
        member = bool(hits.any(axis=0).all())
        assert r['whole'] == whole and r['partial'] == partial
        assert r['member'] == member
        assert r['status'] == ('S' if whole else 'D' if member else 'R')
        assert r['replay_changed'] == (member != r['cached_member'])
        changed += int(r['replay_changed'])
        witness_count += 1
    return dict(queries=total, witness_records=witness_count,
                replay_membership_changes=changed)


def main() -> None:
    """Execute the saved-data audit and write its checksum-bearing record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.out)
    save(args.out / 'audit.json', dict(**result,
         source_sha256=digest(Path(__file__))))
    print(result)


if __name__ == '__main__':
    main()
