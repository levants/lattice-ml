"""Independently verify exploratory extents, closures and token witnesses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from lattmc.contextstudy.operations_codexgen import digest, save


def audit(root: Path, out: Path) -> dict[str, int]:
    """Check dense inequalities and explicit componentwise intent minima.

    Query records may repeat descriptions. Distinct dense masks and closed
    extents are counted separately; the audit does not label semantics.
    """
    query_records = dense_masks = closure_count = witness_count = 0
    for kind in ('sae', 'tc'):
        for block in (0, 8, 11):
            name = f'gpt2_{kind}{block}'
            data = json.loads((out / f'{name}.json').read_text())
            path = root / 'notebooks' / (
                'sae/data/sae' if kind == 'sae' else
                'transcoders/data/transcoders') / f'gpt2/V{block}.npz'
            assert digest(path) == data['previous_record']['matrix_sha256']
            matrix = sparse.load_npz(path).tolil()
            previous = root / 'data/activation_studies/latticemethods_v1'
            traces = np.load(previous / f'{name}_traces.npz')
            for row in (1924, 3457, 4042, 5411):
                matrix[row] = traces[f'trace_{row}'].max(axis=0)
            matrix = matrix.tocsr()
            members = np.load(out / f'{name}_members.npz')
            checked, closures = {}, set()
            records = [data['baseline']['zero']]
            for r in data['records']:
                if 'skipped' not in r:
                    records += r['components'] + [r['meet'], r['join']]
            for record in records:
                query = record['query']
                signature = json.dumps(query, sort_keys=True)
                if signature not in checked:
                    columns = query['coordinates']
                    mask = (matrix[:, columns].toarray().astype(float) >=
                            np.array(query['values'])).all(axis=1)
                    checked[signature] = np.flatnonzero(mask)
                    dense_masks += 1
                ids = checked[signature]
                assert np.array_equal(ids, members[record['array_key']])
                assert len(ids) == record['count']
                query_records += 1
                key = record['closure_key']
                closure = data['closures'][key]
                c = np.zeros(matrix.shape[1])
                c[closure['coordinates']] = closure['values']
                q = np.zeros_like(c)
                q[query['coordinates']] = query['values']
                assert np.all(c >= q)
                if key in closures:
                    continue
                if len(ids):
                    expected = matrix[ids[0]].toarray()[0].astype(float)
                    for start in range(1, len(ids), 128):
                        active = np.flatnonzero(expected)
                        if not len(active):
                            break
                        values = matrix[ids[start:start+128]][:, active]
                        expected[active] = np.minimum(expected[active],
                            values.toarray().astype(float).min(axis=0))
                else:
                    expected = np.zeros(matrix.shape[1])
                    np.maximum.at(expected, matrix.indices, matrix.data)
                np.testing.assert_array_equal(expected, c)
                closures.add(key)
                closure_count += 1
    for path in out.glob('*_witnesses.json'):
        data = json.loads(path.read_text())
        arrays_path = path.with_suffix('.npz')
        assert digest(arrays_path) == data['traces_sha256']
        arrays = np.load(arrays_path)
        for r in data['records']:
            values = arrays[r['trace_key']].astype(float)
            q = np.array(r['query']['values'])
            hits = values >= q
            whole = np.flatnonzero(hits.all(axis=1)).tolist()
            partial = np.flatnonzero(hits.any(axis=1)).tolist()
            assert whole == r['whole'] and partial == r['partial']
            member = bool(hits.any(axis=0).all())
            assert member == r['member']
            assert r['status'] == ('S' if whole else 'D' if member else 'R')
            witness_count += 1
    data = json.loads((out / 'structure_probe.json').read_text())
    path = out / 'structure_traces.npz'
    assert digest(path) == data['traces_sha256']
    arrays = np.load(path)
    for r in data['records']:
        valid = np.array(r['valid'])
        whole = valid[arrays[r['trace_key']][valid].astype(float) >=
                      r['query']['values'][0]].tolist()
        assert whole == r['whole']
        assert bool(whole) == r['cached_member'] == r['replay_member']
        witness_count += 1
    data = json.loads((out / 'prefix_contrast.json').read_text())
    path = out / 'prefix_contrast_traces.npz'
    assert digest(path) == data['traces_sha256']
    arrays = np.load(path)
    for r in data['records']:
        a = arrays[f'{r["row"]}_original']
        b = arrays[f'{r["row"]}_suffix']
        np.testing.assert_array_equal(a, b)
        for key, field in [('original', 'original'), ('edited', 'edited'),
                           ('family', 'family_activation')]:
            assert float(arrays[f'{r["row"]}_{key}'][4355]) == r[field]
    return dict(query_records=query_records, distinct_dense_masks=dense_masks,
                distinct_closed_extents=closure_count,
                token_witness_records=witness_count,
                prefix_contrasts=len(data['records']))


def main() -> None:
    """Run numerical checks and save their counts with a source checksum."""
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
