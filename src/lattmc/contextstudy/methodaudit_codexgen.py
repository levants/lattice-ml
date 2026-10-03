"""Independently audit selected-query metrics and saved token witnesses.

Dense comparisons on selected coordinates deliberately avoid the sparse
posting implementation used for experiment generation. Audits do not run
neural inference or treat category labels as feature annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from lattmc.contextstudy.methodstudy_codexgen import CONDITIONS, load_condition
from lattmc.contextstudy.operations_codexgen import digest, save


def audit(root: Path, out: Path) -> dict[str, int]:
    """Verify all selected-grid confusion counts and every gallery highlight.

    Args:
        root: Repository with the original immutable activation matrices.
        out: Directory containing this study's records and token traces.

    Returns:
        Numbers of independently checked query masks and token galleries.

    Raises:
        AssertionError: If identities, metrics, hashes or highlights differ.
    """
    checks = gallery_checks = 0
    for dataset in ('ag_news', 'dbpedia_14'):
        design = json.loads((out / f'{dataset}_design.json').read_text())
        test = np.array(design['test'])
        labels = np.array(design['labels'])
        assert not set(design['train']).intersection(test)
        for condition in CONDITIONS:
            matrix, _ = load_condition(root, dataset, condition)
            path = out / f'{dataset}_{condition}.json'
            data = json.loads(path.read_text())
            for record in data['records']:
                truth = labels[test] == record['label']
                for row in record['selected']:
                    if 'skipped' in row:
                        continue
                    masks = []
                    for side in ('left', 'right'):
                        q = row[side]
                        values = matrix[test][:, q['coordinates']].toarray()
                        masks.append((values.astype(float) >=
                                      np.array(q['values'])).all(axis=1))
                    masks.append(masks[0] & masks[1])
                    for name, mask in zip(('left', 'right', 'join'), masks):
                        expected = row['results'][name]['test']
                        tp = int((mask & truth).sum())
                        fp = int((mask & ~truth).sum())
                        assert expected['tp'] == tp and expected['fp'] == fp
                        assert expected['n'] == int(mask.sum())
                        assert expected['fn'] == int((~mask & truth).sum())
                        assert expected['tn'] == int((~mask & ~truth).sum())
                        if mask.any():
                            observed = tp / int(mask.sum())
                            assert expected['precision'] == observed
                        else:
                            assert expected['precision'] is None
                        checks += 1
                    assert test[masks[2]].tolist() == row['test_members']
    for path in sorted(out.glob('gpt2_*.json')):
        data = json.loads(path.read_text())
        trace_path = path.with_name(path.stem + '_traces.npz')
        assert digest(trace_path) == data['traces_sha256']
        arrays = np.load(trace_path)
        for row in data['gallery']:
            if row.get('empty'):
                continue
            code = arrays[f'gallery_{row["query_name"]}_{row["row"]}']
            q = row['query']
            positive = q['coordinates']
            values = np.array(q['values'])
            if not positive:
                assert row['status'] == 'N'
            else:
                matches = code[:, positive].astype(float) >= values
                whole = np.flatnonzero(matches.all(axis=1)).tolist()
                partial = np.flatnonzero(matches.any(axis=1)).tolist()
                assert whole == row['whole'] and partial == row['partial']
                member = bool(matches.any(axis=0).all())
                assert member == row['member']
                status = 'S' if whole else 'D' if member else 'R'
                assert status == row['status']
            gallery_checks += 1
    data = json.loads((out / 'document_gallery.json').read_text())
    path = out / 'document_gallery_traces.npz'
    assert digest(path) == data['traces_sha256']
    arrays = np.load(path)
    for row in data['records']:
        if row.get('empty'):
            continue
        valid = np.array(row['valid'])
        values = arrays[row['trace_key']][valid].astype(float)
        matches = values >= np.array(row['query']['values'])
        if not row['query']['coordinates']:
            assert row['status'] == 'N'
        else:
            assert valid[matches.all(axis=1)].tolist() == row['whole']
            assert valid[matches.any(axis=1)].tolist() == row['partial']
            assert bool(matches.any(axis=0).all()) == row['member']
        gallery_checks += 1
    return dict(dense_query_masks=checks, token_gallery_rows=gallery_checks)


def main() -> None:
    """Run independent checks and save their counts and implementation hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    counts = audit(args.root, args.out)
    save(args.out / 'audit.json', dict(**counts,
         audit_source_sha256=digest(Path(__file__))))
    print(counts)


if __name__ == '__main__':
    main()
