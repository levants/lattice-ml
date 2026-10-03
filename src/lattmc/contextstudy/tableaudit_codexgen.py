"""Check the document-meet example independently against frozen caches."""

from __future__ import annotations
from typing import Any

import json
import numpy as np
from scipy import sparse

from lattmc.activationstudy.common_codexgen import sha256, save_json
from .commonexample_codexgen import BASE, DEST
from .witnesses_codexgen import classify


def main() -> dict[str, Any]:
    """Verify the cached evidence and save a structured audit report."""
    data = json.loads((DEST / 'records.json').read_text())
    design = json.loads((BASE / 'design.json').read_text())
    assert data['source_sha256'] == sha256(
        __file__.replace('tableaudit', 'commonexample'))
    assert data['design_sha256'] == sha256(BASE / 'design.json')
    sources = [i for i, row in enumerate(design['rows'])
               if row['label'] == 7 and row['split'] == 'train'][:5]
    assert sources == data['sources']
    texts = json.loads((BASE / 'texts.local.json').read_text())
    assert data['texts'] == [texts[i] for i in sources]
    rows_checked = coordinates = 0
    for family, report in data['families'].items():
        path = BASE / f'{family}_max.npz'
        assert report['matrix_sha256'] == sha256(path)
        matrix = sparse.load_npz(path)
        arrays = np.load(DEST / f'{family}.npz')
        assert report['trace_sha256'] == sha256(DEST / f'{family}.npz')
        expected = matrix[sources].toarray().min(0)
        np.testing.assert_array_equal(arrays['query'], expected)
        active = np.flatnonzero(expected)
        selected = matrix[:, active].toarray()
        extent = np.flatnonzero((selected >= expected[active]).all(1))
        assert extent.tolist() == report['extent'] == sources
        np.testing.assert_array_equal(matrix[extent].toarray().min(0),
                                      expected)
        tokens = np.load(BASE / f'{family}_tokens.local.npz')
        for row in report['rows']:
            index = row['row']
            valid = np.flatnonzero(tokens['attention_mask'][index])
            valid = valid[valid != 0]
            assert valid.tolist() == row['valid_positions']
            assert row['tokens'] == tokens['input_ids'][index].tolist()
            assert row['dataset_identity'] == design['rows'][index]
            trace = arrays[str(index)][valid]
            # Exact replay equality, not only an allclose tolerance check.
            np.testing.assert_array_equal(trace.max(0), selected[index])
            result = classify(trace, expected[active], positions=valid)
            for key, value in result.items():
                assert row[key] == value, (family, index, key)
            assert row['status'] == 'D' and not row['whole']
            np.testing.assert_array_equal(row['coordinates'], active)
            coordinates += len(active)
            rows_checked += 1
    result = dict(status='passed', source_rows=5, measured_condition_rows=10,
                  coordinate_witnesses=coordinates,
                  cached_extent_rows_checked=4200,
                  source_summaries_bitwise_equal=True,
                  full_intent_closure=True, cross_model_infomorphism=False,
                  source_sha256=sha256(__file__),
                  records_sha256=sha256(DEST / 'records.json'))
    assert rows_checked == 10
    save_json(DEST / 'audit.json', result)
    print(result)
    return result


if __name__ == '__main__':
    main()
