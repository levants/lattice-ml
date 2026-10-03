"""Audit stored held-out scores independently of the table generator."""

from __future__ import annotations

import json

from .paths_codexgen import PROTOCOL, NOTEBOOKS

import nbformat
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

from .conference_experiments_codexgen import HERE, OUT, METHODS, sha256


def main() -> None:
    """Audit stored held-out scores independently of the table generator."""
    design = json.loads((OUT / 'design.json').read_text())
    train, cal, test = (np.array(design['splits'][key])
                        for key in ('train', 'calibration', 'test'))
    assert not set(train) & set(cal)
    assert not set(train) & set(test)
    assert not set(cal) & set(test)
    assert len(set(train) | set(cal) | set(test)) == design['rows']
    assert design['protocol_sha256'] == sha256(PROTOCOL)
    for task in design['tasks']:
        assert set(task['sources']) <= set(train) & set(task['positives'])
        assert set(task['random_sources']) <= set(train)
    checked = 0
    for kind in ('sae', 'tc'):
        for layer in (0, 8, 11):
            stem = f'{kind}_{layer}'
            report = json.loads((OUT / f'{stem}.json').read_text())
            assert report['design_sha256'] == sha256(OUT / 'design.json')
            scores = np.load(OUT / f'{stem}_scores.npz')
            for i, (task, result) in enumerate(zip(design['tasks'],
                                                   report['runs'])):
                y = np.isin(test, task['positives'])
                for method in METHODS:
                    score = scores[f'{i}_{method}']
                    metric = result['metrics'][method]
                    assert np.all(np.isfinite(score))
                    ap = average_precision_score(y, score)
                    np.testing.assert_allclose(ap, metric['ap'], atol=1e-14)
                    if method == 'graded':
                        pred = score >= result['alpha']
                        tn, fp, fn, tp = confusion_matrix(
                            y, pred, labels=[False, True]).ravel()
                        assert [tn, fp, fn, tp] == [
                            metric[key] for key in ('tn', 'fp', 'fn', 'tp')]
                    checked += 1
    notebook = nbformat.read(NOTEBOOKS /
                 'conference_retrieval_codexgen.ipynb', 4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is not None for cell in code)
    assert not any(output.output_type == 'error' for cell in code
                   for output in cell.outputs)
    print(f'PASS: {checked} stored AP values, 960 graded coincidence tables,')
    print('split/source isolation, protocol hashes,',
          len(code), 'notebook cells.')


if __name__ == '__main__':
    main()
