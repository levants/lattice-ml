"""Independently verify the ranked gallery and its token witnesses."""

from __future__ import annotations
from typing import Any

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from lattmc.activationstudy.common_codexgen import save_json, sha256
from .depth_codexgen import OLD, OUT


def main() -> dict[str, Any]:
    """Verify the cached evidence and save a structured audit report."""
    count = 0
    for path in sorted(OUT.glob('*_gallery.json')):
        family = path.name.removesuffix('_gallery.json')
        data = json.loads(path.read_text())
        source = OLD / 'dbpedia_14'
        assert data['source_sha256'] == sha256(
            Path(__file__).with_name('gallery_codexgen.py'))
        paths = dict(design='design.json', tokens=f'{family}_tokens.local.npz',
                     extraction=f'{family}_extraction.json',
                     results=f'{family}_max_results.json')
        for key, name in paths.items():
            assert data[key + '_sha256'] == sha256(source / name)
        trace_path = OUT / f'{family}_gallery.npz'
        assert data['traces_sha256'] == sha256(trace_path)
        traces = np.load(trace_path)
        result = json.loads((source / paths['results']).read_text())
        scores = np.load(source / f'{family}_max_scores.npz')['graded']
        matrix = sparse.load_npz(source / f'{family}_max.npz').tocsr()
        assert len(data['records']) == 6
        for r in data['records']:
            i = next(i for i, t in enumerate(result['records'])
                     if t['label'] == r['task_label'] and t['repeat'] == 0
                     and t['shot'] == 3)
            task = result['records'][i]
            ranked = np.lexsort((result['test_ids'], -scores[i]))[:3]
            assert result['test_ids'][ranked[r['rank'] - 1]] == r['row']
            assert r['positive'] == task['positive']
            assert not set(r['positive']) & set(result['test_ids'])
            active = np.array(r['selected_coordinates'])
            assert np.array_equal(active, sorted(task['selected_coordinates']))
            query = matrix[r['positive']].toarray().min(0)[active]
            assert np.array_equal(query, r['query']) and (query > 0).all()
            trace = traces[f"{r['task_label']}_{r['rank']}"]
            valid = np.array(r['valid_positions'])
            pooled = trace[valid].max(0)
            weakest = int(np.argmin(pooled / query))
            assert active[weakest] == r['feature']
            p = int(valid[np.argmax(trace[valid, weakest])])
            assert p == r['position']
            assert trace[p, weakest] == r['activation']
            score = float(np.min(pooled / query))
            assert np.isclose(score, r['score'], rtol=1e-6)
            assert np.isclose(score, r['cached_score'], rtol=1e-3, atol=1e-3)
            assert r['accepted'] == (score >= r['calibrated_threshold'])
            count += 1
    assert count == 24
    result = dict(status='passed', examples=count,
                  source_sha256=sha256(__file__))
    save_json(OUT / 'gallery_audit.json', result)
    print(result)
    return result


if __name__ == '__main__':
    main()
