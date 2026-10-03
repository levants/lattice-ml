"""Post hoc fixed-budget ablation; does not retune the primary benchmark.

Added after inspecting the primary comparison to check whether conjunctions
improve over the first ranked coordinate. Uses the existing coordinate order.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from .conference_experiments_codexgen import (
    OUT, ROOT, save_json, sha256, statistics,
)


def run(kind: str, layer: int) -> None:
    """Evaluate ablations on the cached checkpoint activations."""
    design = json.loads((OUT / 'design.json').read_text())
    folder = 'sae' if kind == 'sae' else 'transcoders'
    cache = ROOT / f'notebooks/{folder}/data/{folder}/gpt2/V{layer}.npz'
    matrix = sparse.load_npz(cache).tocsc()
    stem = f'{kind}_{layer}'
    queries = np.load(OUT / f'{stem}_queries.npz')
    test = np.array(design['splits']['test'])
    matrix = matrix[test].tocsc()
    output, saved = [], {}
    for number, task in enumerate(design['tasks']):
        query = queries[f'{number}_full']
        ordered = queries[f'{number}_order'][:64]
        y = np.isin(test, task['positives'])
        if len(ordered):
            scores = matrix[:, ordered].toarray().astype(np.float64)
            scores /= query[ordered]
            np.minimum.accumulate(scores, axis=1, out=scores)
        values = {}
        for budget in (1, 4, 16, 64):
            score = (scores[:, min(budget, len(ordered)) - 1]
                     if len(ordered) else np.ones(len(test)))
            values[str(budget)] = statistics(y, score)
            saved[f'{number}_{budget}'] = score
        output.append(dict(phrase=task['phrase'], repeat=task['repeat'],
                           metrics=values))
    save_json(OUT / f'ablation_{stem}.json', dict(
        analysis='post hoc fixed-budget ablation',
        source_sha256=sha256(__file__),
        primary_sha256=sha256(OUT / f'{stem}.json'), runs=output,
    ))
    np.savez_compressed(OUT / f'ablation_{stem}_scores.npz', **saved)
    print(stem, 'fixed-budget ablation complete', flush=True)


if __name__ == '__main__':
    for model in ('sae', 'tc'):
        for layer in (0, 8, 11):
            run(model, layer)
