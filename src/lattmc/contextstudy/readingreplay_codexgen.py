"""Replay explicitly selected exploratory cases and save token evidence.

Case selection follows recorded close reading, not a held-out success
criterion. Original pooled membership and replay membership are separate.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/latticesemantics/mpl')

import numpy as np
import torch
from scipy import sparse

from lattmc.contextstudy.operations_codexgen import (
    Record, digest, save, witnesses,
)

CASES = {
    'tc11': [('research_three', 'full', 'meet', 'sample'),
             ('research_three', 'rank23', 'join', 'all'),
             ('mixed_pair', 'rank1', 'join', 'all'),
             ('biological_four', 'rank1', 'join', 'all'),
             ('mixed_three', 'rank3', 'join', 'all'),
             ('cat_dog', 'full', 'meet', 'all')],
    'sae11': [('new_de', 'rank23', 'join', 'all'),
              ('research_three', 'full', 'meet', 'sample')],
    'tc8': [('research_three', 'full', 'meet', 'sample')],
    'tc0': [('research_three', 'full', 'meet', 'sample')],
    'sae8': [('research_three', 'full', 'meet', 'sample')],
}


def replay(root: Path, out: Path, condition: str) -> None:
    """Replay selected corpus rows and save coordinate-level inequalities.

    Uses the existing CPU loader and immutable source traces. The code
    records discrepancies instead of modifying non-source cached rows.
    """
    from lattmc.tc.transcoder_analyzers_codexgen import init_transcoder_or_sae

    kind = 'sae' if condition.startswith('sae') else 'tc'
    layer = int(condition[len(kind):])
    data = json.loads((out / f'gpt2_{condition}.json').read_text())
    arrays = np.load(out / f'gpt2_{condition}_members.npz')
    previous = root / 'data/activation_studies/latticemethods_v1'
    source_traces = np.load(previous / f'gpt2_{condition}_traces.npz')
    token_path = root / 'notebooks/transcoders/data/transcoders/gpt2'
    token_path /= 'owt_tokens/owt_tokens_torch.pt'
    tokens = torch.load(token_path, map_location='cpu', weights_only=True)
    matrix_path = root / 'notebooks' / (
        'sae/data/sae' if kind == 'sae' else 'transcoders/data/transcoders')
    matrix_path /= f'gpt2/V{layer}.npz'
    matrix = sparse.load_npz(matrix_path).tocsr()
    selections = []
    for group, rule, op, selection in CASES[condition]:
        case = next(r for r in data['records']
                    if r['group'] == group and r['rule'] == rule)
        query = case[op]
        ids = (arrays[query['array_key']].tolist() if selection == 'all'
               else query['sampled_ids'])
        selections.extend((case, op, row) for row in ids)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    model = init_transcoder_or_sae(model_name='gpt2-small', layers=[layer],
                                  device=torch.device('cpu'),
                                  tr_or_sae=kind == 'tc')
    records, traces = [], {}
    for row in sorted({row for _, _, row in selections}):
        key = f'trace_{row}'
        code = (source_traces[key] if key in source_traces.files else
                np.asarray(model.run_layers(tokens[row], [layer])[layer]))
        pooled = code.max(axis=0)
        cached = matrix[row].toarray()[0]
        error = float(np.max(abs(pooled - cached)))
        np.testing.assert_allclose(pooled, cached, rtol=1e-3, atol=1e-3)
        pieces = [model.tokenizer.decode([int(t)],
            clean_up_tokenization_spaces=False) for t in tokens[row]]
        for case, op, selected_row in selections:
            if row != selected_row:
                continue
            definition = case[op]['query']
            query = np.zeros(code.shape[1], dtype=code.dtype)
            query[definition['coordinates']] = definition['values']
            evidence = witnesses(code, query)
            coordinates = np.array(definition['coordinates'], dtype=int)
            name = f'{case["group"]}_{case["rule"]}_{op}_{row}'
            traces[name] = code[:, coordinates]
            comparison = []
            for j in coordinates:
                positions = np.flatnonzero(
                    code[:, j].astype(float) >= float(query[j]))
                comparison.append(dict(
                    coordinate=int(j), requirement=float(query[j]),
                    item_max=float(pooled[j]), cached_max=float(cached[j]),
                    witness_positions=positions.tolist(),
                    witness_pieces=[pieces[i] for i in positions]))
            records.append(dict(
                group=case['group'], rule=case['rule'], operation=op,
                condition=condition, row=row, extent=case[op]['count'],
                query=definition, members_key=case[op]['array_key'],
                tokens=tokens[row].tolist(), pieces=pieces,
                raw_text=model.tokenizer.decode(tokens[row],
                    clean_up_tokenization_spaces=False),
                trace_key=name, inequalities=comparison,
                cached_member=True, replay_changed=not evidence['member'],
                replay_max_abs_error=error, **evidence))
        print(condition, row, 'checked', flush=True)
    path = out / f'{condition}_witnesses.npz'
    np.savez_compressed(path, **traces)
    save(out / f'{condition}_witnesses.json', dict(
        records=records, selections=CASES[condition],
        selection_status='exploratory after text reading; all/sampled as set',
        traces_sha256=digest(path), source_sha256=digest(Path(__file__))))
    del model
    gc.collect()


def main() -> None:
    """Execute one bounded offline replay condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--condition', choices=CASES, required=True)
    args = parser.parse_args()
    replay(args.root, args.out, args.condition)


if __name__ == '__main__':
    main()
