"""Replay newly isolated closure coordinates and retain numerical witnesses."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch
from scipy import sparse

from lattmc.contextstudy.operations_codexgen import (
    digest, save, witnesses,
)

CASES = [('research_three', 'full', 'meet', 'additional_sample'),
         ('mixed_pair', 'rank1', 'join', 'all'),
         ('cat_dog', 'rank23', 'join', 'additional_sample')]


def run(root: Path, out: Path) -> None:
    """Replay fixed TC11 follow-ups, recording both u and h at every token.

    All added research and selected Cat/dog samples and all six sports
    join members are retained. This is bounded CPU inference with unchanged
    checkpoints, not a new corpus or a semantic relevance annotation.
    """
    from lattmc.tc.transcoder_analyzers_codexgen import init_transcoder_or_sae

    data = json.loads((out / 'gpt2_tc11.json').read_text())
    arrays = np.load(out / 'gpt2_tc11_members.npz')
    token_path = root / 'notebooks/transcoders/data/transcoders/gpt2'
    token_path /= 'owt_tokens/owt_tokens_torch.pt'
    tokens = torch.load(token_path, map_location='cpu', weights_only=True)
    path = root / 'notebooks/transcoders/data/transcoders/gpt2/V11.npz'
    matrix = sparse.load_npz(path).tocsr()
    selections = []
    for group, rule, op, selection in CASES:
        record = next(r for r in data['records'] if r['group'] == group
                      and r['rule'] == rule and r['operation'] == op)
        ids = (arrays[record['members_key']].tolist() if selection == 'all'
               else record[selection])
        selections.extend((record, row) for row in ids)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    model = init_transcoder_or_sae(model_name='gpt2-small', layers=[11],
                                  device=torch.device('cpu'), tr_or_sae=True)
    records, traces = [], {}
    for case, row in selections:
        code = np.asarray(model.run_layers(tokens[row], [11])[11])
        pooled = code.max(axis=0)
        cached = matrix[row].toarray()[0]
        error = float(np.max(abs(pooled - cached)))
        np.testing.assert_allclose(pooled, cached, rtol=1e-3, atol=1e-3)
        pieces = [model.tokenizer.decode([int(t)],
            clean_up_tokenization_spaces=False) for t in tokens[row]]
        for query_name, definition in [('u', case['query']), ('h', case['h'])]:
            query = np.zeros(code.shape[1], dtype=code.dtype)
            query[definition['coordinates']] = definition['values']
            evidence = witnesses(code, query)
            key = f'{case["members_key"]}_{query_name}_{row}'
            columns = np.array(definition['coordinates'], dtype=int)
            traces[key] = code[:, columns]
            cached_member = bool(np.all(cached[columns].astype(float) >=
                                        query[columns].astype(float)))
            records.append(dict(
                group=case['group'], rule=case['rule'],
                operation=case['operation'], query_name=query_name,
                row=row, query=definition, trace_key=key,
                tokens=tokens[row].tolist(), pieces=pieces,
                raw_text=model.tokenizer.decode(tokens[row],
                    clean_up_tokenization_spaces=False),
                cached_member=cached_member,
                replay_changed=cached_member != evidence['member'],
                replay_max_abs_error=error,
                maxima=pooled[columns].tolist(),
                failed_coordinates=columns[
                    pooled[columns].astype(float) <
                    query[columns].astype(float)].tolist(), **evidence))
    path = out / 'tc11_witnesses.npz'
    np.savez_compressed(path, **traces)
    save(out / 'tc11_witnesses.json', dict(
        records=records, traces_sha256=digest(path),
        source_sha256=digest(Path(__file__))))
    print('Query/item records:', len(records), 'changed:',
          sum(r['replay_changed'] for r in records), flush=True)


def main() -> None:
    """Execute only the recorded bounded closure-coordinate witness study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.out)


if __name__ == '__main__':
    main()
