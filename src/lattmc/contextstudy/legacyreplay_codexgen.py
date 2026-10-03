"""Targeted replay of identifiable historical Cat/dog table items only."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from scipy import sparse
import torch
from transformers import AutoTokenizer

from lattmc.activationstudy.common_codexgen import sha256, save_json
from .witnesses_codexgen import DEST, ROOT, classify

# These identities are recovered from the existing excerpts, not ranked
# or chosen for their classifications. Multiple matches remain unresolved.
ITEMS = {
    '8-exact-tc': [[1946], [2003], [3887, 4978, 5645], [4042]],
    '8-exact-sae': [[1197], [2835], [1313, 2954], [4042]],
    '8-floor-tc': [[510], [915], [1036]],
    '8-floor-sae': [[694], [2128], [2990]],
    '11-exact-tc': [[4042], [4558], [12388]],
    '11-exact-sae': [[327], [409], [740]],
    '11-floor-tc': [[275], [2311], [2553, 3135]],
    '11-floor-sae': [[11], [94], [922, 1015]],
}


def main() -> None:
    """Targeted replay of identifiable historical Cat/dog table items only."""
    sys.path.insert(0, str(ROOT))
    from src.lattmc.tc.transcoder_analyzers_codexgen import (
        init_transcoder_or_sae,
    )
    from .run_codexgen import state_digest
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    token_path = ROOT / 'notebooks/transcoders/data/transcoders/gpt2'
    token_path /= 'owt_tokens/owt_tokens_torch.pt'
    tokens = torch.load(token_path, map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    records, arrays, provenance = [], {}, []
    for kind in ('tc', 'sae'):
        for layer in (8, 11):
            model = init_transcoder_or_sae(
                model_name='gpt2-small', layers=[layer],
                device=torch.device('cpu'), tr_or_sae=kind == 'tc')
            old = ROOT / 'data/activation_studies/legacy/lattconference'
            old /= 'activation_results' / Path(f'{kind}_layer{layer}.npz')
            query = np.load(old)['animals_meet_query']
            source = model.run_layers(tokens[4042], [layer])[layer]
            assert np.allclose(source[[8, 82]].min(0), query,
                               atol=1e-3, rtol=1e-3)
            folder = 'transcoders' if kind == 'tc' else 'sae'
            matrix_path = ROOT / 'notebooks' / folder / 'data' / folder
            matrix_path /= f'gpt2/V{layer}.npz'
            matrix = sparse.load_npz(matrix_path).tocsc()
            active = np.flatnonzero(query > 0)
            floors = []
            for j in active:
                values = matrix.data[matrix.indptr[j]:matrix.indptr[j+1]]
                floors.append(values[values > 0].min())
            for mode in ('exact', 'floor'):
                key = f'{layer}-{mode}-{kind}'
                q = query[active] if mode == 'exact' else np.array(floors)
                for i, candidates in enumerate(ITEMS[key], 1):
                    identity = key + f'-{i}'
                    if len(candidates) != 1:
                        records.append(dict(id=identity, group='legacy',
                            family=kind, layer=layer, mode=mode, ordinal=i,
                            status='U', candidates=candidates))
                        continue
                    row = candidates[0]
                    values = model.run_layers(tokens[row], [layer])[layer]
                    cached = matrix.getrow(row).toarray().ravel()
                    error = float(abs(values.max(0) - cached).max())
                    assert np.allclose(values.max(0), cached,
                                       atol=1e-3, rtol=1e-3)
                    result = classify(values[:, active], q)
                    arrays[identity] = values[:, active]
                    records.append(dict(
                        id=identity, group='legacy', family=kind,
                        layer=layer, mode=mode, ordinal=i, case='animals',
                        row=row, sources=[4042], source_positions=[8, 82],
                        source_kind='token', alpha=1.,
                        coordinates=active.tolist(), query=q.tolist(),
                        tokens=tokens[row].tolist(),
                        pieces=[tokenizer.decode([int(t)])
                                for t in tokens[row]],
                        valid_positions=list(range(128)), diagnostic=64,
                        summary_max_error=error, trace_key=identity,
                        **result,
                    ))
                    print(identity,row,result['status'],flush=True)
            provenance.append(dict(kind=kind, layer=layer,
                query_sha256=sha256(old), matrix_sha256=sha256(matrix_path),
                backbone_sha256=state_digest(model.model),
                surrogate_sha256=state_digest(model.transcoders[layer])))
            del model, matrix
            gc.collect()
    DEST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEST / 'legacy.npz', **arrays)
    save_json(DEST / 'legacy.json', dict(
        source_sha256=sha256(__file__), token_sha256=sha256(token_path),
        trace_sha256=sha256(DEST / 'legacy.npz'), provenance=provenance,
        records=records, limitation='Four excerpts have ambiguous row IDs; '
        'their historical text remains unclassified and unhighlighted.',
    ))


if __name__ == '__main__':
    main()
