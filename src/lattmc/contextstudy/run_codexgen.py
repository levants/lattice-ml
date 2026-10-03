"""Replay full-code queries and test their token witnesses using cached data.

Run from the repository with its uv environment:
    PYTHONPATH=src .venv/bin/python -m lattmc.contextstudy.run_codexgen
No training, downloads, or changes to the historical caches are performed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/contextstudy-mpl')

import numpy as np
from scipy import sparse
import torch
from transformers import AutoTokenizer

ROOT = Path(os.environ.get('MY_PAPERS_REPOSITORY', Path.cwd())).resolve()
DATA = ROOT / 'data/activation_studies/context_v1'
OLD = ROOT / 'data/activation_studies/legacy/lattconference'
OLD /= 'activation_results'
TOKEN = ROOT / 'notebooks/transcoders/data/transcoders/gpt2'
TOKEN /= 'owt_tokens/owt_tokens_torch.pt'
SEED = 20261001
CASES = {
    'nyc': (3457, [1, 2, 3], r'\b(new|york|city)\b'),
    'rio': (5411, [15, 16, 17], r'\b(rio|janeiro)\b'),
    'animals': (4042, [8, 82], r'\b(cats?|dogs?)\b'),
    'sports': (1924, [15, 39, 94],
               r'\b(cleveland|clippers|cavaliers)\b'),
}


def digest(path: Path | str) -> str:
    """Compute the SHA-256 digest of an experiment artifact."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state_digest(module: torch.nn.Module) -> str:
    """Hash a module's named parameter tensors in a deterministic order."""
    if hasattr(module, 'sae'):
        module = module.sae
    result = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        result.update(name.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(str(tensor.dtype).encode())
        result.update(tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def shuffled(
    tokens: torch.Tensor,
    position: int,
    seed: int,
    prefix: bool,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Preserve target position, token multiset, and special-token slots."""
    result = tokens.clone()
    start, end = (1, position) if prefix else (position + 1, len(tokens))
    slots = np.array([p for p in range(start, end)
                      if int(tokens[p]) != 50256], dtype=int)
    permuted = np.random.default_rng(seed).permutation(slots)
    result[slots] = tokens[permuted]
    assert int(result[position]) == int(tokens[position])
    assert sorted(result.tolist()) == sorted(tokens.tolist())
    return result, slots.tolist(), permuted.tolist()


def run(
    kind: str,
    layer: int,
    tokens: torch.Tensor,
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Replay contextual token interventions and save measured activations."""
    sys.path.insert(0, str(ROOT))
    from src.lattmc.tc.transcoder_analyzers_codexgen import (
        init_transcoder_or_sae,
    )

    stem = f'{kind}_layer{layer}'
    saved = np.load(OLD / f'{stem}.npz')
    prior = json.loads((OLD / f'{stem}.json').read_text())
    assert digest(TOKEN) == prior['token_sha256']
    folder = ('sae/data/sae' if kind == 'sae'
              else 'transcoders/data/transcoders')
    matrix_path = ROOT / 'notebooks' / folder / 'gpt2' / f'V{layer}.npz'
    assert digest(matrix_path) == prior['matrix_sha256']
    matrix = sparse.load_npz(matrix_path).tocsc()
    model = init_transcoder_or_sae(
        model_name='gpt2-small', layers=[layer], device=torch.device('cpu'),
        tr_or_sae=kind == 'tc',
    )
    report = dict(
        kind=kind, layer=layer, seed=SEED, corpus_size=len(tokens),
        token_sha256=digest(TOKEN), matrix_sha256=digest(matrix_path),
        query_archive_sha256=digest(OLD / f'{stem}.npz'),
        backbone_state_sha256=state_digest(model.model),
        surrogate_state_sha256=state_digest(model.transcoders[layer]),
        source_sha256=digest(__file__),
        versions={p: metadata.version(p) for p in (
            'torch', 'numpy', 'scipy', 'transformers', 'transformer-lens',
            'sae-lens',
        )},
        cases=[],
    )
    arrays = {}
    for case, (source, positions, pattern) in CASES.items():
        query = saved[f'{case}_meet_query']
        active = np.flatnonzero(query > 0)
        assert len(active), 'Zero queries require a separate null protocol.'
        fresh_source = model.run_layers(tokens[source], [layer])[layer]
        fresh_query = fresh_source[positions].min(axis=0)
        assert np.allclose(query, fresh_query, atol=1e-3, rtol=1e-3)
        ids = saved[f'{case}_meet_1.0'].astype(int)
        eligible = np.array([i for i in ids if i != source and not re.search(
            pattern, texts[i], re.IGNORECASE,
        )], dtype=int)
        chosen = sorted(np.random.default_rng(SEED).choice(
            eligible, min(3, len(eligible)), replace=False,
        ).tolist())
        frequencies = [(int((matrix[:, j].data >= query[j]).sum()), int(j))
                       for j in active]
        _, feature = min(frequencies)
        arrays[f'{case}_query'] = query
        arrays[f'{case}_eligible'] = eligible
        result = dict(
            case=case, source_row=source, source_positions=positions,
            exclusion_pattern=pattern, extent_count=len(ids),
            non_source_count=len(ids) - int(source in ids),
            eligible_count=len(eligible), active_coordinates=len(active),
            selected_feature=feature,
            feature_threshold=float(query[feature]),
            source_query_max_error=float(abs(query - fresh_query).max()),
            rows=[],
        )
        for row in chosen:
            key = f'{case}_{row}'
            values = model.run_layers(tokens[row], [layer])[layer]
            cached = matrix.getrow(row).toarray().ravel()
            assert np.allclose(values.max(axis=0), cached,
                               atol=1e-3, rtol=1e-3)
            witnesses = np.argmax(values[:, active], axis=0)
            position = int(np.argmax(values[:, feature]))
            arrays[f'{key}_active'] = active
            arrays[f'{key}_codes'] = values[:, active]
            arrays[f'{key}_target'] = values[position]
            token_text = [tokenizer.decode([int(t)]) for t in tokens[row]]
            entry = dict(
                row=row, tokens=tokens[row].tolist(), pieces=token_text,
                feature=feature, position=position,
                threshold=float(query[feature]),
                activation=float(values[position, feature]),
                all_coordinate_witnesses=witnesses.tolist(),
                fresh_member=bool(np.all(values.max(axis=0) >= query)),
                summary_max_error=float(
                    abs(values.max(axis=0) - cached).max(),
                ),
                prefix=[],
            )
            for repeat in range(3):
                seed = SEED + row * 10 + repeat
                altered, slots, permutation = shuffled(
                    tokens[row], position, seed, True,
                )
                code = model.run_layers(altered, [layer])[layer][position]
                arrays[f'{key}_prefix{repeat}'] = code
                entry['prefix'].append(dict(
                    seed=seed, slots=slots, permutation=permutation,
                    changed_tokens=int((altered != tokens[row]).sum()),
                    activation=float(code[feature]),
                ))
            altered, slots, permutation = shuffled(
                tokens[row], position, SEED + row, False,
            )
            suffix = model.run_layers(altered, [layer])[layer][position]
            arrays[f'{key}_suffix'] = suffix
            entry['suffix'] = dict(
                slots=slots, permutation=permutation,
                changed_tokens=int((altered != tokens[row]).sum()),
                max_code_error=float(abs(suffix - values[position]).max()),
            )
            result['rows'].append(entry)
            print(stem, case, row, 'j,p=', feature, position,
                  'a=', entry['activation'], 'prefix=',
                  [round(p['activation'], 4) for p in entry['prefix']],
                  'suffix=', entry['suffix']['max_code_error'], flush=True)
        report['cases'].append(result)
    np.savez_compressed(DATA / f'{stem}.npz', **arrays)
    (DATA / f'{stem}.json').write_text(json.dumps(report, indent=2) + '\n')
    del model, matrix
    gc.collect()


def main() -> None:
    """Replay full-code queries and test their token witnesses using cached
    data.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    tokens = torch.load(TOKEN, map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    for kind in ('sae', 'tc'):
        for layer in (8, 11):
            run(kind, layer, tokens, texts, tokenizer)


if __name__ == '__main__':
    main()
