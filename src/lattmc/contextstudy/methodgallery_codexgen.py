"""Replay fixed selected-component examples without changing query rules.

The first NaturalPlace source group and rank-1/rank-1 components are fixed
by the protocol. Display choices follow corpus order, not semantic quality.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from scipy import sparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lattmc.activationstudy.families_adapters_codexgen import (
    attach, load_surrogate,
)
from lattmc.activationstudy.families_config_codexgen import FAMILIES
from lattmc.contextstudy.operations_codexgen import (
    digest, encoded, extent, project, save, witnesses,
)


def main() -> None:
    """Replay Pythia and SmolLM2 examples and preserve numerical mismatches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    base = args.root / 'data/activation_studies/families_v2/dbpedia_14'
    design = json.loads((args.out / 'dbpedia_14_design.json').read_text())
    original = json.loads((base / 'design.json').read_text())
    group = next(g for g in design['groups']
                 if g['label'] == 7 and g['group'] == 0)
    records, arrays = [], {}
    for family in ('pythia_topk', 'smol_topk'):
        cfg = FAMILIES[family]
        matrix = sparse.load_npz(base / f'{family}_max.npz').tocsr()
        values = matrix[group['sources'][:2]].toarray()
        p, q = project(values[0], (1,)), project(values[1], (1,))
        queries = dict(left=p, right=q, meet=np.minimum(p, q),
                       join=np.maximum(p, q))
        masks = {name: extent(matrix.tocsc(), query)
                 for name, query in queries.items()}
        choices = []
        for name, mask in masks.items():
            ids = [i for i in design['test'] if mask[i]]
            if ids:
                choices.append((name, ids[0]))
            else:
                records.append(dict(family=family, query_name=name,
                                    empty=True))
        positive = [i for i in design['test'] if masks['join'][i]
                    and design['labels'][i] == 7]
        if positive:
            choices.append(('join_category', positive[0]))
        else:
            records.append(dict(family=family, query_name='join_category',
                                empty=True))
        ids = [i for i in design['test'] if not masks['join'][i]]
        if ids:
            choices.append(('nonmember', ids[0]))
        sae = load_surrogate(family, cfg, original['pinned'], 'mps')
        revision = original['pinned'][cfg['model']]['revision']
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model'], revision=revision, local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').to('mps').eval()
        tokenizer = AutoTokenizer.from_pretrained(
            cfg['model'], revision=revision, local_files_only=True)
        run = attach(model, cfg)
        tokens = np.load(base / f'{family}_tokens.local.npz')
        with torch.inference_mode():
            for start in sorted({i // 4 * 4 for _, i in choices}):
                ids = torch.tensor(tokens['input_ids'][start:start + 4],
                                   device='mps')
                mask = torch.tensor(tokens['attention_mask'][start:start + 4],
                                    device='mps')
                x, _ = run(ids, mask)
                codes = sae.encode(x).cpu().numpy()
                for name, row in choices:
                    if not start <= row < start + 4:
                        continue
                    query = queries['join' if name in (
                        'nonmember', 'join_category') else name]
                    valid = np.flatnonzero(tokens['attention_mask'][row])
                    valid = valid[valid != 0]
                    code = codes[row - start]
                    pooled = code[valid].max(axis=0)
                    old = matrix[row].toarray()[0]
                    error = float(np.max(abs(pooled - old)))
                    np.testing.assert_allclose(pooled, old,
                                               rtol=1e-5, atol=1e-5)
                    trace = code[valid]
                    record = witnesses(trace, query)
                    record['whole'] = valid[record['whole']].tolist()
                    record['partial'] = valid[record['partial']].tolist()
                    key = f'{family}_{name}_{row}'
                    active = np.flatnonzero(query)
                    arrays[key] = code[:, active]
                    expected = bool(masks[
                        'join' if name in ('nonmember', 'join_category')
                        else name][row])
                    pieces = [tokenizer.decode([int(t)],
                        clean_up_tokenization_spaces=False)
                        for t in tokens['input_ids'][row]]
                    records.append(dict(
                        family=family, query_name=name, row=row,
                        sources=group['sources'][:2], query=encoded(query),
                        configuration=cfg, label=design['labels'][row],
                        identity=design['rows'][row], valid=valid.tolist(),
                        pieces=pieces,
                        tokens=tokens['input_ids'][row].tolist(),
                        replay_max_abs_error=error, cached_member=expected,
                        replay_changed=expected != record['member'],
                        trace_key=key, **record))
        del model, sae
    trace_path = args.out / 'document_gallery_traces.npz'
    np.savez_compressed(trace_path, **arrays)
    save(args.out / 'document_gallery.json', dict(
        records=records, traces_sha256=digest(trace_path),
        source_sha256=digest(Path(__file__)),
        selection='NaturalPlace group 0; ranks 1/1; first test ID per mask.'))


if __name__ == '__main__':
    main()
