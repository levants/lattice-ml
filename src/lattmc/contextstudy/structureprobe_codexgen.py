"""Probe title-boundary and naming hypotheses using label-withheld traces.

Select sixteen members and sixteen nonmembers per fixed historical query.
Only split IDs, codes and text enter this stage; dataset categories are
reserved for a subsequent comparison. Historical source labels are known.
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
from lattmc.contextstudy.operations_codexgen import digest, extent, save


def run(root: Path, out: Path) -> None:
    """Replay fixed-size stratified samples without reading class labels."""
    torch.set_num_threads(4)
    base = root / 'data/activation_studies/families_v2/dbpedia_14'
    original = json.loads((base / 'design.json').read_text())
    texts = json.loads((base / 'texts.local.json').read_text())
    history = root / 'data/activation_studies/latticemethods_v1'
    gallery = json.loads((history / 'document_gallery.json').read_text())
    design_path = history / 'dbpedia_14_design.json'
    test = np.array(json.loads(design_path.read_text())['test'])
    records, arrays = [], {}
    for family in ('pythia_topk', 'smol_topk'):
        cfg = FAMILIES[family]
        matrix = sparse.load_npz(base / f'{family}_max.npz').tocsr()
        definition = next(r['query'] for r in gallery['records']
                          if r['family'] == family and
                          r['query_name'] == 'join')
        query = np.zeros(matrix.shape[1], dtype=np.float32)
        query[definition['coordinates']] = definition['values']
        assert len(definition['coordinates']) == 1
        coordinate = definition['coordinates'][0]
        mask = extent(matrix.tocsc(), query)
        rng = np.random.default_rng(20261004)
        chosen = sorted(np.concatenate([
            rng.choice(test[mask[test]], 16, replace=False),
            rng.choice(test[~mask[test]], 16, replace=False)]).tolist())
        sae = load_surrogate(family, cfg, original['pinned'], 'mps')
        revision = original['pinned'][cfg['model']]['revision']
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model'], revision=revision, local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').to('mps').eval()
        tokenizer = AutoTokenizer.from_pretrained(
            cfg['model'], revision=revision, local_files_only=True)
        forward = attach(model, cfg)
        tokens = np.load(base / f'{family}_tokens.local.npz')
        with torch.inference_mode():
            for start in sorted({i // 4 * 4 for i in chosen}):
                ids = torch.tensor(tokens['input_ids'][start:start + 4],
                                   device='mps')
                valid_mask = torch.tensor(
                    tokens['attention_mask'][start:start + 4], device='mps')
                x, _ = forward(ids, valid_mask)
                codes = sae.encode(x).cpu().numpy()
                for row in chosen:
                    if not start <= row < start + 4:
                        continue
                    valid = np.flatnonzero(tokens['attention_mask'][row])
                    valid = valid[valid != 0]
                    code = codes[row - start]
                    pooled = code[valid].max(axis=0)
                    old = matrix[row].toarray()[0]
                    np.testing.assert_allclose(pooled, old,
                                               rtol=1e-5, atol=1e-5)
                    tokenized = tokenizer(texts[row],
                        add_special_tokens=False, return_offsets_mapping=True)
                    expected = tokenized['input_ids'][:len(valid)]
                    assert expected == tokens['input_ids'][row][valid].tolist()
                    boundary = texts[row].find('.  ')
                    positions = [int(valid[i]) for i, (a, b) in enumerate(
                        tokenized['offset_mapping'][:len(valid)])
                        if a <= boundary < b]
                    values = code[:, coordinate]
                    whole = valid[values[valid].astype(float) >=
                                  float(query[coordinate])].tolist()
                    pieces = [tokenizer.decode([int(t)],
                        clean_up_tokenization_spaces=False)
                        for t in tokens['input_ids'][row]]
                    key = f'{family}_{row}'
                    arrays[key] = values
                    records.append(dict(
                        family=family, row=row, query=definition,
                        cached_member=bool(mask[row]),
                        replay_member=bool(whole), whole=whole,
                        valid=valid.tolist(), pieces=pieces,
                        tokens=tokens['input_ids'][row].tolist(),
                        raw_text=tokenizer.decode(
                            tokens['input_ids'][row][valid],
                            clean_up_tokenization_spaces=False),
                        title_boundary=positions,
                        boundary_witness=bool(set(positions) & set(whole)),
                        replay_max_abs_error=float(abs(pooled-old).max()),
                        trace_key=key))
        del model, sae
        torch.mps.empty_cache()
    trace_path = out / 'structure_traces.npz'
    np.savez_compressed(trace_path, **arrays)
    save(out / 'structure_probe.json', dict(
        records=records, seed=20261004, traces_sha256=digest(trace_path),
        source_sha256=digest(Path(__file__)),
        selection='16 members and 16 nonmembers per fixed query; no labels',
        limitation='Exploratory hypothesis from previously seen examples.'))
    for family in ('pythia_topk', 'smol_topk'):
        for member in (True, False):
            selected = [r for r in records if r['family'] == family and
                        r['cached_member'] == member]
            print(family, member, 'boundary witnesses',
                  sum(r['boundary_witness'] for r in selected), '/',
                  len(selected), flush=True)


def main() -> None:
    """Run the bounded structural probe with the original MPS batches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.out)


if __name__ == '__main__':
    main()
