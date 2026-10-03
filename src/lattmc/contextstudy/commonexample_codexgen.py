"""Exact document-meet illustration on five fixed DBpedia source items."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from scipy import sparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lattmc.activationstudy.common_codexgen import save_json, sha256
from lattmc.activationstudy.families_adapters_codexgen import (
    attach, load_surrogate,
)
from lattmc.activationstudy.families_config_codexgen import FAMILIES
from lattmc.fca.lattice_utils import meet_all, upper_mask
from .witnesses_codexgen import classify, ROOT

BASE = ROOT / 'data/activation_studies/families_v2/dbpedia_14'
DEST = ROOT / 'data/activation_studies/tablecontexts_v1'


def main() -> None:
    """Exact document-meet illustration on five fixed DBpedia source items."""
    torch.set_num_threads(4)
    DEST.mkdir(parents=True, exist_ok=True)
    design = json.loads((BASE / 'design.json').read_text())
    texts = json.loads((BASE / 'texts.local.json').read_text())
    sources = [i for i, row in enumerate(design['rows'])
               if row['split'] == 'train' and row['label'] == 7][:5]
    assert sources == [700, 701, 702, 703, 704]
    result = dict(
        dataset='dbpedia_14', sources=sources,
        identities=[design['rows'][i] for i in sources],
        texts=[texts[i] for i in sources],
        selection='First five NaturalPlace training rows in cache order; '
                  'display first three, retain all five in records.',
        query='Full coordinatewise meet of five tokenwise-join summaries; '
              'alpha=1; no coordinate selection or calibration.',
        interpretation='Separate canonical concepts; no cross-model '
                       'infomorphism or feature alignment asserted.',
        design_sha256=sha256(BASE / 'design.json'),
        source_sha256=sha256(__file__), families={},
    )
    for family in ('pythia_topk', 'smol_topk'):
        cfg = FAMILIES[family]
        tokens_path = BASE / f'{family}_tokens.local.npz'
        tokens = np.load(tokens_path)
        path = BASE / f'{family}_extraction.json'
        extraction = json.loads(path.read_text())
        assert extraction['configuration'] == cfg
        assert extraction['token_sha256'] == sha256(tokens_path)
        matrix_path = BASE / f'{family}_max.npz'
        assert extraction['files']['max']['sha256'] == sha256(matrix_path)
        matrix = sparse.load_npz(matrix_path)
        query = meet_all(matrix[sources].toarray())
        active = np.flatnonzero(query)
        assert len(active)
        extent = np.flatnonzero(upper_mask(
            query[active], matrix[:, active].toarray()))
        closed = meet_all(matrix[extent].toarray())
        assert np.array_equal(closed, query)
        assert set(sources) <= set(extent)
        # Preserve the original extraction's batch boundaries and device.
        sae = load_surrogate(family, cfg, design['pinned'], 'mps')
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model'], revision=design['pinned'][cfg['model']]['revision'],
            local_files_only=True, dtype=torch.float32,
            attn_implementation='eager').to('mps').eval()
        tokenizer = AutoTokenizer.from_pretrained(
            cfg['model'], revision=design['pinned'][cfg['model']]['revision'],
            local_files_only=True)
        capture = attach(model, cfg)
        rows, arrays = [], {}
        maximum_error = 0.
        with torch.inference_mode():
            for start in sorted({i // 4 * 4 for i in sources}):
                ids = torch.tensor(tokens['input_ids'][start:start + 4],
                                   device='mps')
                mask = torch.tensor(tokens['attention_mask'][start:start + 4],
                                    device='mps')
                x, _ = capture(ids, mask)
                codes = sae.encode(x).cpu().numpy()
                for i in sources:
                    if not start <= i < start + 4:
                        continue
                    code = codes[i - start]
                    valid = np.flatnonzero(tokens['attention_mask'][i])
                    valid = valid[valid != 0]
                    pooled = code[valid].max(0)
                    old = matrix[i].toarray()[0]
                    error = float(np.max(np.abs(pooled - old)))
                    maximum_error = max(maximum_error, error)
                    np.testing.assert_allclose(pooled, old,
                                               rtol=1e-5, atol=1e-5)
                    record = classify(code[valid][:, active], query[active],
                                      positions=valid)
                    # No tolerance is applied to query membership.
                    assert record['member'], (family, i, record['score'])
                    ids_row = tokens['input_ids'][i].tolist()
                    arrays[str(i)] = code[:, active]
                    row = dict(
                        display_id=f'D{family[0].upper()}{i}', family=family,
                        layer=cfg['layer'], row=i, sources=sources,
                        source_kind='document', mode='full', case=7,
                        alpha=1., coordinates=active.tolist(),
                        query=query[active].tolist(), tokens=ids_row,
                        pieces=[tokenizer.decode([t]) for t in ids_row],
                        decoded_text=tokenizer.decode(ids_row),
                        valid_positions=valid.tolist(),
                        dataset_identity=design['rows'][i],
                        diagnostic=int(valid[0]), **record,
                    )
                    rows.append(row)
        np.savez_compressed(DEST / f'{family}.npz', query=query, **arrays)
        result['families'][family] = dict(
            configuration=cfg, sae_config=sae.cfg.to_dict(),
            extraction_sha256=sha256(path), tokens_sha256=sha256(tokens_path),
            matrix_sha256=sha256(matrix_path), extent=extent.tolist(),
            positive_coordinates=len(active),
            maximum_replay_error=maximum_error,
            rows=rows, trace_sha256=sha256(DEST / f'{family}.npz'),
            checks=dict(source_inclusion=True, intent_closure=True,
                        exact_token_membership=True),
        )
        print(family, len(active), extent.tolist(), maximum_error,
              [r['status'] for r in rows], flush=True)
        del model, sae
    save_json(DEST / 'records.json', result)


if __name__ == '__main__':
    main()
