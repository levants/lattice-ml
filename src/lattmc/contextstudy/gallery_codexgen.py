"""Replay ranked external examples and retain measured token highlights."""

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

from lattmc.activationstudy.common_codexgen import save_json, sha256
from lattmc.activationstudy.families_adapters_codexgen import (
    attach, load_surrogate,
)
from lattmc.activationstudy.families_config_codexgen import FAMILIES
from .depth_codexgen import OLD, OUT


def main(family: str) -> None:
    """Build contextual retrieval galleries for a checkpoint family."""
    torch.set_num_threads(4)
    source = OLD / 'dbpedia_14'
    OUT.mkdir(parents=True, exist_ok=True)
    design = json.loads((source / 'design.json').read_text())
    texts = json.loads((source / 'texts.local.json').read_text())
    report = json.loads((source / f'{family}_max_results.json').read_text())
    scores = np.load(source / f'{family}_max_scores.npz')['graded']
    token_path = source / f'{family}_tokens.local.npz'
    tokens = np.load(token_path)
    matrix = sparse.load_npz(source / f'{family}_max.npz').tocsr()
    cfg = FAMILIES[family]
    pinned = design['pinned']
    sae = load_surrogate(family, cfg, pinned, 'mps')
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model'], revision=pinned[cfg['model']]['revision'],
        local_files_only=True, dtype=getattr(torch, cfg['dtype']),
        attn_implementation='eager').to('mps').eval()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model'], revision=pinned[cfg['model']]['revision'],
        local_files_only=True)
    capture = attach(model, cfg)
    records, arrays = [], {}
    for label in (7, 9):
        index = next(i for i, r in enumerate(report['records'])
                     if r['label'] == label and r['shot'] == 3
                     and r['repeat'] == 0)
        task = report['records'][index]
        active = np.sort(task['selected_coordinates'])
        assert len(active)
        query = matrix[task['positive']].toarray().min(0)[active]
        order = np.lexsort((report['test_ids'], -scores[index]))[:3]
        for rank, pos in enumerate(order, 1):
            row = report['test_ids'][pos]
            start = 4 * (row // 4)
            ids = torch.tensor(tokens['input_ids'][start:start + 4],
                               device='mps')
            mask = torch.tensor(tokens['attention_mask'][start:start + 4],
                                device='mps')
            with torch.inference_mode():
                x, _ = capture(ids, mask)
                z = sae.encode(x)[row - start].float().cpu().numpy()
            valid = tokens['attention_mask'][row].copy()
            valid[0] = False
            maximum = z[valid].max(0)
            cached = matrix.getrow(row).toarray().ravel()
            assert np.allclose(maximum, cached, atol=1e-3, rtol=1e-3), (
                family, row, abs(maximum - cached).max())
            ratio = maximum[active] / query
            weakest = int(np.argmin(ratio))
            feature = int(active[weakest])
            values = z[:, feature].copy()
            values[~valid] = -np.inf
            p = int(np.argmax(values))
            score = float(ratio.min())
            threshold = task['thresholds']['graded']
            assert np.isclose(score, scores[index, pos], rtol=1e-3, atol=1e-3)
            assert (score >= threshold) == (scores[index, pos] >= threshold)
            key = f'{label}_{rank}'
            arrays[key] = z[:, active]
            ids = tokens['input_ids'][row].tolist()
            lo, hi = max(1, p - 16), min(int(valid.sum()) + 1, p + 17)
            entry = dict(
                family=family, layer=cfg['layer'], task_label=label,
                rank=rank, row=row, dataset_identity=design['rows'][row],
                positive=task['positive'],
                source_titles=[texts[i].split('. ')[0]
                               for i in task['positive']],
                selected_coordinates=active.tolist(), query=query.tolist(),
                feature=feature, position=p, token_ids=ids,
                valid_positions=np.flatnonzero(valid).tolist(),
                activation=float(z[p, feature]),
                feature_query=float(query[weakest]), score=score,
                calibrated_threshold=threshold, accepted=score >= threshold,
                cached_score=float(scores[index, pos]),
                max_summary_error=float(abs(maximum - cached).max()),
                text=texts[row],
                before=tokenizer.decode(ids[lo:p]),
                highlighted=tokenizer.decode(ids[p:p + 1]),
                after=tokenizer.decode(ids[p + 1:hi]),
            )
            records.append(entry)
            print(family, label, rank, row, 'j,p', feature, p,
                  repr(entry['highlighted']), 'score', round(score, 3),
                  'class', entry['dataset_identity']['label'], flush=True)
    np.savez_compressed(OUT / f'{family}_gallery.npz', **arrays)
    save_json(OUT / f'{family}_gallery.json', dict(
        source_sha256=sha256(__file__), records=records,
        design_sha256=sha256(source / 'design.json'),
        tokens_sha256=sha256(token_path),
        extraction_sha256=sha256(source / f'{family}_extraction.json'),
        traces_sha256=sha256(OUT / f'{family}_gallery.npz'),
        results_sha256=sha256(source / f'{family}_max_results.json'),
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=FAMILIES)
    main(parser.parse_args().family)
