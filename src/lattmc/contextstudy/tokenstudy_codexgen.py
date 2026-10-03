"""Replay contextual GPT-2 codes and evaluate full and selected joins.

Uses original token IDs and existing model-loading helpers. Source rows
are refreshed in memory only after a numerical consistency check. Raw
traces and every query are saved; AI-edited text never defines highlights.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/latticemethods/mpl')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from scipy import sparse
import torch

from lattmc.contextstudy.operations_codexgen import (
    GRID, Record, digest, encoded, extent, project, relation, save, witnesses,
)

SOURCES = dict(New=(3457, 1), York=(3457, 2), City=(3457, 3),
               Rio=(5411, 15), de=(5411, 16), Janeiro=(5411, 17),
               Cat=(4042, 8), dog=(4042, 82),
               Cleveland=(1924, 15), Clippers=(1924, 39),
               Cavaliers=(1924, 94))
PAIRS = [('New', 'York'), ('Rio', 'Janeiro'), ('Rio', 'de'),
         ('New', 'de'), ('York', 'de'), ('Cat', 'dog')]


def evaluate(root: Path, out: Path, kind: str, layer: int,
             protocol: Path) -> None:
    """Replay sources, run the fixed query grid, and inspect fixed examples.

    Writes records, complete source traces and selected gallery traces to
    out. Assertions reject inconsistent source caches rather than silently
    changing experimental conditions.
    """
    from lattmc.tc.transcoder_analyzers_codexgen import init_transcoder_or_sae

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    token_path = root / 'notebooks/transcoders/data/transcoders/gpt2'
    token_path /= 'owt_tokens/owt_tokens_torch.pt'
    tokens = torch.load(token_path, map_location='cpu', weights_only=True)
    matrix_path = root / 'notebooks' / (
        'sae/data/sae' if kind == 'sae' else 'transcoders/data/transcoders')
    matrix_path /= f'gpt2/V{layer}.npz'
    model = init_transcoder_or_sae(model_name='gpt2-small', layers=[layer],
                                  device=torch.device('cpu'),
                                  tr_or_sae=kind == 'tc')
    module = model.transcoders[layer]
    actual = module.sae if hasattr(module, 'sae') else module
    decoder = actual.W_dec.detach().float().cpu().numpy()
    norms = np.linalg.norm(decoder, axis=1)
    cfg = actual.cfg
    config = cfg.to_dict() if hasattr(cfg, 'to_dict') else vars(cfg)
    # Checkpoints can contain device/dtype objects; stringify metadata only.
    config = json.loads(json.dumps(config, default=str))
    source_ids = sorted({row for row, _ in SOURCES.values()})
    traces = {row: np.asarray(model.run_layers(tokens[row], [layer])[layer])
              for row in source_ids}
    matrix = sparse.load_npz(matrix_path).tolil()
    refresh = []
    for row, code in traces.items():
        assert code.shape == (128, 24576)
        assert np.isfinite(code).all() and (code >= 0).all()
        fresh = code.max(axis=0)
        old = matrix[row].toarray()[0]
        np.testing.assert_allclose(fresh, old, rtol=1e-3, atol=1e-3)
        refresh.append(dict(row=row, max_abs_error=float(abs(fresh-old).max()),
                            support_changes=int(((fresh > 0) !=
                                                 (old > 0)).sum())))
        matrix[row] = fresh
    matrix = matrix.tocsc()
    matrix.eliminate_zeros()
    vectors = {name: traces[row][pos] for name, (row, pos) in SOURCES.items()}
    records = []
    arrays = {f'trace_{row}': code for row, code in traces.items()}
    for left_name, right_name in PAIRS:
        u, v = vectors[left_name], vectors[right_name]
        masks = dict(left=extent(matrix, u), right=extent(matrix, v),
                     meet=extent(matrix, np.minimum(u, v)),
                     join=extent(matrix, np.maximum(u, v)))
        assert np.array_equal(masks['join'], masks['left'] & masks['right'])
        assert np.all(~(masks['left'] | masks['right']) | masks['meet'])
        masks['extension'] = masks['meet'] & ~(masks['left'] | masks['right'])
        row = dict(left_name=left_name, right_name=right_name,
                   source_positions=[SOURCES[left_name], SOURCES[right_name]],
                   full={k: int(m.sum()) for k, m in masks.items()},
                   meet_positive=int((np.minimum(u, v) > 0).sum()),
                   selected=[])
        for name, mask in masks.items():
            arrays[f'{left_name}_{right_name}_{name}'] = np.flatnonzero(mask)
        for lr, rr in GRID:
            p, q = project(u, lr), project(v, rr)
            selected = dict(left_ranks=list(lr), right_ranks=list(rr))
            if p is None or q is None:
                row['selected'].append(dict(**selected,
                                            skipped='insufficient_rank'))
                continue
            a, b = extent(matrix, p), extent(matrix, q)
            combined = extent(matrix, np.maximum(p, q))
            assert np.array_equal(combined, a & b)
            selected.update(left=encoded(p), right=encoded(q),
                            relation=relation(a, b),
                            same_coordinate=bool(np.array_equal(
                                np.flatnonzero(p), np.flatnonzero(q))))
            row['selected'].append(selected)
        records.append(row)
    triples = []
    for names in [('New', 'York', 'City'), ('Rio', 'de', 'Janeiro'),
                  ('Cleveland', 'Clippers', 'Cavaliers')]:
        values = np.array([vectors[n] for n in names])
        a = [extent(matrix, v) for v in values]
        meet = extent(matrix, values.min(axis=0))
        joined = extent(matrix, values.max(axis=0))
        assert np.array_equal(joined, np.logical_and.reduce(a))
        triples.append(dict(names=names, meet=int(meet.sum()),
                            join=int(joined.sum()), positive=int(
                                (values.min(axis=0) > 0).sum()),
                            extension=int((meet & ~np.logical_or.reduce(
                                a)).sum())))
    gallery = []
    if layer in (8, 11):
        p, q = project(vectors['New'], (2,)), project(vectors['de'], (3,))
        queries = dict(left=p, right=q, meet=np.minimum(p, q),
                       join=np.maximum(p, q))
        for name, query in queries.items():
            members = extent(matrix, query)
            choices = [i for i in np.flatnonzero(members)
                       if i not in source_ids]
            if not choices:
                gallery.append(dict(query_name=name, empty=True))
                continue
            row = int(choices[0])
            code = np.asarray(model.run_layers(tokens[row], [layer])[layer])
            arrays[f'gallery_{name}_{row}'] = code
            record = witnesses(code, query)
            gallery.append(dict(
                query_name=name, row=row, query=encoded(query),
                tokens=tokens[row].tolist(),
                pieces=[model.tokenizer.decode([int(t)],
                    clean_up_tokenization_spaces=False) for t in tokens[row]],
                cached_member=True, replay_changed=not record['member'],
                **record))
        query = queries['join']
        members = extent(matrix, query)
        choices = [i for i in np.flatnonzero(~members) if i not in source_ids]
        if choices:
            row = int(choices[0])
            code = np.asarray(model.run_layers(tokens[row], [layer])[layer])
            arrays[f'gallery_nonmember_{row}'] = code
            record = witnesses(code, query)
            gallery.append(dict(
                query_name='nonmember', row=row, query=encoded(query),
                tokens=tokens[row].tolist(),
                pieces=[model.tokenizer.decode([int(t)],
                    clean_up_tokenization_spaces=False) for t in tokens[row]],
                cached_member=False, replay_changed=record['member'],
                **record))
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / f'gpt2_{kind}{layer}_traces.npz'
    np.savez_compressed(trace_path, **arrays)
    save(out / f'gpt2_{kind}{layer}.json', dict(
        kind=kind, layer=layer, records=records, triples=triples,
        sources={n: dict(row=r, position=p,
                         token=int(tokens[r, p]),
                         piece=model.tokenizer.decode([int(tokens[r, p])]))
                 for n, (r, p) in SOURCES.items()},
        config=config, decoder_norm=dict(min=float(norms.min()),
            max=float(norms.max()), median=float(np.median(norms))),
        source_refresh=refresh, gallery=gallery,
        token_sha256=digest(token_path),
        matrix_sha256=digest(matrix_path), traces_sha256=digest(trace_path),
        protocol_sha256=digest(protocol), source_sha256=digest(Path(__file__)),
        tolerance=0, corpus_size=len(tokens), included_positions='all 128'))
    print(kind, layer, 'complete', len(gallery), 'gallery rows', flush=True)
    del model, traces, matrix
    gc.collect()


def audit_text(root: Path, out: Path) -> None:
    """Compare corrected text with raw decoded cached token rows by ID."""
    from transformers import AutoTokenizer

    path = root / 'owt_tokens/dataset_corrected.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    tokens_path = root / 'notebooks/transcoders/data/transcoders/gpt2'
    tokens_path /= 'owt_tokens/owt_tokens_torch.pt'
    tokens = torch.load(tokens_path, map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    ids = [int(r['id']) for r in rows]
    assert len(set(ids)) == len(ids)
    assert set(ids) == set(range(len(tokens)))
    changed = [int(r['id']) for r in rows
               if r['text'] != decoded[int(r['id'])]]
    local = root / 'owt_tokens/owt_tokens_torch.pt'
    save(out / 'corrected_text_audit.json', dict(
        rows=len(rows), changed_from_raw_decode=len(changed),
        changed_ids=changed, corrected_sha256=digest(path),
        tokens_sha256=digest(tokens_path),
        root_token_file_equal=torch.equal(tokens, torch.load(
            local, map_location='cpu', weights_only=True)),
        policy='Corrected prose is not activation-aligned evidence.',
        csv_ids='exactly cover cached row IDs; not an alignment proof'))


def main() -> None:
    """Run one bounded offline condition or audit the corrected text file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--kind', choices=('sae', 'tc'))
    parser.add_argument('--layer', type=int, choices=(0, 8, 11))
    args = parser.parse_args()
    if args.kind:
        evaluate(args.root, args.out, args.kind, args.layer, args.protocol)
    else:
        audit_text(args.root, args.out)


if __name__ == '__main__':
    main()
