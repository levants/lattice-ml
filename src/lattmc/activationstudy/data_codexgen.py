"""Build fixed, deduplicated external-corpus subsamples and token arrays."""

from __future__ import annotations
from typing import Any

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import unicodedata

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import pandas as pd
from scipy import sparse
from transformers import AutoTokenizer

from .common_codexgen import SEED, save_json, sha256


def normalized(text: str) -> str:
    """Normalize Unicode, case, and word spacing for duplicate detection."""
    return ' '.join(re.findall(r'\w+', unicodedata.normalize(
        'NFKC', text).lower()))


def deduplicate(
    records: list[dict[str, Any]],
    token_arrays: dict[str, dict[str, np.ndarray]],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Group exact text/token copies and >=.8 word-shingle near copies."""
    parent = list(range(len(records)))

    def root(i: int) -> int:
        """Find a duplicate component representative with path compression."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        """Merge the duplicate components containing two record indices."""
        parent[root(i)] = root(j)

    seen = {}
    for i, row in enumerate(records):
        keys = [('text', normalized(row['text']))]
        keys += [(name, array['input_ids'][i][array['attention_mask'][i]]
                  .tobytes()) for name, array in token_arrays.items()]
        for key in keys:
            if key in seen:
                union(i, seen[key])
            seen[key] = i
    vocabulary, rr, cc = {}, [], []
    for i, row in enumerate(records):
        words = normalized(row['text']).split()
        shingles = {' '.join(words[j:j + 5])
                    for j in range(max(0, len(words) - 4))}
        for shingle in shingles:
            rr.append(i)
            cc.append(vocabulary.setdefault(shingle, len(vocabulary)))
    matrix = sparse.csr_matrix((np.ones(len(rr), dtype=np.int32), (rr, cc)),
                               shape=(len(records), len(vocabulary)))
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    intersection = sparse.triu(matrix @ matrix.T, k=1).tocoo()
    for i, j, value in zip(intersection.row, intersection.col,
                           intersection.data):
        if value / (sizes[i] + sizes[j] - value) >= .8:
            union(int(i), int(j))
    groups = {}
    for i in range(len(records)):
        groups.setdefault(root(i), []).append(i)
    priority = dict(test=0, calibration=1, train=2)
    kept, removed = [], []
    for ids in groups.values():
        if len({records[i]['label'] for i in ids}) > 1:
            removed.append(dict(rows=ids, reason='conflicting labels'))
            continue
        winner = min(ids, key=lambda i: (priority[records[i]['split']], i))
        kept.append(winner)
        if len(ids) > 1:
            removed.append(dict(rows=[i for i in ids if i != winner],
                                retained=winner, reason='duplicate group'))
    return sorted(kept), removed


def prepare(output: Path, protocol: Path) -> None:
    """Save deduplicated corpus splits, token arrays, and provenance."""
    tokenizers = {name: AutoTokenizer.from_pretrained(repo,
                   local_files_only=True) for name, repo in dict(
                       gpt2='gpt2', gemma2='google/gemma-2-2b').items()}
    for offset, (dataset, sizes) in enumerate(dict(
            ag_news=(150, 50, 100), dbpedia_14=(75, 25, 50)).items()):
        raw = output / 'raw' / dataset
        source = json.loads((raw / 'source.json').read_text())
        records = []
        rng = np.random.default_rng(SEED + offset)
        for official in ('train', 'test'):
            frames = []
            for entry in source['files']:
                if entry['local'].startswith(official + '-'):
                    assert sha256(raw / entry['local']) == entry['sha256']
                    frames.append(pd.read_parquet(raw / entry['local']))
            frame = pd.concat(frames, ignore_index=True)
            for label in sorted(frame.label.unique()):
                ids = np.flatnonzero(frame.label.to_numpy() == label)
                ids = rng.permutation(ids)
                count = (sizes[0] + sizes[1] if official == 'train'
                         else sizes[2])
                for rank, source_id in enumerate(ids[:count]):
                    row = frame.iloc[int(source_id)]
                    text = (row['text'] if dataset == 'ag_news' else
                            row['title'] + '. ' + row['content'])
                    split = ('test' if official == 'test' else
                             'train' if rank < sizes[0] else 'calibration')
                    records.append(dict(
                        source_split=official, source_id=int(source_id),
                        label=int(label), split=split, text=text,
                        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    ))
            del frame, frames
        arrays = {}
        for name, tokenizer in tokenizers.items():
            encoded = tokenizer([r['text'] for r in records],
                                add_special_tokens=False, truncation=True,
                                max_length=127)['input_ids']
            pad = tokenizer.pad_token_id
            if pad is None:
                pad = tokenizer.eos_token_id
            ids = np.full((len(records), 128), pad, dtype=np.int64)
            mask = np.zeros_like(ids, dtype=bool)
            for i, tokens in enumerate(encoded):
                tokens = [tokenizer.bos_token_id] + tokens
                ids[i, :len(tokens)] = tokens
                mask[i, :len(tokens)] = True
            arrays[name] = dict(input_ids=ids, attention_mask=mask)
        kept, removed = deduplicate(records, arrays)
        target = output / dataset
        target.mkdir(parents=True, exist_ok=True)
        public = [{k: v for k, v in records[i].items() if k != 'text'}
                  for i in kept]
        save_json(target / 'texts.local.json', [records[i]['text']
                                               for i in kept])
        for name, array in arrays.items():
            np.savez_compressed(target / f'{name}_tokens.local.npz',
                                **{k: v[kept] for k, v in array.items()})
        counts = {split: {str(label): sum(r['split'] == split
                   and r['label'] == label for r in public)
                  for label in sorted({r['label'] for r in public})}
                  for split in ('train', 'calibration', 'test')}
        manifest = dict(dataset=dataset, source=source, rows=public,
                        requested_per_class=sizes, counts=counts,
                        removed=removed, initial_rows=len(records),
                        protocol_sha256=sha256(protocol),
                        source_sha256=sha256(__file__), seed=SEED,
                        versions={k: metadata.version(k) for k in
                                  ('numpy', 'pandas', 'transformers')})
        save_json(target / 'design.json', manifest)
        print(dataset, counts, 'removed', len(records) - len(kept), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output, args.protocol)
