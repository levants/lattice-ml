"""Reuse the frozen documents and audit new tokenizers before inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .common_codexgen import save_json, sha256
from .data_codexgen import deduplicate
from .families_config_codexgen import FAMILIES


def prepare(
    previous: Path,
    output: Path,
    protocol: Path,
    downloads: Path,
) -> None:
    """Prepare family-specific token caches using the frozen corpus design."""
    pinned = json.loads(downloads.read_text())
    tokenizers = {}
    for name, cfg in FAMILIES.items():
        kwargs = {'revision': pinned[cfg['model']]['revision']}
        tokenizers[name] = AutoTokenizer.from_pretrained(
            cfg['model'], local_files_only=True, **kwargs)
    for dataset in ('ag_news', 'dbpedia_14'):
        source = previous / dataset
        design = json.loads((source / 'design.json').read_text())
        texts = json.loads((source / 'texts.local.json').read_text())
        records = [dict(row, text=text) for row, text in
                   zip(design['rows'], texts, strict=True)]
        arrays, identities = {}, {}
        for name, tokenizer in tokenizers.items():
            prefix = tokenizer.bos_token_id
            prefix_kind = 'bos'
            if prefix is None:
                prefix, prefix_kind = tokenizer.eos_token_id, 'eos_delimiter'
            assert prefix is not None
            pad = tokenizer.pad_token_id
            if pad is None:
                pad = tokenizer.eos_token_id
            encoded = tokenizer(texts, add_special_tokens=False,
                                truncation=True, max_length=127)['input_ids']
            ids = np.full((len(texts), 128), pad, np.int64)
            mask = np.zeros_like(ids, dtype=bool)
            for i, sequence in enumerate(encoded):
                tokens = [prefix] + sequence
                assert len(tokens) > 1
                ids[i, :len(tokens)] = tokens
                mask[i, :len(tokens)] = True
            arrays[name] = dict(input_ids=ids, attention_mask=mask)
            identities[name] = dict(prefix_id=prefix, prefix_kind=prefix_kind,
                                     pad_id=pad, model=FAMILIES[name]['model'])
        kept, removed = deduplicate(records, arrays)
        target = output / dataset
        target.mkdir(parents=True, exist_ok=True)
        save_json(target / 'texts.local.json', [texts[i] for i in kept])
        for name, array in arrays.items():
            np.savez_compressed(target / f'{name}_tokens.local.npz',
                                **{key: value[kept]
                                   for key, value in array.items()})
        save_json(target / 'design.json', dict(
            dataset=dataset, rows=[design['rows'][i] for i in kept],
            source=design['source'], previous_indices=kept, removed=removed,
            parent_design_sha256=sha256(source / 'design.json'),
            protocol_sha256=sha256(protocol), tokenizers=identities,
            source_sha256=sha256(__file__), pinned=pinned))
        print(dataset, 'rows', len(kept), 'removed', len(texts)-len(kept),
              flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--downloads', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.previous, args.output, args.protocol, args.downloads)
