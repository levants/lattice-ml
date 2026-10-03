"""Compare saved text-first interpretations with labels and reading packets.

This module never chooses queries or semantic interpretations from labels.
It verifies the saved constituent sample texts and adds a later descriptive
label comparison. Earlier source categories were known to the analyst.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from lattmc.contextstudy.operations_codexgen import digest, save


def compare(root: Path, out: Path) -> dict[str, object]:
    """Verify original decodings and tabulate later dataset categories.

    Reads immutable interpretation notes, cached tokens and dataset labels.
    Writes only the subsequent comparison record; no inference is run.
    """
    interpretation = out / 'interpretations_before_labels.json'
    assert interpretation.exists()
    data = json.loads((out / 'structure_probe.json').read_text())
    base = root / 'data/activation_studies/families_v2/dbpedia_14'
    design = json.loads((base / 'design.json').read_text())
    result: dict[str, object] = {
        'interpretation_sha256': digest(interpretation),
        'comparison': ('Dataset categories consulted after the recorded '
                       'text/trace interpretation.'),
        'families': {},
    }
    families = {}
    for family in ('pythia_topk', 'smol_topk'):
        rows = [r['row'] for r in data['records'] if
                r['family'] == family and r['cached_member']]
        counts = Counter(design['rows'][r]['label'] for r in rows)
        families[family] = dict(
            member_label_counts=dict(sorted(counts.items())),
            distinct_labels=len(counts), naturalplace_count=counts[7],
            sample_size=len(rows))
    result['families'] = families
    token_path = root / 'notebooks/transcoders/data/transcoders/gpt2'
    tokens = torch.load(token_path / 'owt_tokens/owt_tokens_torch.pt',
                        map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    packet = json.loads((out / 'component_reading.json').read_text())
    for records in packet.values():
        for row, text in records.items():
            assert text == tokenizer.decode(tokens[int(row)],
                clean_up_tokenization_spaces=False)
    save(out / 'labels_after_reading.json', result)
    return result


def main() -> None:
    """Run only the post-reading label comparison and decoding audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(compare(args.root, args.out))


if __name__ == '__main__':
    main()
