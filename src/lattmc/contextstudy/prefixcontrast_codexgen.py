"""Test an exploratory verb-complement hypothesis with local input edits.

Single-token replacements preserve length and target position. This is
input sensitivity of a surrogate coordinate, not an internal feature
intervention or evidence of a feature-mediated behavioral pathway.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/latticesemantics/mpl')

import numpy as np
import torch

from lattmc.contextstudy.operations_codexgen import digest, save

EDITS = [(3457, 65, 62, ' put', ' introduced'),
         (5342, 108, 107, ' put', ' get'),
         (9155, 36, 32, ' put', ' used'),
         (9486, 22, 21, ' placing', ' spending'),
         (10441, 120, 119, ' puts', ' leaves'),
         (14210, 55, 53, ' put', ' apply'),
         (25220, 117, 116, ' put', ' score')]
FAMILY = {3457: ' placed', 5342: ' place', 9155: ' placed',
          9486: ' putting', 10441: ' places', 14210: ' place',
          25220: ' place'}


def run(root: Path, out: Path) -> None:
    """Run seven prefix contrasts, verb-family and suffix controls."""
    from lattmc.tc.transcoder_analyzers_codexgen import init_transcoder_or_sae

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    path = root / 'notebooks/transcoders/data/transcoders/gpt2'
    path /= 'owt_tokens/owt_tokens_torch.pt'
    tokens = torch.load(path, map_location='cpu', weights_only=True)
    model = init_transcoder_or_sae(model_name='gpt2-small', layers=[11],
                                  device=torch.device('cpu'), tr_or_sae=True)
    records, arrays = [], {}
    for row, target, position, before, after in EDITS:
        ids = tokens[row].clone()
        assert model.tokenizer.decode([int(ids[position])]) == before
        replacement = model.tokenizer.encode(after, add_special_tokens=False)
        assert len(replacement) == 1
        changed = ids.clone()
        changed[position] = replacement[0]
        related = ids.clone()
        related_ids = model.tokenizer.encode(
            FAMILY[row], add_special_tokens=False)
        assert len(related_ids) == 1
        related[position] = related_ids[0]
        suffix = ids.clone()
        suffix[-1] = 50256 if int(ids[-1]) != 50256 else 0
        original = np.asarray(model.run_layers(ids, [11])[11])
        edited = np.asarray(model.run_layers(changed, [11])[11])
        family_code = np.asarray(model.run_layers(related, [11])[11])
        control = np.asarray(model.run_layers(suffix, [11])[11])
        assert np.array_equal(original[:target + 1], control[:target + 1])
        for name, code in [('original', original), ('edited', edited),
                           ('family', family_code), ('suffix', control)]:
            arrays[f'{row}_{name}'] = code[target]
        records.append(dict(
            row=row, target=target, edited_position=position,
            before=before, after=after, coordinate=4355,
            original=float(original[target, 4355]),
            edited=float(edited[target, 4355]),
            family_replacement=FAMILY[row],
            family_activation=float(family_code[target, 4355]),
            suffix_control=float(control[target, 4355]),
            full_target_code_suffix_equal=True,
            original_prefix=model.tokenizer.decode(ids[:target + 1],
                clean_up_tokenization_spaces=False),
            edited_prefix=model.tokenizer.decode(changed[:target + 1],
                clean_up_tokenization_spaces=False),
            original_ids=ids.tolist(), edited_ids=changed.tolist()))
    trace_path = out / 'prefix_contrast_traces.npz'
    np.savez_compressed(trace_path, **arrays)
    save(out / 'prefix_contrast.json', dict(
        traces_sha256=digest(trace_path),
        records=records, source_sha256=digest(Path(__file__)),
        design='Exploratory after observing seven verb-complement cases.',
        condition='GPT-2-small ReLU transcoder, block 11, MLP input code',
        limitation='Input edits; no latent ablation or output behavior test.'))
    print([(r['row'], r['original'], r['edited'],
            r['family_activation']) for r in records])


def main() -> None:
    """Run the fixed exploratory prefix contrast on cached token rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.out)


if __name__ == '__main__':
    main()
