"""Pinned native RA-SAE inference with a verified compact dictionary."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy import sparse

from lattmc.vision import overcomplete_compat_codexgen  # noqa: F401
from overcomplete.sae.modules import MLPEncoder

from lattmc.vision.overcomplete_extract_codexgen import load_dense
from lattmc.vision.overcomplete_fetch_codexgen import ROOT, digest


NAME = 'pretrained_ra'
SHA = '3cf20c7e8a97e063273e5116a049f965cb74799e2abded0c5e77f2ec30e27faa'
FOLDER = ROOT / 'checkpoints/pretrained'


def compact() -> None:
    """Extract and save the compact pretrained surrogate checkpoint."""
    target = FOLDER / 'compact_codexgen.pt'
    if target.exists():
        return
    path = FOLDER / 'RA-SAE-DINOv2-32k.pth'
    assert digest(path) == SHA
    # This is the pinned upstream full-module checkpoint, not remote code.
    original = torch.load(path, map_location='cpu', mmap=True,
                          weights_only=False).module
    print(original.encoder, original.top_k, flush=True)
    dictionary = original.dictionary
    print('dictionary training', dictionary.training, flush=True)
    with torch.inference_mode():
        # Apply the native training-to-eval fusion in bounded row blocks.
        rows = []
        for start in range(0, len(dictionary.W), 512):
            w = torch.relu(dictionary.W[start:start + 512])
            w = w / (w.sum(1, keepdim=True) + 1e-8)
            relax = dictionary.Relax[start:start + 512]
            factor = (dictionary.delta / relax.norm(dim=1, keepdim=True))
            relax = relax * factor.clamp(max=1)
            rows.append((w @ dictionary.C + relax)
                        * dictionary.multiplier.exp())
        fused = torch.cat(rows)
        # Verify the row-block fusion against native dictionary evaluation
        # on a small row subset, with the same class and stored parameters.
        import copy
        subset = copy.copy(dictionary)
        subset._parameters = dictionary._parameters.copy()
        subset.W = torch.nn.Parameter(dictionary.W[:32].clone())
        subset.Relax = torch.nn.Parameter(dictionary.Relax[:32].clone())
        subset.training = True
        reference = subset.get_dictionary()
        fusion_error = float((reference - fused[:32]).abs().max())
        assert fusion_error < 1e-4, fusion_error
        dictionary._fused_dictionary = fused
        dictionary.training = False
        original.running_threshold = 0.829
        original.eval()
        x = torch.from_numpy(load_dense('imagenette')[0, :32])
        _, z, prediction = original(x)
        pre, rectified = original.encoder(x)
        explicit = rectified * (rectified >= 0.829)
        code_error = float((z - explicit).abs().max())
        output_error = float((prediction - explicit @ fused).abs().max())
        assert max(code_error, output_error) < 1e-4
        enc = original.encoder
        parameters = {'input_shape': enc.input_size,
                      'n_components': enc.n_components,
                      'hidden_dim': enc.hidden_dim,
                      'nb_blocks': enc.nb_blocks,
                      'residual': enc.residual}
        saved = {'encoder': enc.state_dict(), 'dictionary': fused,
                 'parameters': parameters, 'threshold': 0.829}
        torch.save(saved, target)
    receipt = {'source_sha256': SHA, 'native_class': type(original).__name__,
               'encoder': str(enc), 'stored_top_k': original.top_k,
               'evaluation_threshold': 0.829,
               'fusion_max_error': fusion_error,
               'code_max_error': code_error,
               'reconstruction_max_error': output_error,
               'compact_sha256': digest(target),
               'note': 'Native BatchTopK threshold inference; not fixed k=5.'}
    (FOLDER / 'verification_codexgen.json').write_text(
        json.dumps(receipt, indent=2) + '\n')


def extract(dataset: str) -> None:
    """Cache pretrained surrogate codes and reconstruction statistics."""
    torch.set_num_threads(4)
    compact()
    checkpoint = torch.load(FOLDER / 'compact_codexgen.pt',
                            weights_only=True, map_location='cpu')
    encoder = MLPEncoder(**checkpoint['parameters'],
                         norm_layer=torch.nn.LayerNorm).eval()
    encoder.load_state_dict(checkpoint['encoder'], strict=True)
    dictionary = checkpoint['dictionary']
    folder = ROOT / 'codes' / NAME
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{dataset}_codexgen.npz'
    if target.exists():
        return
    values = torch.from_numpy(load_dense(dataset).reshape(-1, 768))
    local = torch.load(ROOT / 'checkpoints/topk_k32_s0/model_codexgen.pt',
                       weights_only=True, map_location='cpu')
    blocks, errors, totals = [], [], []
    with torch.inference_mode():
        for i, x in enumerate(values.split(128)):
            _, rectified = encoder(x)
            z = rectified * (rectified >= checkpoint['threshold'])
            output = z @ dictionary
            assert torch.isfinite(z).all() and (z >= 0).all()
            blocks.append(sparse.csr_matrix(z.numpy()))
            errors.extend((output - x).square().sum(1).tolist())
            totals.extend((x - local['mean']).square().sum(1).tolist())
            if i % 100 == 0:
                print(dataset, i * 128, flush=True)
    matrix = sparse.vstack(blocks, format='csr')
    np.savez_compressed(target, data=matrix.data, indices=matrix.indices,
                        indptr=matrix.indptr, shape=matrix.shape,
                        error=np.array(errors).reshape(-1, 256).sum(1),
                        baseline=np.array(totals).reshape(-1, 256).sum(1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    extract(parser.parse_args().dataset)
