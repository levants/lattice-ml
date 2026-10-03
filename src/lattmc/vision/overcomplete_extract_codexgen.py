"""Cache final normalized DINOv2 register-model patch activations."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torchvision.transforms import functional as tf
from transformers import AutoModel

from lattmc.vision.overcomplete_data_codexgen import load
from lattmc.vision.overcomplete_fetch_codexgen import ROOT, digest
from lattmc.vision.paths_codexgen import experiment_root


BACKBONE = (experiment_root('patch_contexts')
            / 'checkpoints/saev_register_backbone')


def backbone() -> torch.nn.Module:
    """Load the frozen local vision backbone for dense feature extraction."""
    from lattmc.vision.patch_weights_codexgen import ensure
    ensure(BACKBONE / 'model.safetensors')
    model = AutoModel.from_pretrained(
        BACKBONE, local_files_only=True, attn_implementation='eager')
    return model.eval().requires_grad_(False)


def dense(model: torch.nn.Module, images: np.ndarray) -> torch.Tensor:
    """Extract patch tokens while omitting classification and register tokens.
    """
    pixels = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2)
    pixels = tf.normalize(pixels.float() / 255,
                          [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    hidden = model(pixels).last_hidden_state
    assert hidden.shape[1:] == (261, 768), hidden.shape
    return hidden[:, 5:].contiguous()


def extract(name: str) -> None:
    """Save dense patch features in deterministic dataset chunks."""
    torch.set_num_threads(4)
    sample, records = load(name)
    out = ROOT / 'activations' / name
    out.mkdir(parents=True, exist_ok=True)
    model = backbone()
    start_time = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(records), 10):
            path = out / f'dense_{start:04d}_codexgen.npz'
            if path.exists():
                continue
            values = dense(model, sample['images'][start:start + 10])
            np.savez_compressed(path, dense=values.numpy(),
                                positions=np.arange(
                                    start, start + len(values)))
            print(name, start + len(values),
                  round(time.monotonic() - start_time, 1), flush=True)
    manifest = {'backbone': json.loads(
        (BACKBONE / 'metadata.json').read_text()),
        'activation': 'final layer, final LayerNorm, remove CLS+4 registers',
        'preprocessing': 'RGB; bicubic short edge 256; center crop 224',
        'grid': [16, 16], 'dimension': 768,
        'dataset_sha256': digest(ROOT / f'dataset/{name}_codexgen.npz')}
    (out / 'metadata_codexgen.json').write_text(
        json.dumps(manifest, indent=2) + '\n')


def load_dense(name: str) -> np.ndarray:
    """Concatenate cached dense patch-feature chunks."""
    paths = sorted((ROOT / 'activations' / name).glob('dense_*.npz'))
    assert paths, name
    return np.concatenate([np.load(p)['dense'] for p in paths])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    extract(parser.parse_args().name)
