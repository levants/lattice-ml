"""Independent cache, closure, pixel-control and model-contract checks."""

from __future__ import annotations
from typing import Any
from collections.abc import Sequence

import argparse
import json

import numpy as np
import torch
from scipy import sparse
from torchvision.transforms import functional as tf

from lattmc.vision.patch_contexts_codexgen import read
from lattmc.vision.patch_extract_codexgen import load
from lattmc.vision.patch_models_codexgen import Adapter, ROOT, dataset


def verify(name: str, native: bool = False) -> dict[str, Any]:
    """Verify cached patch-query artifacts and optional native inference."""
    sample, codes, features, metadata = read(name)
    root = ROOT / name
    expected = dataset(name)
    for key in expected:
        np.testing.assert_array_equal(sample[key], expected[key])
    _, records = load(name)
    flat = codes.reshape(-1, len(features))
    count = 0
    for r in records:
        matrix = sparse.csr_matrix((r['data'], r['indices'], r['indptr']),
                                   shape=r['shape'])
        assert np.isfinite(matrix.data).all() and (matrix.data > 0).all()
        selected = matrix[:, features].toarray().reshape(
            len(r['positions']), -1, len(features))
        np.testing.assert_array_equal(selected, codes[r['positions']])
        for k, pos in enumerate(r['positions']):
            sites = codes.shape[1]
            pooled = matrix[k * sites:(k + 1) * sites].max(0).toarray()[0]
            np.testing.assert_array_equal(pooled, r['pooled'][k])
        count += matrix.shape[0]
    for i in range(10):
        with np.load(root / f'contexts/class_{i:02d}_codexgen.npz') as c:
            masks = []
            for query, closed in zip(c['queries'], c['closed']):
                mask = np.all(flat >= query, axis=1)
                intent = flat[mask].min(0) if mask.any() else flat.max(0)
                np.testing.assert_array_equal(intent, closed)
                np.testing.assert_array_equal((flat >= closed).all(1), mask)
                masks.append(mask.reshape(codes.shape[:2]))
            np.testing.assert_array_equal(masks, c['patch_extents'])
            np.testing.assert_array_equal(masks[0] & masks[1], masks[3])
            np.testing.assert_array_equal(np.any(masks, 2), c['common_images'])
            assert np.all(~c['common_images'] | c['pooled'])
            # Independent membership check on a threshold grid. Every cell
            # of the finite two-coordinate arrangement has a grid corner.
            pair = c['pair']
            source = codes[c['sources']][:, :, pair]
            xs = np.unique(source[:, :, 0])
            ys = np.unique(source[:, :, 1])
            probes = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
            actual = np.zeros(len(probes), dtype=bool)
            for generator in c['generators']:
                actual |= (probes <= generator).all(1)
            wanted = np.ones(len(probes), dtype=bool)
            for image in source:
                wanted &= np.any((probes[:, None] <= image).all(2), 1)
            np.testing.assert_array_equal(actual, wanted)
            extent = np.ones(len(codes), dtype=bool)
            for generator in c['generators']:
                extent &= (codes[:, :, pair] >= generator).all(2).any(1)
            np.testing.assert_array_equal(extent, c['downset_extent'])
            for image in codes[extent][:, :, pair]:
                assert np.all(~wanted | np.any(
                    (probes[:, None] <= image).all(2), 1))
    path = root / 'results/controls_codexgen.json'
    controls = json.loads(path.read_text())
    for control in controls['identical_patch_controls']:
        i = control['class_id']
        with np.load(root / f'controls/context_{i:02d}_codexgen.npz') as c:
            grid = int(np.sqrt(codes.shape[1]))
            pixel = 224 // grid
            patches = []
            for image, site in zip(c['images'], c['sites']):
                y, x = divmod(int(site), grid)
                patches.append(image[y * pixel:(y + 1) * pixel,
                                     x * pixel:(x + 1) * pixel])
            for patch in patches[1:]:
                np.testing.assert_array_equal(patch, patches[0])
            positive = c['query'] > 0
            points = c['codes'][np.arange(4), c['sites']]
            ratios = (points[:, positive] / c['query'][positive]).min(1)
            np.testing.assert_allclose(ratios, c['ratios'], rtol=1e-6)
            np.testing.assert_allclose(ratios, control['ratios'], rtol=1e-6)
    result = {'model': name, 'cached_images': len(codes),
              'cached_sites': count, 'vector_queries': 40,
              'downset_concepts': 10, 'identical_pixel_cases': 3,
              'native': None}
    if native:
        result['native'] = check_native(name, sample, records, features)
    if native:
        (root / 'results/verification_codexgen.json').write_text(
            json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def check_native(
    name: str,
    sample: dict[str, np.ndarray],
    records: list[dict[str, np.ndarray]],
    features: Sequence[int],
) -> dict[str, Any]:
    """Recompute upstream activations and compare them with cached values."""
    torch.set_num_threads(4)
    adapter = Adapter(name)
    # Recompute an entire extraction batch, avoiding batch-size drift.
    images = sample['images'][:10]
    with torch.no_grad():
        dense = adapter.dense(images)
        np.testing.assert_allclose(dense.numpy(), records[0]['dense'],
                                   rtol=1e-5, atol=2e-5)
        values = adapter.normalize(dense)
        encoded = adapter.encode(values)
        if name == 'saev':
            manual = torch.relu((values - adapter.sae.b_dec) @
                                adapter.sae.W_enc + adapter.sae.b_enc)
            torch.testing.assert_close(encoded, manual)
        codes = encoded.numpy()
    record = records[0]
    matrix = sparse.csr_matrix(
        (record['data'], record['indices'], record['indptr']),
        shape=record['shape'])
    np.testing.assert_allclose(codes.reshape(matrix.shape), matrix.toarray(),
                               rtol=1e-5, atol=2e-5)
    # Check hook/layer conventions against an independent access route.
    x = torch.tensor(images[:2]).permute(0, 3, 1, 2).float() / 255
    x = tf.normalize(x, adapter.mean, adapter.std)
    captured = []
    with torch.no_grad():
        if name == 'prisma':
            import open_clip
            from safetensors.torch import load_file
            original = open_clip.create_model('ViT-B-32', pretrained=None)
            path = ROOT / 'checkpoints/prisma_backbone'
            original.load_state_dict(load_file(str(
                path / 'open_clip_model.safetensors')))
            original.eval()
            handle = original.visual.transformer.resblocks[11]\
                .register_forward_hook(lambda module, args, out:
                                       captured.append(out.detach()))
            original.encode_image(x)
            handle.remove()
            reference = captured[0][:, 1:]
        else:
            handle = adapter.model.encoder.layer[10].register_forward_hook(
                lambda module, args, out: captured.append(out.detach()))
            adapter.model(x)
            handle.remove()
            reference = captured[0][:, 5:]
        current = adapter.dense(images[:2])
    np.testing.assert_allclose(current.numpy(), reference.numpy(),
                               rtol=1e-4, atol=1e-4)
    for target in [1, 4, 6]:
        path = ROOT / name / f'controls/context_{target:02d}_codexgen.npz'
        with np.load(path) as c, torch.no_grad():
            dense_control = adapter.dense(c['images'])
            new = adapter.encode(adapter.normalize(dense_control))
            np.testing.assert_allclose(new[:, :, features].numpy(),
                                       c['codes'], rtol=1e-4, atol=1e-4)
    return {'recomputed_images': 10, 'full_dictionary_reencoded': True,
            'recomputed_control_images': 12,
            'hook_max_absolute_error': float(
                (current - reference).abs().max()),
            'reference': 'OpenCLIP residual block 11' if name == 'prisma'
            else 'HF register backbone layer 10 forward hook'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=['prisma', 'saev'])
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    verify(args.name, args.native)
