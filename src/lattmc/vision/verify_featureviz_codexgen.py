"""Independently recompute image provenance and CNN/ViT feature evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tf

from lattmc.vision.backbones_codexgen import (
    Backbone, dataset, load_codes, load_surrogate, pixels)
from lattmc.vision.contexts_codexgen import spatial_extents
from lattmc.vision.datasets_codexgen import sha256
from lattmc.vision.featureviz_codexgen import choose_features, codes_for
from lattmc.vision.paths_codexgen import experiment_root


def verify(inference: bool = True) -> dict[str, int | bool]:
    """Verify feature-visualization caches and optionally recompute inference.
    """
    torch.set_num_threads(4)
    root = experiment_root('imagenette_imagewoof')
    for name in ['imagenette', 'imagewoof']:
        metadata = json.loads((root / f'results/{name}_codexgen.json')
                              .read_text())
        for filename, value in metadata['artifact_sha256'].items():
            assert sha256(root / 'dataset' / filename) == value
        with np.load(root / f'dataset/{name}_codexgen.npz') as data:
            assert len(np.unique(data['source_ids'])) == len(data['images'])
            with zipfile.ZipFile(root / 'dataset'
                                 / f'{name}_originals_codexgen.zip') as z:
                for row, record in enumerate(metadata['records']):
                    raw = z.read(record['archive_member'])
                    assert hashlib.sha256(raw).hexdigest() == (
                        record['original_sha256'])
                    image = Image.open(io.BytesIO(raw)).convert('RGB')
                    image = tf.resize(image, 256, InterpolationMode.BICUBIC)
                    image = tf.center_crop(image, [224, 224])
                    np.testing.assert_array_equal(np.array(image),
                                                  data['images'][row])
    sample = dataset()
    assert len(np.unique(sample['source_ids'])) == 450
    for name in ['resnet34', 'dinov2']:
        root = experiment_root('imagenette_' + name)
        report = json.loads((root / 'results/experiment_codexgen.json')
                            .read_text())
        for filename, value in report['artifact_sha256'].items():
            assert sha256(root / filename) == value
        record = json.loads((root / 'results/featureviz_codexgen.json')
                            .read_text())
        if name == 'dinov2':
            provenance = json.loads((root / 'checkpoints'
                / 'backbone_provenance_codexgen.json').read_text())
            for filename, expected in provenance['files'].items():
                actual = sha256(root / 'checkpoints/model' / filename)
                assert actual == expected
        cache = load_codes(name)
        np.testing.assert_array_equal(cache['rows'], np.arange(450))
        assert choose_features(sample, cache['codes']) == record['features']
        with np.load(root / 'retrieval/queries_codexgen.npz') as q:
            base = np.zeros_like(q['vectors'][:2])
            base[:, q['features']] = 0.5 * cache['codes'].max(1)[
                q['sources']][:, q['features']]
            np.testing.assert_array_equal(base, q['vectors'][:2])
            np.testing.assert_array_equal(np.minimum(*base), q['vectors'][2])
            np.testing.assert_array_equal(np.maximum(*base), q['vectors'][3])
            for i, query in enumerate(q['vectors']):
                masks = spatial_extents(cache['codes'][q['rows']], query)
                np.testing.assert_array_equal(masks[0], q['pooled'][i])
                np.testing.assert_array_equal(masks[1], q['same_site'][i])
            np.testing.assert_array_equal(q['pooled'][3],
                                          q['pooled'][0] & q['pooled'][1])
        sae, center, scale = load_surrogate(name)
        dense = torch.from_numpy(cache['dense'])
        with torch.no_grad():
            codes = torch.cat([sae.encode((x - center) / scale)
                               for x in dense.split(25)])
            recovered = sae.decoder(codes) * scale + center
        np.testing.assert_array_equal(codes.numpy(), cache['codes'])
        for split, metric in report['metrics'].items():
            ids = np.flatnonzero(sample['splits'] == split)
            r2 = 1 - ((dense[ids] - recovered[ids]).square().sum()
                      / (dense[ids] - center).square().sum())
            np.testing.assert_allclose(float(r2), metric['r2'], atol=1e-7)
        if not inference:
            continue
        backbone = Backbone(name)
        with torch.no_grad():
            fresh = torch.cat([backbone(pixels(batch)) for batch in
                               np.array_split(sample['images'], 45)])
            np.testing.assert_array_equal(fresh.numpy(), cache['dense'])
            with np.load(root / 'visualization/optimized_codexgen.npz') as v:
                for key, score_key in [('initial', 'initial_codes'),
                                       ('optimized', 'optimized_codes')]:
                    z = codes_for(backbone, sae, center, scale,
                                  torch.from_numpy(v[key])).amax(1)
                    measured = z[np.arange(4), v['features']].numpy()
                    np.testing.assert_array_equal(measured, v[score_key])
            with np.load(root / 'visualization/probes_codexgen.npz') as v:
                scores = torch.cat([codes_for(backbone, sae, center, scale,
                                              batch).amax(1) for batch in
                                    torch.from_numpy(v['synthetic']).split(6)])
                features = [r['feature'] for r in record['features']]
                np.testing.assert_array_equal(scores[:, features].numpy(),
                                              v['synthetic_scores'])
                for column, feature in enumerate(features):
                    rotated = torch.from_numpy(v['rotation_images'][column])
                    z = torch.cat([codes_for(backbone, sae, center, scale,
                                            batch).amax(1)
                                   for batch in rotated.split(6)])
                    np.testing.assert_array_equal(
                        z[:, feature].numpy(), v['rotation_scores'][column])
                mean = v['synthetic'].mean((1, 2, 3))
                std = v['synthetic'].std((1, 2, 3))
                np.testing.assert_allclose(mean, 0.5, atol=1e-6)
                np.testing.assert_allclose(std, std[0], atol=1e-6)
        print('Verified', name, 'including full inference', flush=True)
    return {'images': 450, 'backbones': 2, 'optimized_stimuli': 8,
            'synthetic_probes_per_model': 36, 'full_inference': inference}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached-only', action='store_true')
    print(verify(inference=not parser.parse_args().cached_only))
