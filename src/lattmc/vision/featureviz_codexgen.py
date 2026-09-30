"""Measured exemplars, optimized stimuli, and controlled response probes."""

import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score
from torch.nn import functional as functional
from torchvision.transforms import functional as tf
from torchvision.transforms import InterpolationMode

from lattmc.vision.backbones_codexgen import (
    Backbone, dataset, load_codes, load_surrogate, pixels)
from lattmc.vision.contexts_codexgen import spatial_extents
from lattmc.vision.paths_codexgen import experiment_root


def choose_features(sample, codes):
    pooled = codes.max(1)
    train = np.flatnonzero(sample['splits'] == 'train')
    test = np.flatnonzero(sample['splits'] == 'test')
    records, chosen = [], []
    for target in [1, 4]:
        labels = sample['labels'][train]
        values = pooled[train]
        contrast = (values[labels == target].mean(0)
                    - values[labels != target].mean(0))
        contrast /= values.std(0).clip(1e-8)
        contrast[chosen] = -np.inf
        feature = int(np.argmax(contrast))
        chosen.append(feature)
        candidates = train[labels == target]
        source = int(candidates[np.argmax(pooled[candidates, feature])])
        ranked = test[np.argsort(-pooled[test, feature], kind='stable')[:3]]
        records.append({'feature': feature, 'class_id': target,
                        'class': str(sample['classes'][target]),
                        'source_row': source, 'test_rows': ranked.tolist(),
                        'test_ap': float(average_precision_score(
                            sample['labels'][test] == target,
                            pooled[test, feature]))})
    return records


def codes_for(backbone, sae, center, scale, values):
    return sae.encode((backbone(values) - center) / scale)


def optimize(backbone, sae, center, scale, features, training_scale):
    """Two fixed initializations per feature; keep every resulting image."""
    torch.manual_seed(2028)
    targets = torch.tensor(np.repeat(features, 2))
    count, size = len(targets), 224
    parameter = torch.randn(count, 3, size, size // 2 + 1, 2) * 0.2
    parameter.requires_grad_(True)
    fy = torch.fft.fftfreq(size)[:, None]
    fx = torch.fft.rfftfreq(size)[None, :]
    frequency = (fx.square() + fy.square()).sqrt().clamp_min(1 / size)
    spectrum = frequency.pow(-1.5)
    spectrum /= spectrum.square().mean().sqrt()
    optimizer = torch.optim.Adam([parameter], lr=0.05)
    divisor = torch.tensor(training_scale)[targets].clamp_min(1e-6)
    trace = []

    def render():
        complex_values = torch.view_as_complex(parameter)
        return torch.fft.irfft2(complex_values * spectrum,
                                s=(size, size), norm='ortho').sigmoid()

    initial = render().detach()
    for step in range(120):
        image = render()
        offset = torch.randint(0, 17, (2,))
        shifted = functional.pad(image, (8, 8, 8, 8), mode='reflect')
        shifted = shifted[:, :, offset[0]:offset[0] + size,
                          offset[1]:offset[1] + size]
        hidden = (backbone(shifted) - center) / scale
        pre = sae.encoder(hidden)
        chosen = pre.gather(-1, targets[:, None, None].expand(
            -1, pre.shape[1], 1)).squeeze(-1)
        # Smooth max of pre-gating responses avoids a zero Top-k gradient.
        response = torch.logsumexp(chosen, dim=1) - np.log(chosen.shape[1])
        tv = ((image[:, :, 1:] - image[:, :, :-1]).square().mean()
              + (image[:, :, :, 1:] - image[:, :, :, :-1]).square().mean())
        loss = -(response / divisor).mean() + 0.1 * tv
        assert torch.isfinite(loss), "Nonfinite visualization objective"
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach()))
        if step % 30 == 0:
            print(backbone.name, 'visualization step', step,
                  trace[-1], flush=True)
    optimized = render().detach()
    with torch.no_grad():
        before = codes_for(backbone, sae, center, scale, initial).amax(1)
        after = codes_for(backbone, sae, center, scale, optimized).amax(1)
        rows = torch.arange(count)
        before, after = before[rows, targets], after[rows, targets]
    return {'initial': initial.numpy(), 'optimized': optimized.numpy(),
            'features': targets.numpy(), 'initial_codes': before.numpy(),
            'optimized_codes': after.numpy(), 'loss': np.array(trace)}


def synthetic_stimuli():
    """Mean/contrast-matched curves, straight lines, and right angles."""
    size, supersampling = 224, 4
    patterns, kinds, angles = [], [], []
    for kind in ['curve', 'line', 'corner']:
        image = Image.new('L', (size * supersampling,) * 2, 0)
        draw = ImageDraw.Draw(image)
        bounds = tuple(v * supersampling for v in [48, 48, 176, 176])
        if kind == 'curve':
            draw.arc(bounds, 0, 180, fill=255, width=6 * supersampling)
        elif kind == 'line':
            draw.line((48 * 4, 112 * 4, 176 * 4, 112 * 4),
                      fill=255, width=6 * 4)
        else:
            draw.line([(48 * 4, 112 * 4), (112 * 4, 112 * 4),
                       (112 * 4, 176 * 4)], fill=255, width=6 * 4)
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        for angle in range(0, 360, 30):
            rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC)
            array = np.array(rotated).astype(np.float32) / 255
            array = (array - array.mean()) / max(array.std(), 1e-8)
            patterns.append(array)
            kinds.append(kind)
            angles.append(angle)
    patterns = np.stack(patterns)
    factor = 0.45 / np.abs(patterns).max()
    patterns = 0.5 + factor * patterns
    patterns = np.repeat(patterns[:, None], 3, axis=1)
    return patterns, np.array(kinds), np.array(angles)


def probe(backbone, sae, center, scale, sample, records):
    synthetic, kinds, angles = synthetic_stimuli()
    features = [r['feature'] for r in records]
    scores = []
    with torch.no_grad():
        for batch in torch.tensor(synthetic).split(6):
            z = codes_for(backbone, sae, center, scale, batch)
            scores.append(z.amax(1)[:, features].numpy())
    rotation_images, rotation_scores = [], []
    for record in records:
        base = pixels(sample['images'][[record['source_row']]])[0]
        rotated = torch.stack([tf.rotate(
            base, angle, interpolation=InterpolationMode.BILINEAR,
            fill=[0.5, 0.5, 0.5]) for angle in range(0, 360, 30)])
        rotation_images.append(rotated.numpy())
        with torch.no_grad():
            z = torch.cat([codes_for(backbone, sae, center, scale, batch)
                           for batch in rotated.split(6)])
        rotation_scores.append(z.amax(1)[:, record['feature']].numpy())
    return {'synthetic': synthetic, 'kinds': kinds, 'angles': angles,
            'synthetic_scores': np.concatenate(scores),
            'rotation_images': np.stack(rotation_images),
            'rotation_scores': np.stack(rotation_scores)}


def queries(sample, cache, records, folder):
    pooled = cache['codes'].max(1)
    features = [r['feature'] for r in records]
    sources = [r['source_row'] for r in records]
    base = np.zeros((2, pooled.shape[1]), dtype=np.float32)
    base[:, features] = 0.5 * pooled[sources][:, features]
    vectors = np.stack([base[0], base[1], np.minimum(*base),
                        np.maximum(*base)])
    rows = np.flatnonzero(np.isin(sample['splits'], ['test', 'transfer']))
    masks = [spatial_extents(cache['codes'][rows], q) for q in vectors]
    np.savez_compressed(folder / 'retrieval/queries_codexgen.npz',
                        sources=sources, features=features, vectors=vectors,
                        rows=rows, pooled=np.stack([m[0] for m in masks]),
                        same_site=np.stack([m[1] for m in masks]))
    return [{'operation': op, 'pooled': int(m[0].sum()),
             'same_site': int(m[1].sum())}
            for op, m in zip(['u', 'v', 'meet', 'join'], masks)]


def run(name):
    torch.set_num_threads(4)
    torch.manual_seed(2028)
    sample, cache = dataset(), load_codes(name)
    folder = experiment_root('imagenette_' + name)
    records = choose_features(sample, cache['codes'])
    backbone = Backbone(name)
    sae, center, scale = load_surrogate(name)
    sae.requires_grad_(False)
    train = np.flatnonzero(sample['splits'] == 'train')
    rms = np.sqrt((cache['codes'][train].max(1) ** 2).mean(0))
    visual = optimize(backbone, sae, center, scale,
                      [r['feature'] for r in records], rms)
    probes = probe(backbone, sae, center, scale, sample, records)
    destination = folder / 'visualization'
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination / 'optimized_codexgen.npz', **visual)
    np.savez_compressed(destination / 'probes_codexgen.npz', **probes)
    result = {'features': records, 'optimization_steps': 120,
              'initializations_per_feature': 2, 'seed': 2028,
              'objective': 'log-mean-exp pre-Top-k response / training RMS',
              'initial_actual_codes': visual['initial_codes'].tolist(),
              'optimized_actual_codes': visual['optimized_codes'].tolist(),
              'rotation_degrees': list(range(0, 360, 30)),
              'lattice_queries': queries(sample, cache, records, folder)}
    (folder / 'results/featureviz_codexgen.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(name, result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['resnet34', 'dinov2'])
    run(parser.parse_args().model)
