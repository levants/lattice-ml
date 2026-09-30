"""Patch correspondence and identical-pixel context controls, not masks."""

import argparse
import json

import numpy as np
import torch

from lattmc.vision.contexts_codexgen import graded_score
from lattmc.vision.patch_contexts_codexgen import read
from lattmc.vision.patch_models_codexgen import Adapter, ROOT


def overlap(first, second):
    union = (first | second).sum(1)
    valid = union > 0
    values = (first & second).sum(1)[valid] / union[valid]
    return float(values.mean()) if len(values) else None, int(valid.sum())


def run(name):
    torch.set_num_threads(4)
    sample, codes, features, _ = read(name)
    root = ROOT / name
    (root / 'controls').mkdir(exist_ok=True)
    model = Adapter(name)
    test = np.flatnonzero(sample['splits'] == 'test')
    flipped = []
    with torch.no_grad():
        for batch in np.array_split(test, 20):
            images = sample['images'][batch, :, ::-1].copy()
            dense = model.dense(images)
            flipped.append(model.encode(model.normalize(dense))[
                :, :, features].numpy())
            print(name, 'flipped', int(batch[-1]), flush=True)
    flipped = np.concatenate(flipped)
    aligned = flipped.reshape(len(test), model.grid, model.grid, -1)
    aligned = aligned[:, :, ::-1].reshape(flipped.shape)
    np.savez_compressed(root / 'controls/flip_codexgen.npz',
                        codes=flipped, rows=test, features=features)
    overlaps = []
    for i in range(10):
        with np.load(root / f'contexts/class_{i:02d}_codexgen.npz') as c:
            q = c['queries'][3]
        original = (codes[test] >= q).all(2)
        mirrored = (aligned >= q).all(2)
        observed, valid = overlap(original, mirrored)
        rng = np.random.default_rng(2031 + i)
        null = []
        for repeat in range(100):
            permuted = np.stack([rng.permutation(row) for row in mirrored])
            value, _ = overlap(original, permuted)
            if value is not None:
                null.append(value)
        overlaps.append({'class_id': i, 'mean_jaccard': observed,
                         'nonempty_union_images': valid,
                         'shuffled_mean': float(np.mean(null)) if null
                         else None,
                         'shuffled_quantiles': np.quantile(
                             null, [.025, .975]).tolist() if null else None})
    controls = []
    for target in [1, 4, 6]:
        with np.load(root / f'contexts/class_{target:02d}_codexgen.npz') as c:
            q, pair = c['queries'][3], c['pair']
            strength = c['scores'][3, test].max(1)
        position = int(test[np.argmax(strength)])
        score = graded_score(codes[position], q)
        site = int(np.argmax(score))
        y, x = divmod(site, model.grid)
        pixel = model.patch
        image = sample['images'][position]
        full = image.copy()
        window = np.full_like(image, 128)
        y0, y1 = max(0, y - 1) * pixel, min(model.grid, y + 2) * pixel
        x0, x1 = max(0, x - 1) * pixel, min(model.grid, x + 2) * pixel
        window[y0:y1, x0:x1] = image[y0:y1, x0:x1]
        crop = image[y * pixel:(y + 1) * pixel,
                     x * pixel:(x + 1) * pixel].copy()
        isolated = np.full_like(image, 128)
        isolated[y * pixel:(y + 1) * pixel,
                 x * pixel:(x + 1) * pixel] = crop
        # A cyclic shift by half the grid changes token position only.
        ny, nx = (y + model.grid // 2) % model.grid, x
        moved = np.full_like(image, 128)
        moved[ny * pixel:(ny + 1) * pixel,
              nx * pixel:(nx + 1) * pixel] = crop
        images = np.stack([full, window, isolated, moved])
        sites = np.array([site, site, site, ny * model.grid + nx])
        for version, at in zip(images, sites):
            yy, xx = divmod(int(at), model.grid)
            np.testing.assert_array_equal(
                version[yy * pixel:(yy + 1) * pixel,
                        xx * pixel:(xx + 1) * pixel], crop)
        with torch.no_grad():
            z = model.encode(model.normalize(model.dense(images)))
        selected = z[:, :, features].numpy()
        point = selected[np.arange(4), sites]
        ratios = graded_score(point, q)
        np.savez_compressed(
            root / f'controls/context_{target:02d}_codexgen.npz',
            images=images, codes=selected, sites=sites, query=q,
            source_position=position, features=features,
            pair=pair, ratios=ratios)
        controls.append({'class_id': target,
                         'source_row': int(sample['rows'][position]),
                         'site': site,
                         'conditions': ['original', '3x3 window',
                                        'isolated patch', 'relocated patch'],
                         'ratios': ratios.tolist(),
                         'pair_codes': point[:, pair].tolist()})
    # Record the effect of the current notebook's different lower clip.
    sensitivity = None
    if name == 'saev':
        with torch.no_grad():
            balanced = np.array([np.flatnonzero(
                (sample['labels'] == i) &
                (sample['splits'] == 'train'))[0] for i in range(10)])
            dense = model.dense(sample['images'][balanced])
            canonical = model.normalize(dense)
            clipped = model.normalize(dense, reference_clip=True)
            rows = []
            for x in [canonical, clipped]:
                z = model.encode(x)
                reconstruction = model.decode(z)
                mse = float((reconstruction - x).square().mean())
                rows.append({'mse': mse,
                             'mean_l0': float((z > 0).sum(-1).float().mean())})
            uncentered = model.sae.encode(canonical.flatten(0, 1)).f_x
            uncentered_error = (model.decode(uncentered) -
                                canonical.flatten(0, 1)).square().mean()
            sensitivity = {'current_encoder_mse': float(uncentered_error),
                           'current_encoder_l0': float(
                               (uncentered > 0).sum(-1).float().mean()),
                           'images': 10,
                           'training_rows': sample['rows'][balanced]
                           .tolist(), 'training_normalization': rows[0],
                           'notebook_normalization': rows[1],
                           'input_rmse': float((canonical - clipped)
                                               .square().mean().sqrt())}
    result = {'flip': overlaps, 'identical_patch_controls': controls,
              'normalization_sensitivity': sensitivity}
    (root / 'results/controls_codexgen.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=['prisma', 'saev'])
    run(parser.parse_args().name)
