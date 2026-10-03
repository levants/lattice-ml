"""Cache actual upstream SAE codes as lossless sparse patch matrices."""

from __future__ import annotations
from typing import Any

import argparse
import json
import time

import numpy as np
import torch
from scipy import sparse

from lattmc.vision.patch_models_codexgen import Adapter, ROOT, dataset


def extract(name: str) -> None:
    """Cache upstream patch codes and reconstruction diagnostics."""
    torch.set_num_threads(4)
    sample = dataset(name)
    out = ROOT / name
    (out / 'activations').mkdir(parents=True, exist_ok=True)
    (out / 'dataset').mkdir(exist_ok=True)
    (out / 'results').mkdir(exist_ok=True)
    np.savez_compressed(out / 'dataset/images_codexgen.npz', **sample)
    model = Adapter(name)
    statistics = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(sample['rows']), 10):
            end = min(start + 10, len(sample['rows']))
            target = out / f'activations/chunk_{start:03d}_codexgen.npz'
            if target.exists():
                continue
            dense = model.dense(sample['images'][start:end])
            values = model.normalize(dense)
            codes = model.encode(values)
            reconstruction = model.decode(codes)
            matrix = sparse.csr_matrix(codes.flatten(0, 1).numpy())
            errors = (reconstruction - values).square().sum((1, 2)).numpy()
            norms = values.square().sum((1, 2)).numpy()
            active = (codes > 0).sum(2).float().mean(1).numpy()
            np.savez_compressed(
                target, dense=dense.numpy(), shape=np.array(matrix.shape),
                data=matrix.data, indices=matrix.indices,
                indptr=matrix.indptr, pooled=codes.amax(1).numpy(),
                input_sum=values.sum(1).numpy(), input_sumsq=norms,
                squared_error=errors, mean_l0=active,
                positions=np.arange(start, end))
            statistics.append([float(errors.sum()), float(active.mean())])
            print(name, end, 'images;', round(time.monotonic() - started, 1),
                  'seconds', flush=True)
    print(name, 'extraction complete', flush=True)


def load(
    name: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    """Load cached images and ordered activation chunks."""
    directory = ROOT / name
    with np.load(directory / 'dataset/images_codexgen.npz') as data:
        sample = {k: data[k] for k in data.files}
    records = []
    paths = (directory / 'activations').glob('chunk_*_codexgen.npz')
    for path in sorted(paths):
        with np.load(path) as data:
            records.append({k: data[k] for k in data.files})
    assert np.array_equal(np.concatenate([r['positions'] for r in records]),
                          np.arange(len(sample['rows'])))
    return sample, records


def select(name: str) -> dict[str, Any]:
    """Select distinct feature pairs using training-set class contrasts."""
    sample, records = load(name)
    pooled = np.concatenate([r['pooled'] for r in records])
    train = sample['splits'] == 'train'
    values = pooled[train].astype(np.float64)
    labels = sample['labels'][train]
    spread = values.std(0).clip(1e-12)
    features, cases = [], []
    for target in range(10):
        contrast = (values[labels == target].mean(0)
                    - values[labels != target].mean(0)) / spread
        ordered = np.argsort(-contrast, kind='stable')
        pair = [int(j) for j in ordered if int(j) not in features][:2]
        features.extend(pair)
        cases.append({'class_id': target,
                      'class': str(sample['classes'][target]),
                      'features': pair,
                      'contrasts': contrast[pair].tolist()})
    selected = []
    for r in records:
        matrix = sparse.csr_matrix((r['data'], r['indices'], r['indptr']),
                                   shape=r['shape'])
        selected.append(matrix[:, features].toarray().reshape(
            len(r['positions']), -1, len(features)))
    codes = np.concatenate(selected)
    np.savez_compressed(ROOT / name / 'activations/selected_codexgen.npz',
                        codes=codes, features=features)
    sites = codes.shape[1]
    input_sum = np.concatenate([r['input_sum'] for r in records])
    mean = input_sum[train].sum(0, dtype=np.float64) / (train.sum() * sites)
    sumsq = np.concatenate([r['input_sumsq'] for r in records])
    error = np.concatenate([r['squared_error'] for r in records])
    l0 = np.concatenate([r['mean_l0'] for r in records])
    metrics = {}
    for split in ['train', 'test']:
        ids = sample['splits'] == split
        baseline = (sumsq[ids].sum(dtype=np.float64)
                    - 2 * input_sum[ids].sum(0, dtype=np.float64) @ mean
                    + ids.sum() * sites * (mean @ mean))
        metrics[split] = {'r2': float(1 - error[ids].sum() / baseline),
                          'mean_l0': float(l0[ids].mean()),
                          'images': int(ids.sum()), 'sites': int(sites)}
    result = {'name': name, 'dictionary_size': int(pooled.shape[1]),
              'selected_features': features, 'cases': cases,
              'metrics': metrics}
    (ROOT / name / 'results/extraction_codexgen.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=['prisma', 'saev'])
    args = parser.parse_args()
    extract(args.name)
    select(args.name)
