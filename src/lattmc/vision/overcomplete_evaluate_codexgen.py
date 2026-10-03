"""Evaluate reconstruction and label-free queries on arbitrary patch sites."""

from __future__ import annotations
from typing import Any

import argparse
import json

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from lattmc.vision.overcomplete_data_codexgen import load
from lattmc.vision.overcomplete_extract_codexgen import load_dense
from lattmc.vision.overcomplete_fetch_codexgen import ROOT
from lattmc.vision.overcomplete_train_codexgen import restore


def encode(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    name: str,
    dataset: str,
) -> None:
    """Cache sparse codes and reconstruction errors for a dataset."""
    folder = ROOT / 'codes' / name
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{dataset}_codexgen.npz'
    if target.exists():
        return
    values = load_dense(dataset)
    flat = torch.from_numpy(values.reshape(-1, 768))
    flat = (flat - checkpoint['mean']) / checkpoint['scale']
    blocks, errors, totals = [], [], []
    with torch.no_grad():
        for x in flat.split(256):
            _, z, output = model(x)
            assert torch.isfinite(z).all() and (z >= 0).all()
            blocks.append(sparse.csr_matrix(z.numpy()))
            errors.extend((output - x).square().sum(1).tolist())
            totals.extend(x.square().sum(1).tolist())
    matrix = sparse.vstack(blocks, format='csr')
    np.savez_compressed(target, data=matrix.data, indices=matrix.indices,
                        indptr=matrix.indptr, shape=matrix.shape,
                        error=np.array(errors).reshape(-1, 256).sum(1),
                        baseline=np.array(totals).reshape(-1, 256).sum(1))
    print(name, dataset, 'encoded', matrix.shape, flush=True)


def read_codes(
    name: str,
    dataset: str,
) -> tuple[sparse.csr_matrix, dict[str, np.ndarray]]:
    """Load sparse activation codes and reconstruction statistics."""
    path = ROOT / f'codes/{name}/{dataset}_codexgen.npz'
    with np.load(path) as data:
        matrix = sparse.csr_matrix((data['data'], data['indices'],
                                    data['indptr']), shape=data['shape'])
        stats = {k: data[k] for k in ['error', 'baseline']}
    return matrix, stats


def pool(matrix: sparse.csr_matrix, sites: int = 256) -> np.ndarray:
    """Max-pool contiguous site groups into dense image codes."""
    return np.stack([matrix[i:i + sites].max(0).toarray()[0]
                     for i in range(0, matrix.shape[0], sites)])


def queries(name: str) -> dict[str, Any]:
    """Select label-free feature pairs and freeze query definitions."""
    matrix, _ = read_codes(name, 'imagenette')
    _, records = load('imagenette')
    train = np.array([r['split'] == 'train' for r in records])
    sites = matrix.shape[0] // len(records)
    pooled = pool(matrix, sites)
    p = pooled[train]
    frequency = (p > 0).mean(0)
    variance = p.var(0)
    eligible = (frequency >= 0.05) & (frequency <= 0.95)
    if eligible.sum() < 4:
        # Dense image supports still have informative activation magnitudes.
        eligible = frequency >= 0.05
    candidates = np.flatnonzero(eligible)
    selected = candidates[np.argsort(-variance[candidates],
                                     kind='stable')[:24]]
    assert len(selected) >= 4
    codes = matrix[:, selected].toarray().reshape(len(records), sites, -1)
    training = codes[train].reshape(-1, len(selected))
    corr = np.corrcoef(training.T)
    pairs = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            common = ((training[:, i] > 0) & (training[:, j] > 0)).sum()
            if common >= 10:
                pairs.append((float(corr[i, j]), i, j))
    pairs.sort(reverse=True)
    chosen, used = [], set()
    for correlation, i, j in pairs:
        if i in used or j in used:
            continue
        used.update([i, j])
        values = training[:, [i, j]]
        quantiles = [np.quantile(values[values[:, k] > 0, k], [0.5, 0.8])
                     for k in range(2)]
        u = [quantiles[0][0], quantiles[1][1]]
        v = [quantiles[0][1], quantiles[1][0]]
        chosen.append({'features': [int(selected[i]), int(selected[j])],
                       'u': u, 'v': v, 'correlation': correlation})
        if len(chosen) == 4:
            break
    result = {'selected_features': selected.tolist(), 'queries': chosen,
              'selection': 'training variance, frequency, coactivation; '
              'no class or position constraints'}
    folder = ROOT / 'results' / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'queries_codexgen.json').write_text(
        json.dumps(result, indent=2) + '\n')
    return result


def evaluate(
    name: str,
    dataset: str,
    definition: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate fixed spatial queries and reconstruction statistics."""
    matrix, stats = read_codes(name, dataset)
    _, records = load(dataset)
    count = len(records)
    sites = matrix.shape[0] // count
    pooled = pool(matrix, sites)
    nnz = np.diff(matrix.indptr).reshape(count, sites).mean(1)
    metrics = {}
    for split in sorted({r['split'] for r in records}):
        ids = np.array([r['split'] == split for r in records])
        metrics[split] = {'images': int(ids.sum()),
                          'r2': float(1 - stats['error'][ids].sum()
                                      / stats['baseline'][ids].sum()),
                          'l0': float(nnz[ids].mean())}
    outcomes, closure_checks = [], 0
    test = np.array([r['split'] == 'test' for r in records])
    for item in definition['queries']:
        features = item['features']
        z = matrix[:, features].toarray().reshape(count, sites, 2)
        u, v = np.array(item['u']), np.array(item['v'])
        operations = {'u': u, 'v': v, 'meet': np.minimum(u, v),
                      'join': np.maximum(u, v)}
        result = {'features': features, 'operations': {}}
        masks = {}
        for operation, query in operations.items():
            patch = (z >= query).all(2)
            common = patch.any(1)
            image = (pooled[:, features] >= query).all(1)
            assert not np.any(common & ~image)
            extent = z[patch]
            top = z.max((0, 1))
            closed = extent.min(0) if len(extent) else np.maximum(top, query)
            # Queries above a dataset bound are evaluated in an enlarged box.
            # Its top alone could coincide with a code only if extent nonempty.
            assert np.array_equal((z >= closed).all(2), patch)
            closure_checks += 1
            masks[operation] = patch
            classes = sorted({records[i]['label']
                              for i in np.flatnonzero(common & test)})
            result['operations'][operation] = {
                'query': query.tolist(), 'closed': closed.tolist(),
                'pooled_test': int((image & test).sum()),
                'common_test': int((common & test).sum()),
                'test_labels': classes, 'test_label_count': len(classes),
                'test_sites': int(patch[test].sum())}
        assert np.array_equal(masks['join'], masks['u'] & masks['v'])
        assert np.all((masks['u'] | masks['v']) <= masks['meet'])
        outcomes.append(result)
    result = {'model': name, 'dataset': dataset, 'metrics': metrics,
              'queries': outcomes, 'closure_checks': closure_checks}
    target = ROOT / f'results/{name}/{dataset}_codexgen.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(name, dataset, metrics, flush=True)
    return result


def stability() -> None:
    """Compare feature alignment and query stability across trained runs."""
    results = []
    for family in ['topk', 'batchtopk', 'jump', 'relu', 'archetypal',
                   'relu_fixed']:
        for budget in [16, 32]:
            name = f'{family}_k{budget}_s0'
            reference, _ = restore(name)
            d0 = reference.get_dictionary().detach().numpy()
            d0 /= np.linalg.norm(d0, axis=1, keepdims=True).clip(1e-12)
            m0, _ = read_codes(name, 'imagenette')
            p0 = pool(m0)
            for seed in [1, 2]:
                other = f'{family}_k{budget}_s{seed}'
                model, _ = restore(other)
                d1 = model.get_dictionary().detach().numpy()
                d1 /= np.linalg.norm(d1, axis=1, keepdims=True).clip(1e-12)
                similarity = d0 @ d1.T
                a, b = linear_sum_assignment(-similarity)
                m1, _ = read_codes(other, 'imagenette')
                p1 = pool(m1)[:, b]
                # Compare image extents above each feature's training median.
                _, records = load('imagenette')
                tr = np.array([r['split'] == 'train' for r in records])
                te = np.array([r['split'] == 'test' for r in records])
                eligible = (p0[tr] > 0).sum(0) >= 10
                overlap = []
                for j in np.flatnonzero(eligible):
                    t0 = np.median(p0[tr, j][p0[tr, j] > 0])
                    positive = p1[tr, j][p1[tr, j] > 0]
                    if not len(positive):
                        continue
                    t1 = np.median(positive)
                    e0, e1 = p0[te, j] >= t0, p1[te, j] >= t1
                    union = (e0 | e1).sum()
                    if union:
                        overlap.append(float((e0 & e1).sum() / union))
                results.append({'family': family, 'budget': budget,
                                'seed_pair': [0, seed],
                                'decoder_cosine': float(
                                    similarity[a, b].mean()),
                                'extent_jaccard': float(np.mean(overlap)),
                                'nonempty_comparisons': len(overlap)})
    (ROOT / 'results/stability_codexgen.json').write_text(
        json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='imagenette')
    parser.add_argument('--name')
    parser.add_argument('--stability', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stability:
        stability()
    else:
        names = ([args.name] if args.name else
                 [p.parent.name for p in sorted(
                     (ROOT / 'checkpoints').glob('*/model_codexgen.pt'))])
        for name in names:
            model, checkpoint = restore(name)
            encode(model, checkpoint, name, args.dataset)
            definition = queries(name)
            evaluate(name, args.dataset, definition)
