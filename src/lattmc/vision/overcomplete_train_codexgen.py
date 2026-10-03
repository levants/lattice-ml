"""Controlled native Overcomplete SAE comparison on a frozen image split."""

from __future__ import annotations
from typing import Any

import argparse
import json
import time

import numpy as np
import torch

from lattmc.vision import overcomplete_compat_codexgen  # noqa: F401

from overcomplete.sae import BatchTopKSAE, JumpSAE, RATopKSAE, SAE, TopKSAE
from overcomplete.sae.jump_sae import heaviside

from lattmc.vision.overcomplete_data_codexgen import load
from lattmc.vision.overcomplete_extract_codexgen import load_dense
from lattmc.vision.overcomplete_fetch_codexgen import ROOT


WIDTH = 1536
FAMILIES = ['topk', 'batchtopk', 'jump', 'relu', 'archetypal', 'relu_fixed']


def training_data(

) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.floating, np.ndarray]:
    """Sample training sites and return normalized data and scale statistics.
    """
    _, records = load('imagenette')
    dense = load_dense('imagenette')
    train = np.array([r['split'] == 'train' for r in records])
    val = np.array([r['split'] == 'val' for r in records])
    # Each training image contributes equally, with no label conditioning.
    rng = np.random.default_rng(20260930)
    sites = np.stack([rng.choice(256, 64, replace=False)
                      for _ in range(train.sum())])
    images = dense[train]
    values = images[np.arange(len(images))[:, None], sites].reshape(-1, 768)
    mean = values.mean(0)
    scale = np.sqrt(np.mean((values - mean) ** 2))
    x = torch.from_numpy((values - mean) / scale)
    v = torch.from_numpy((dense[val, ::4].reshape(-1, 768) - mean) / scale)
    return x, v, mean, scale, sites


def make(
    family: str,
    budget: int,
    points: torch.Tensor,
    device: str = 'cpu',
) -> SAE:
    """Construct the requested sparse autoencoder family."""
    common = {'input_shape': 768, 'nb_concepts': WIDTH, 'device': device}
    if family == 'topk':
        return TopKSAE(**common, top_k=budget)
    if family == 'batchtopk':
        return BatchTopKSAE(**common, top_k=512 * budget)
    if family == 'jump':
        return JumpSAE(**common, bandwidth=0.1, kernel='rectangle')
    if family in ['relu', 'relu_fixed']:
        return SAE(**common)
    return RATopKSAE(**common, top_k=budget, points=points.to(device),
                     delta=0.2, use_multiplier=True)


def score(model: torch.nn.Module, values: torch.Tensor) -> dict[str, float]:
    """Measure reconstruction error, sparsity, and unused feature fraction."""
    errors, nonzero, used = [], [], torch.zeros(WIDTH, dtype=torch.bool)
    with torch.no_grad():
        for batch in values.split(512):
            _, z, out = model(batch)
            errors.append((out - batch).square().sum().item())
            nonzero.append((z > 0).sum().item())
            used |= (z > 0).any(0).cpu()
    return {'mse': sum(errors) / values.numel(),
            'l0': sum(nonzero) / len(values),
            'dead_fraction': float((~used).float().mean())}


def train(
    family: str,
    budget: int,
    seed: int,
    epochs: int = 12,
    device: str = 'cpu',
) -> None:
    """Train the selected surrogate and save its checkpoint and history."""
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    name = f'{family}_k{budget}_s{seed}'
    folder = ROOT / 'checkpoints' / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'model_codexgen.pt').exists():
        print('Resume:', name, flush=True)
        return
    x, val, mean, scale, sites = training_data()
    # Shared candidate points control the landmark choice across seeds.
    ids = torch.randperm(len(x), generator=torch.Generator().manual_seed(91))
    points = x[ids[:WIDTH]].clone()
    model = make(family, budget, points, device)
    x, val = x.to(device), val.to(device)
    if family == 'jump':
        with torch.no_grad():
            pre, _ = model.encoder(x[:512])
            threshold = torch.quantile(pre.cpu(), 1 - budget / WIDTH, dim=0)
            model.thresholds.copy_(threshold.clamp_min(0.05).log().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Budget is a target, not a post-hoc truncation, for soft penalties.
    penalty = 0.03 if family in ['jump', 'relu_fixed'] else 0.1
    history = []
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x), device=device)
        for batch_ids in order.split(512):
            batch = x[batch_ids]
            if family == 'batchtopk':
                model.top_k = len(batch) * budget
            pre, z, out = model(batch)
            mse = (out - batch).square().mean()
            loss = mse
            if family in ['relu', 'relu_fixed']:
                loss = loss + penalty * z.sum(1).mean() / budget
            elif family == 'jump':
                active = heaviside(torch.relu(pre), model.thresholds.exp(),
                                   model.kernel_fn, model.bandwidth)
                loss = loss + penalty * active.sum(1).mean() / budget
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        stats = score(model, val)
        stats.update(epoch=epoch + 1, penalty=penalty)
        history.append(stats)
        if family in ['relu', 'jump']:
            # Regulate sparsity on validation, never on test or transfer data.
            factor = np.clip(stats['l0'] / budget, 0.5, 2.0) ** 0.5
            penalty = float(np.clip(penalty * factor, 1e-5, 10))
        print(name, epoch + 1, stats, flush=True)
    model.cpu()
    checkpoint = {'state_dict': model.state_dict(), 'family': family,
                  'budget': budget, 'seed': seed, 'width': WIDTH,
                  'mean': torch.from_numpy(mean), 'scale': float(scale),
                  'points': points, 'train_sites': torch.from_numpy(sites),
                  'running_threshold': getattr(model, 'running_threshold',
                                               None)}
    if checkpoint['running_threshold'] is not None:
        checkpoint['running_threshold'] = checkpoint['running_threshold'].cpu()
    torch.save(checkpoint, folder / 'model_codexgen.pt')
    report = {'name': name, 'family': family, 'budget': budget, 'seed': seed,
              'width': WIDTH, 'epochs': epochs, 'training_images': 200,
              'training_sites': len(x), 'validation_images': 50,
              'seconds': time.monotonic() - started, 'device': device,
              'history': history, 'overcomplete_version': '0.3.0'}
    (folder / 'training_codexgen.json').write_text(
        json.dumps(report, indent=2) + '\n')


def restore(name: str) -> tuple[SAE, dict[str, Any]]:
    """Reconstruct a trained surrogate from its saved checkpoint."""
    checkpoint = torch.load(ROOT / f'checkpoints/{name}/model_codexgen.pt',
                            map_location='cpu', weights_only=True)
    model = make(checkpoint['family'], checkpoint['budget'],
                 checkpoint['points'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if checkpoint['running_threshold'] is not None:
        model.running_threshold = checkpoint['running_threshold']
    model.eval().requires_grad_(False)
    return model, checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    for family in ([args.family] if args.family else FAMILIES):
        for budget in [16, 32]:
            for seed in [0, 1, 2]:
                train(family, budget, seed, args.epochs, args.device)
