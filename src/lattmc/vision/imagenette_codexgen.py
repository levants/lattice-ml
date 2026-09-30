"""Execute matched Imagenette CNN/ViT sparse-surrogate feature studies."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as functional

from lattmc.vision.backbones_codexgen import Backbone, dataset, pixels
from lattmc.vision.datasets_codexgen import sha256
from lattmc.vision.models_codexgen import TopKSAE
from lattmc.vision.paths_codexgen import experiment_root, prepare_folders


def run(name):
    torch.set_num_threads(4)
    torch.manual_seed(2027)
    torch.use_deterministic_algorithms(True)
    folder = prepare_folders(experiment_root('imagenette_' + name))
    sample = dataset()
    backbone = Backbone(name)
    images = sample['images']
    hidden = []
    with torch.no_grad():
        for start in range(0, len(images), 10):
            hidden.append(backbone(pixels(images[start:start + 10])))
            if start % 100 == 0:
                print(name, 'extract', start, '/', len(images), flush=True)
    hidden = torch.cat(hidden)
    train = np.flatnonzero(sample['splits'] == 'train')
    center = hidden[train].mean((0, 1))
    scale = hidden[train].std((0, 1), correction=0).clamp_min(1e-6)
    scaled = (hidden - center) / scale
    training = scaled[train].flatten(0, 1)
    sae = TopKSAE(width=backbone.width, latents=2 * backbone.width, k=32)
    with torch.no_grad():
        sae.encoder.weight.copy_(sae.decoder.weight.T)
        sae.encoder.bias.zero_()
        sae.decoder.bias.zero_()
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
    history = []
    for epoch in range(30):
        losses = []
        for indices in torch.randperm(len(training)).split(512):
            batch = training[indices]
            loss = functional.mse_loss(sae(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if epoch % 5 == 0:
            print(name, 'SAE epoch', epoch, history[-1], flush=True)
    sae.eval()
    codes, reconstruction = [], []
    with torch.no_grad():
        for batch in scaled.split(25):
            z = sae.encode(batch)
            codes.append(z)
            reconstruction.append(sae.decoder(z) * scale + center)
    codes, reconstruction = torch.cat(codes), torch.cat(reconstruction)
    metrics = {}
    for split in ['train', 'calibration', 'test', 'transfer']:
        ids = np.flatnonzero(sample['splits'] == split)
        residual = (hidden[ids] - reconstruction[ids]).square().sum()
        total = (hidden[ids] - center).square().sum()
        metrics[split] = {'count': len(ids), 'r2': float(1 - residual / total),
                         'site_l0': float((codes[ids] > 0).float()
                                          .sum(-1).mean())}
    for start in range(0, len(images), 75):
        selection = slice(start, start + 75)
        np.savez_compressed(
            folder / f'activations/codes_{start:04d}_codexgen.npz',
            rows=np.arange(len(images))[selection],
            dense=hidden[selection].numpy(), codes=codes[selection].numpy())
    torch.save({'sae': sae.state_dict(), 'center': center, 'scale': scale,
                'width': backbone.width, 'latents': 2 * backbone.width},
               folder / 'checkpoints/sae_codexgen.pt')
    artifacts = sorted(folder.glob('activations/*.npz')) + list(
        folder.glob('checkpoints/sae_codexgen.pt'))
    result = {'model': name, 'seed': 2027, 'epochs': 30, 'top_k': 32,
              'latents': 2 * backbone.width, 'grid': backbone.grid,
              'training_history': history, 'metrics': metrics,
              'training_alive': int((codes[train] > 0).any(0).any(0).sum()),
              'artifact_sha256': {str(p.relative_to(folder)): sha256(p)
                                  for p in artifacts}}
    (folder / 'results/experiment_codexgen.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(name, metrics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['resnet34', 'dinov2'])
    run(parser.parse_args().model)
