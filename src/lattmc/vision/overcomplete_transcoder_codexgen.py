"""Native pretrained Prisma TopK transcoder with explicit skip diagnostics."""

import argparse
import json

import numpy as np
import torch
from scipy import sparse
from torchvision.transforms import functional as tf
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.transcoder import Transcoder

from lattmc.vision.overcomplete_data_codexgen import load
from lattmc.vision.overcomplete_fetch_codexgen import ROOT
from lattmc.vision.patch_models_codexgen import CLIP_ID, ROOT as PATCH_ROOT


NAME = 'prisma_transcoder'


def backbone():
    from lattmc.vision.patch_weights_codexgen import ensure
    from safetensors.torch import load_file
    from vit_prisma.models.base_vit import HookedViT
    from vit_prisma.models.model_loader import load_config
    from vit_prisma.models.weight_conversion import convert_open_clip_weights
    from vit_prisma.utils.enums import ModelType
    path = PATCH_ROOT / 'checkpoints/prisma_backbone'
    ensure(path / 'open_clip_model.safetensors')
    cfg = load_config(CLIP_ID, ModelType.VISION, local_path=str(path))
    cfg.device = 'cpu'
    model = HookedViT(cfg)
    weights = load_file(str(path / 'open_clip_model.safetensors'))
    model.load_state_dict(convert_open_clip_weights(weights, cfg), strict=True)
    norm = json.loads((path / 'open_clip_config.json').read_text())
    return model.eval().requires_grad_(False), norm['preprocess_cfg']


def extract(dataset):
    torch.set_num_threads(4)
    folder = ROOT / 'checkpoints/transcoder'
    config = VisionModelSAERunnerConfig.load_config(
        str(folder / 'config.json'))
    config.device = 'cpu'
    model = Transcoder(config).eval().requires_grad_(False)
    with torch.serialization.safe_globals([VisionModelSAERunnerConfig]):
        saved = torch.load(folder / 'weights.pt', map_location='cpu',
                           weights_only=True)
    model.load_state_dict(saved['state_dict'], strict=True)
    vision, norm = backbone()
    sample, records = load(dataset)
    output_dir = ROOT / 'codes' / NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f'{dataset}_codexgen.npz'
    if target.exists():
        return
    blocks, errors, skip_errors, sums, sumsq = [], [], [], [], []
    checks, dense_cache = [], []
    hooks = ['blocks.1.ln2.hook_normalized', 'blocks.1.hook_mlp_out']
    with torch.inference_mode():
        for start in range(0, len(records), 10):
            pixels = torch.from_numpy(sample['images'][start:start + 10])
            pixels = pixels.permute(0, 3, 1, 2).float() / 255
            pixels = tf.normalize(pixels, norm['mean'], norm['std'])
            _, cache = vision.run_with_cache(pixels, names_filter=hooks)
            x, y = [cache[h][:, 1:].contiguous() for h in hooks]
            flat_x, flat_y = x.flatten(0, 1), y.flatten(0, 1)
            prediction, codes, *_ = model(flat_x, flat_y)
            mu = flat_x.mean(-1, keepdim=True)
            std = flat_x.std(-1, keepdim=True)
            pre = ((flat_x - mu) / (std + 1e-5) - model.b_dec)
            pre = pre @ model.W_enc + model.b_enc
            values, indices = torch.topk(torch.relu(pre), 256, dim=-1)
            reference = torch.zeros_like(pre).scatter(-1, indices, values)
            checks.append(float((reference - codes).abs().max()))
            skip = (flat_x @ model.W_skip.T) * std + mu
            reconstructed = ((codes @ model.W_dec + model.b_dec_out
                              + flat_x @ model.W_skip.T) * std + mu)
            checks.append(float((reconstructed - prediction).abs().max()))
            assert torch.isfinite(codes).all() and (codes >= 0).all()
            blocks.append(sparse.csr_matrix(codes.numpy()))
            errors.extend((prediction - flat_y).square().sum(1).reshape(
                -1, 49).sum(1).tolist())
            skip_errors.extend((skip - flat_y).square().sum(1).reshape(
                -1, 49).sum(1).tolist())
            sums.extend(y.sum(1).numpy())
            sumsq.extend(y.square().sum((1, 2)).tolist())
            dense_cache.append({'input': x.numpy(), 'target': y.numpy()})
            print(NAME, dataset, start + len(x), flush=True)
    matrix = sparse.vstack(blocks, format='csr')
    if dataset == 'imagenette':
        train = np.array([r['split'] == 'train' for r in records])
        mean = np.array(sums)[train].sum(0) / (49 * train.sum())
        np.save(folder / 'output_mean_codexgen.npy', mean)
    else:
        mean = np.load(folder / 'output_mean_codexgen.npy')
    baseline = np.array(sumsq) - 2 * np.array(sums) @ mean + 49 * (mean @ mean)
    np.savez_compressed(target, data=matrix.data, indices=matrix.indices,
                        indptr=matrix.indptr, shape=matrix.shape,
                        error=errors, baseline=baseline,
                        skip_error=skip_errors)
    dense_folder = ROOT / 'activations' / NAME / dataset
    dense_folder.mkdir(parents=True, exist_ok=True)
    for index, values in enumerate(dense_cache):
        np.savez_compressed(dense_folder / f'chunk_{index:03d}.npz', **values)
    receipt = {'native_vs_explicit_max_error': max(checks),
               'hooks': hooks, 'spatial_sites': 49,
               'normalization': 'native sample std with correction=1',
               'skip_connection': True}
    (dense_folder / 'verification_codexgen.json').write_text(
        json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    extract(parser.parse_args().dataset)
