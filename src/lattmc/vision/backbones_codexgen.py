"""Frozen CNN and ViT adapters with explicit patch-token conventions."""

import numpy as np
import torch
from torch import nn
from transformers import AutoModel

from lattmc.vision.models_codexgen import TopKSAE
from lattmc.vision.natural_codexgen import feature_model
from lattmc.vision.paths_codexgen import experiment_root


REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'


def dataset():
    folder = experiment_root('imagenette_imagewoof') / 'dataset'
    with np.load(folder / 'imagenette_codexgen.npz') as data:
        result = {key: data[key] for key in data.files}
    with np.load(folder / 'imagewoof_codexgen.npz') as data:
        result['images'] = np.concatenate([result['images'], data['images']])
        result['labels'] = np.r_[result['labels'], data['labels']]
        result['splits'] = np.r_[result['splits'], ['transfer'] * 100]
        result['source_ids'] = np.r_[result['source_ids'], data['source_ids']]
        result['transfer_classes'] = data['classes']
    return result


class Backbone(nn.Module):
    """Input float RGB [0, 1]; output B x sites x channels, excluding CLS."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == 'resnet34':
            path = experiment_root('cifar10_resnet34') / (
                'checkpoints/resnet34_imagenet1k_v1_codexgen.pt')
            self.model = feature_model(path)
            self.width, self.grid = 256, 14
        elif name == 'dinov2':
            path = experiment_root('imagenette_dinov2') / 'checkpoints/model'
            self.model = AutoModel.from_pretrained(
                path, local_files_only=True, attn_implementation='eager')
            self.width, self.grid = 384, 16
        else:
            raise ValueError(name)
        self.model.eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, images):
        values = (images - self.mean) / self.std
        if self.name == 'resnet34':
            return self.model(values).flatten(2).transpose(1, 2)
        return self.model(values).last_hidden_state[:, 1:]


def pixels(images):
    return torch.tensor(images).permute(0, 3, 1, 2).float() / 255


def load_surrogate(name):
    folder = experiment_root('imagenette_' + name)
    state = torch.load(folder / 'checkpoints/sae_codexgen.pt',
                       map_location='cpu', weights_only=True)
    sae = TopKSAE(width=state['width'], latents=state['latents'], k=32)
    sae.load_state_dict(state['sae'])
    return sae.eval(), state['center'], state['scale']


def load_codes(name):
    folder = experiment_root('imagenette_' + name) / 'activations'
    blocks = []
    for p in sorted(folder.glob('codes_*_codexgen.npz')):
        with np.load(p) as cache:
            blocks.append({k: cache[k] for k in cache.files})
    return {key: np.concatenate([b[key] for b in blocks])
            for key in blocks[0]}
