"""Pinned upstream vision SAEs, model-specific crops, and spatial tokens."""

import io
import json
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tf

from lattmc.vision.paths_codexgen import experiment_root
from lattmc.vision.patch_weights_codexgen import ensure


ROOT = experiment_root('patch_contexts')
CLIP_ID = 'open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K'


def normalization():
    path = ROOT / 'upstream/saev_normalization.json'
    values = json.loads(path.read_text())
    assert len(values['mean']) == 768
    return torch.tensor(values['mean']), values['scalar']


def dataset(name):
    folder = experiment_root('imagenette_imagewoof') / 'dataset'
    with np.load(folder / 'imagenette_codexgen.npz') as data:
        sample = {k: data[k] for k in data.files}
    rows = np.flatnonzero(np.isin(sample['splits'], ['train', 'test']))
    images = []
    with zipfile.ZipFile(folder / 'imagenette_originals_codexgen.zip') as z:
        for row in rows:
            img = Image.open(io.BytesIO(z.read(sample['source_ids'][row])))
            img = img.convert('RGB')
            if name == 'prisma':
                img = tf.resize(img, 224, InterpolationMode.BICUBIC)
            else:
                img = tf.resize(img, 256, InterpolationMode.BILINEAR)
            images.append(np.array(tf.center_crop(img, [224, 224])))
    return {'images': np.stack(images), 'rows': rows,
            'labels': sample['labels'][rows],
            'splits': sample['splits'][rows], 'classes': sample['classes'],
            'source_ids': sample['source_ids'][rows]}


class Adapter:
    """Use actual upstream encoders; omit CLS and register tokens."""

    def __init__(self, name):
        self.name = name
        folder = ROOT / 'checkpoints'
        if name == 'prisma':
            ensure(folder / 'prisma_backbone/open_clip_model.safetensors')
            ensure(folder / 'prisma_sae/weights.pt')
        else:
            ensure(folder / 'saev_register_backbone/model.safetensors')
            ensure(folder / 'saev_sae/sae.pt')
        if name == 'prisma':
            from safetensors.torch import load_file
            from vit_prisma.models.base_vit import HookedViT
            from vit_prisma.models.model_loader import load_config
            from vit_prisma.models.weight_conversion import (
                convert_open_clip_weights)
            from vit_prisma.sae import StandardSparseAutoencoder
            from vit_prisma.sae.config import VisionModelSAERunnerConfig
            from vit_prisma.utils.enums import ModelType
            path = folder / 'prisma_backbone'
            cfg = load_config(CLIP_ID, ModelType.VISION, local_path=str(path))
            cfg.device = 'cpu'
            self.model = HookedViT(cfg)
            weights = load_file(str(path / 'open_clip_model.safetensors'))
            converted = convert_open_clip_weights(weights, cfg)
            self.model.load_state_dict(converted, strict=True)
            del weights, converted
            config = VisionModelSAERunnerConfig.load_config(
                str(folder / 'prisma_sae/config.json'))
            config.device = 'cpu'
            self.sae = StandardSparseAutoencoder(config)
            weights = torch.load(folder / 'prisma_sae/weights.pt',
                                 map_location='cpu', weights_only=True)
            self.sae.load_state_dict(weights)
            norm = json.loads((path / 'open_clip_config.json').read_text())
            norm = norm['preprocess_cfg']
            self.mean, self.std = norm['mean'], norm['std']
            self.grid, self.patch = 7, 32
        elif name == 'saev':
            import saev.nn
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                folder / 'saev_register_backbone', local_files_only=True,
                attn_implementation='eager')
            self.sae = saev.nn.load(folder / 'saev_sae/sae.pt')
            self.center, self.scalar = normalization()
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.grid, self.patch = 16, 14
        else:
            raise ValueError(name)
        self.model.eval().requires_grad_(False)
        self.sae.eval().requires_grad_(False)

    def dense(self, images):
        x = torch.from_numpy(np.ascontiguousarray(images))
        x = x.permute(0, 3, 1, 2).float() / 255
        x = tf.normalize(x, self.mean, self.std)
        if self.name == 'prisma':
            hook = 'blocks.11.hook_resid_post'
            _, cache = self.model.run_with_cache(x, names_filter=[hook])
            return cache[hook][:, 1:].contiguous()
        output = self.model(x, output_hidden_states=True)
        return output.hidden_states[11][:, 5:].contiguous()

    def normalize(self, dense, reference_clip=False):
        if self.name == 'prisma':
            return dense
        lower = -1e-5 if reference_clip else -1e5
        return (dense.clamp(lower, 1e5) - self.center) / self.scalar

    def encode(self, values):
        shape = values.shape[:-1]
        flat = values.reshape(-1, values.shape[-1])
        if self.name == 'prisma':
            codes = self.sae.encode(flat)[1]
        else:
            # The legacy checkpoint centers by decoder bias before encoding.
            codes = self.sae.encode(flat - self.sae.b_dec).f_x
        return codes.reshape(*shape, -1)

    def decode(self, codes):
        flat = codes.reshape(-1, codes.shape[-1])
        recovered = self.sae.decode(flat)
        if self.name == 'saev':
            recovered = recovered[:, 0]
        return recovered.reshape(*codes.shape[:-1], -1)
