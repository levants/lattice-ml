"""Small offline vision model and nonnegative Top-k sparse surrogate."""

import torch
from torch import nn
from torch.nn import functional as functional


class DigitCNN(nn.Module):
    """8x8 images -> 4x4 sites with 32 channels -> digit logits."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.head = nn.Linear(32 * 4 * 4, 10)

    def features(self, images):
        hidden = functional.max_pool2d(self.conv1(images).relu(), 2)
        return self.conv2(hidden).relu()

    def forward(self, images):
        return self.head(self.features(images).flatten(1))


class TopKSAE(nn.Module):
    """Linear ReLU Top-k encoder and unit-column linear decoder."""

    def __init__(self, width=32, latents=128, k=8):
        super().__init__()
        self.encoder = nn.Linear(width, latents)
        self.decoder = nn.Linear(latents, width)
        self.k = k
        self.normalize_decoder()

    def encode(self, hidden):
        positive = self.encoder(hidden).relu()
        values, indices = positive.topk(self.k, dim=-1)
        return torch.zeros_like(positive).scatter(-1, indices, values)

    def forward(self, hidden):
        return self.decoder(self.encode(hidden))

    @torch.no_grad()
    def normalize_decoder(self):
        weight = self.decoder.weight
        weight.div_(weight.norm(dim=0, keepdim=True).clamp_min(1e-12))
