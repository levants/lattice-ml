"""Audited loading and raw-Hugging-Face activation hooks."""

from __future__ import annotations
from typing import Any
from collections.abc import Callable

import json
from pathlib import Path

from sae_lens import SAE
from safetensors.torch import load_file
import torch

from .common_codexgen import sha256


class ActivationReady(Exception):
    """Stop inference once both the encoder input and target are captured."""


def cache_files(
    repo: str,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    """Inventory files and hashes for one cached model revision."""
    base = Path.home() / '.cache/huggingface/hub'
    snapshots = base / ('models--' + repo.replace('/', '--')) / 'snapshots'
    if revision is None:
        revision = (snapshots.parent / 'refs/main').read_text().strip()
    root = snapshots / revision
    return [dict(revision=revision, path=str(p.relative_to(root)),
                 bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(root.rglob('*')) if p.is_file()]


def load_smol(path: Path, device: str) -> SAE:
    """Preserve Sparsify's centered TopK formula and MLP output identity."""
    native = json.loads((path / 'cfg.json').read_text())
    original = load_file(str(path / 'sae.safetensors'), device='cpu')
    width = (native.get('num_latents')
             or native['d_in'] * native['expansion_factor'])
    sae = SAE.from_dict(dict(
        architecture='topk', d_in=native['d_in'], d_sae=width,
        dtype='float32', device='cpu', k=native['k'],
        apply_b_dec_to_input=True, normalize_activations='none',
        metadata=dict(model_name='HuggingFaceTB/SmolLM2-135M',
                      hook_name='blocks.15.hook_mlp_out')))
    weights = dict(W_enc=original['encoder.weight'].T.float(),
                   W_dec=original['W_dec'].float(),
                   b_enc=original['encoder.bias'].float(),
                   b_dec=original['b_dec'].float())
    assert weights['W_dec'].shape == (width, native['d_in'])
    sae.load_state_dict(weights, strict=True)
    # Check the native centered TopK equation independently of conversion.
    generator = torch.Generator().manual_seed(20261001)
    x = torch.randn(7, native['d_in'], generator=generator)
    encoder = original['encoder.weight'].float()
    bias = original['encoder.bias'].float()
    center = original['b_dec'].float()
    pre = torch.nn.functional.linear(x - center, encoder, bias).relu()
    values, indices = pre.topk(native['k'], dim=-1)
    expected = torch.zeros_like(pre).scatter(-1, indices, values)
    torch.testing.assert_close(sae.encode(x), expected, rtol=1e-5, atol=1e-5)
    decoded = expected @ original['W_dec'].float() + center
    torch.testing.assert_close(sae.decode(expected), decoded,
                               rtol=1e-5, atol=1e-5)
    return sae.to(device).eval()


def load_surrogate(
    name: str,
    cfg: dict[str, Any],
    pinned: dict[str, Any],
    device: str,
) -> SAE:
    """Load the registered surrogate from its pinned checkpoint."""
    if name == 'smol_topk':
        revision = pinned[cfg['release']]['revision']
        root = (Path.home() / '.cache/huggingface/hub'
                / ('models--' + cfg['release'].replace('/', '--'))
                / 'snapshots' / revision)
        return load_smol(root / cfg['sae_id'], device)
    return SAE.from_pretrained(cfg['release'], cfg['sae_id'],
                               device=device, dtype='float32').eval()


def attach(
    model: torch.nn.Module,
    cfg: dict[str, Any],
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Capture exactly the module input/output described by each release."""
    layers = (model.gpt_neox.layers if hasattr(model, 'gpt_neox')
              else model.model.layers)
    block = layers[cfg['layer']]
    module = block if cfg['hook'] == 'residual_post' else block.mlp
    captured = {}

    def pre_hook(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        """Capture the module input before the forward pass."""
        captured['input'] = args[0]

    def post_hook(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        value: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        """Capture the target activation and stop the forward pass."""
        value = value[0] if isinstance(value, tuple) else value
        captured['target'] = value.float()
        if cfg['hook'] == 'mlp_input_output':
            captured['input'] = captured['input'].float()
        else:
            captured['input'] = captured['target']
        raise ActivationReady()

    if cfg['hook'] == 'mlp_input_output':
        module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)

    def run(
        ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the captured encoder input and reconstruction target."""
        captured.clear()
        try:
            model(input_ids=ids, attention_mask=mask, use_cache=False)
        except ActivationReady:
            pass
        assert set(captured) == {'input', 'target'}
        return captured['input'], captured['target']
    return run
