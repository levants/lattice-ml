"""Extract registered SAELens activations from cached backbones."""

from __future__ import annotations
from typing import Any

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from scipy import sparse
import torch
from sae_lens import SAE
from sae_lens.loading.pretrained_saes_directory import (
    get_repo_id_and_folder_name,
)

from .common_codexgen import CHECKPOINTS, save_json, sha256


class CapturedActivation(Exception):
    """Stop a backbone after the requested hook, avoiding later layers."""


def cached_files(repo: str, folder: str | None = None) -> list[dict[str, Any]]:
    """Inventory checkpoint files and their hashes in cached snapshots."""
    base = Path.home() / '.cache/huggingface/hub'
    snapshots = base / ('models--' + repo.replace('/', '--')) / 'snapshots'
    result = []
    for snapshot in sorted(snapshots.iterdir()):
        root = snapshot / folder if folder else snapshot
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.npz',
                                                  '.safetensors'):
                result.append(dict(revision=snapshot.name,
                                   path=str(path.relative_to(snapshot)),
                                   bytes=path.stat().st_size,
                                   sha256=sha256(path)))
    if not result:
        raise FileNotFoundError((repo, folder))
    return result


def extract(
    output: Path,
    backbone: str,
    device: str = 'mps',
    batch_size: int = 4,
) -> None:
    """Extract and cache sparse activations with model and input provenance."""
    torch.set_num_threads(4)
    names = [key for key in CHECKPOINTS if key.startswith(backbone)]
    saes = {name: SAE.from_pretrained(*CHECKPOINTS[name], device=device)
            .eval() for name in names}
    assert all(sae.cfg.normalize_activations == 'none'
               for sae in saes.values())
    started = time.time()
    if backbone == 'gpt2':
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(
            'gpt2-small', device=device, local_files_only=True).eval()
        precision = 'float32'
        repo = 'gpt2'
        hooks = [sae.cfg.metadata.hook_name for sae in saes.values()]

        def activations(
            ids: torch.Tensor,
            mask: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            """Capture the requested backbone activations for a token batch."""
            _, cache = model.run_with_cache(
                ids, attention_mask=mask, names_filter=hooks,
                stop_at_layer=9, return_type=None)
            return {name: cache[saes[name].cfg.metadata.hook_name]
                    for name in names}
    else:
        from transformers import AutoModelForCausalLM
        repo = 'google/gemma-2-2b'
        model = AutoModelForCausalLM.from_pretrained(
            repo, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='eager').to(device).eval()
        precision = 'bfloat16'
        captured = {}

        def hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            outputs: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            """Store the requested activation and stop the remaining forward
            pass.
            """
            captured['x'] = (outputs[0] if isinstance(outputs, tuple)
                             else outputs)
            raise CapturedActivation()

        model.model.layers[8].register_forward_hook(hook)

        def activations(
            ids: torch.Tensor,
            mask: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            """Capture the requested backbone activations for a token batch."""
            try:
                model.model(input_ids=ids, attention_mask=mask,
                            use_cache=False)
            except CapturedActivation:
                pass
            value = captured.pop('x').float()
            return {name: value for name in names}
    provenance = dict(
        backbone=repo, precision=precision, device=device,
        backbone_files=cached_files(repo),
        versions={name: metadata.version(name) for name in
                  ('torch', 'transformers', 'sae-lens', 'transformer-lens')},
        source_sha256=sha256(__file__),
    )
    for dataset in ('ag_news', 'dbpedia_14'):
        target = output / dataset
        design = json.loads((target / 'design.json').read_text())
        tokens = np.load(target / f'{backbone}_tokens.local.npz')
        total = len(design['rows'])
        paths = [target / f'{name}_extraction.json' for name in names]
        if all(path.exists() for path in paths):
            for path in paths:
                previous = json.loads(path.read_text())
                assert previous['source_sha256'] == provenance['source_sha256']
                assert previous['design_sha256'] == sha256(
                    target / 'design.json')
            print('Already extracted', backbone, dataset, flush=True)
            continue
        values = {name: dict(max=[], mean=[], dense=[]) for name in names}
        stats = {name: dict(test_tokens=0, active_sum=0., squared_error=0.,
                            squared_input=0., zero_documents=0)
                 for name in names}
        with torch.inference_mode():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                ids = torch.tensor(tokens['input_ids'][start:end],
                                   device=device)
                mask = torch.tensor(tokens['attention_mask'][start:end],
                                    device=device)
                found = activations(ids, mask)
                valid = mask.clone()
                valid[:, 0] = False
                assert valid.sum(-1).min().item() > 0
                test = torch.tensor([
                    r['split'] == 'test' for r in design['rows'][start:end]
                ], device=device)
                test_mask = valid & test[:, None]
                for name, sae in saes.items():
                    x = found[name].float()
                    z = sae.encode(x)
                    assert torch.isfinite(z).all() and (z >= 0).all()
                    pooled = z.masked_fill(~valid[:, :, None], 0.)
                    maxima = pooled.amax(dim=1)
                    means = pooled.sum(dim=1) / valid.sum(-1)[:, None]
                    dense = x.masked_fill(~valid[:, :, None], 0.).sum(dim=1)
                    dense /= valid.sum(-1)[:, None]
                    values[name]['max'].append(sparse.csr_matrix(
                        maxima.cpu().numpy()))
                    values[name]['mean'].append(sparse.csr_matrix(
                        means.cpu().numpy()))
                    values[name]['dense'].append(dense.cpu().numpy())
                    stats[name]['zero_documents'] += int(
                        (maxima.sum(-1) == 0).sum().item())
                    if test_mask.any():
                        tz, tx = z[test_mask], x[test_mask]
                        rec = sae.decode(tz)
                        stats[name]['test_tokens'] += len(tx)
                        stats[name]['active_sum'] += (tz > 0).sum().item()
                        stats[name]['squared_error'] += (
                            (rec - tx).square().sum().item())
                        stats[name]['squared_input'] += (
                            tx.square().sum().item())
                if start % (batch_size * 25) == 0:
                    print(backbone, dataset, end, '/', total,
                          'seconds', round(time.time() - started), flush=True)
        for name, sae in saes.items():
            files = {}
            for pooling in ('max', 'mean'):
                path = target / f'{name}_{pooling}.npz'
                matrix = sparse.vstack(values[name][pooling], format='csr')
                sparse.save_npz(path, matrix)
                files[pooling] = dict(name=path.name, sha256=sha256(path),
                                     shape=matrix.shape, nnz=matrix.nnz)
            path = target / f'{name}_dense.npz'
            np.savez_compressed(path, values=np.vstack(values[name]['dense']))
            files['dense'] = dict(name=path.name, sha256=sha256(path))
            stats[name]['mean_token_l0'] = (
                stats[name]['active_sum'] / stats[name]['test_tokens'])
            stats[name]['uncentered_nmse'] = (
                stats[name]['squared_error'] / stats[name]['squared_input'])
            repo_id, folder = get_repo_id_and_folder_name(*CHECKPOINTS[name])
            report = dict(**provenance, checkpoint=name,
                          release=CHECKPOINTS[name][0],
                          sae_id=CHECKPOINTS[name][1],
                          sae_repo=repo_id,
                          sae_files=cached_files(repo_id, folder),
                          hook=sae.cfg.metadata.hook_name,
                          design_sha256=sha256(target / 'design.json'),
                          token_sha256=sha256(
                              target / f'{backbone}_tokens.local.npz'),
                          files=files, statistics=stats[name],
                          elapsed_seconds=time.time() - started)
            save_json(target / f'{name}_extraction.json', report)
            print(dataset, name, stats[name], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backbone', choices=['gpt2', 'gemma2'],
                        required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    extract(args.output, args.backbone, args.device, args.batch_size)
