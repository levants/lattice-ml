"""Extract additional surrogate families, with target-correct diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from scipy import sparse
import torch
from transformers import AutoModelForCausalLM

from .common_codexgen import save_json, sha256
from .families_adapters_codexgen import attach, cache_files, load_surrogate
from .families_config_codexgen import FAMILIES
from sae_lens.loading.pretrained_saes_directory import (
    get_repo_id_and_folder_name)


def extract(
    output: Path,
    family: str,
    device: str = 'mps',
    batch_size: int = 4,
) -> None:
    """Extract and cache sparse activations with model and input provenance."""
    torch.set_num_threads(4)
    started = time.time()
    cfg = FAMILIES[family]
    first = json.loads((output / 'ag_news/design.json').read_text())
    pinned = first['pinned']
    sae = load_surrogate(family, cfg, pinned, device)
    assert sae.cfg.normalize_activations == 'none'
    dtype = getattr(torch, cfg['dtype'])
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model'], revision=pinned[cfg['model']]['revision'],
        local_files_only=True, dtype=dtype,
        attn_implementation='eager').to(device).eval()
    capture = attach(model, cfg)
    repo = (cfg['release'] if family == 'smol_topk' else
            get_repo_id_and_folder_name(cfg['release'], cfg['sae_id'])[0])
    sae_revision = (pinned[repo]['revision'] if repo in pinned else None)
    source_hashes = {name: sha256(Path(__file__).with_name(name))
                     for name in ('families_adapters_codexgen.py',
                                  'families_config_codexgen.py',
                                  'common_codexgen.py')}
    provenance = dict(
        family=family, configuration=cfg, dtype=cfg['dtype'], device=device,
        backbone_files=cache_files(cfg['model'],
                                   pinned[cfg['model']]['revision']),
        sae_files=cache_files(repo, sae_revision),
        sae_config=sae.cfg.to_dict(), source_sha256=sha256(__file__),
        companion_hashes=source_hashes,
        versions={k: metadata.version(k) for k in
                  ('torch', 'transformers', 'sae-lens', 'numpy', 'scipy')})
    print(family, 'loaded', sae.cfg.d_in, sae.cfg.d_sae, flush=True)
    for dataset in ('ag_news', 'dbpedia_14'):
        target = output / dataset
        design = json.loads((target / 'design.json').read_text())
        tokens = np.load(target / f'{family}_tokens.local.npz')
        total = len(design['rows'])
        if all((target / f'{v}_extraction.json').exists()
               for v in cfg['views']):
            for view in cfg['views']:
                old = json.loads((target / f'{view}_extraction.json')
                                 .read_text())
                assert old['source_sha256'] == sha256(__file__)
                assert old['companion_hashes'] == source_hashes
                assert old['design_sha256'] == sha256(target / 'design.json')
            print(family, dataset, 'already complete', flush=True)
            continue
        values = {v: dict(max=[], mean=[], dense=[]) for v in cfg['views']}
        stats = {v: dict(test_tokens=0, active_sum=0., squared_error=0.,
                        squared_input=0., zero_documents=0)
                 for v in cfg['views']}
        with torch.inference_mode():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                ids = torch.tensor(tokens['input_ids'][start:end],
                                   device=device)
                mask = torch.tensor(tokens['attention_mask'][start:end],
                                    device=device)
                x, y = capture(ids, mask)
                assert x.shape[-1] == sae.cfg.d_in
                z = sae.encode(x)
                assert torch.isfinite(z).all() and (z >= 0).all()
                valid = mask.clone()
                valid[:, 0] = False
                test_rows = torch.tensor([r['split'] == 'test'
                    for r in design['rows'][start:end]], device=device)
                test_mask = valid & test_rows[:, None]
                dense = x.masked_fill(~valid[:, :, None], 0.).sum(1)
                dense /= valid.sum(-1)[:, None]
                for view, width in cfg['views'].items():
                    width = width or z.shape[-1]
                    code = z[:, :, :width]
                    pooled = code.masked_fill(~valid[:, :, None], 0.)
                    maximum = pooled.amax(1)
                    mean = pooled.sum(1) / valid.sum(-1)[:, None]
                    values[view]['max'].append(sparse.csr_matrix(
                        maximum.cpu().numpy()))
                    values[view]['mean'].append(sparse.csr_matrix(
                        mean.cpu().numpy()))
                    values[view]['dense'].append(dense.cpu().numpy())
                    stat = stats[view]
                    stat['zero_documents'] += int(
                        (maximum.sum(-1) == 0).sum().item())
                    if test_mask.any():
                        tz, ty = code[test_mask], y[test_mask]
                        rec = tz @ sae.W_dec[:width] + sae.b_dec
                        stat['test_tokens'] += len(ty)
                        stat['active_sum'] += (tz > 0).sum().item()
                        stat['squared_error'] += (
                            (rec - ty).square().sum().item())
                        stat['squared_input'] += ty.square().sum().item()
                if start % 200 == 0:
                    print(family, dataset, end, '/', total,
                          'seconds', round(time.time()-started), flush=True)
        for view in cfg['views']:
            files = {}
            for pooling in ('max', 'mean'):
                path = target / f'{view}_{pooling}.npz'
                matrix = sparse.vstack(values[view][pooling], format='csr')
                sparse.save_npz(path, matrix)
                files[pooling] = dict(name=path.name, sha256=sha256(path),
                                     shape=matrix.shape, nnz=matrix.nnz)
            path = target / f'{view}_dense.npz'
            np.savez_compressed(path, values=np.vstack(values[view]['dense']))
            files['dense'] = dict(name=path.name, sha256=sha256(path))
            stat = stats[view]
            stat['mean_token_l0'] = stat['active_sum'] / stat['test_tokens']
            stat['uncentered_nmse'] = (stat['squared_error']
                                      / stat['squared_input'])
            report = dict(**provenance, checkpoint=view, files=files,
                          statistics=stat,
                          design_sha256=sha256(target / 'design.json'),
                          token_sha256=sha256(
                              target / f'{family}_tokens.local.npz'),
                          elapsed_seconds=time.time()-started)
            save_json(target / f'{view}_extraction.json', report)
            print(view, dataset, stat, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    extract(args.output, args.family, args.device, args.batch_size)
