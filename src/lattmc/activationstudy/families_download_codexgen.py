"""Restore pinned family weights and verify their recorded hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from sae_lens.loading.pretrained_saes_directory import (
    get_repo_id_and_folder_name,
)

from .common_codexgen import sha256
from .families_config_codexgen import FAMILIES


def download(output: Path) -> None:
    """Download the registered inputs and save provenance receipts."""
    for family, cfg in FAMILIES.items():
        path = output / 'ag_news' / f'{family}_extraction.json'
        report = json.loads(path.read_text())
        if family == 'smol_topk':
            repo, folder = cfg['release'], cfg['sae_id']
        else:
            repo, folder = get_repo_id_and_folder_name(
                cfg['release'], cfg['sae_id'])
        for kind, name in (('backbone_files', cfg['model']),
                           ('sae_files', repo)):
            for item in report[kind]:
                # A cache may contain unrelated checkpoints in the same repo.
                if kind == 'sae_files':
                    selected = (item['path'].startswith(
                        folder.rstrip('/') + '/') or item['path'] in (
                            folder + '.safetensors', 'config.yaml',
                            'wandb-config.yaml', 'wanb-config.yaml'))
                    if not selected:
                        continue
                local = hf_hub_download(name, item['path'],
                                        revision=item['revision'])
                assert sha256(local) == item['sha256']
        # Registry converters request main; pin an absent local cache alias.
        # Never overwrite a newer alias used by another experiment.
        revision = report['sae_files'][0]['revision']
        ref = (Path(HF_HUB_CACHE) / ('models--' + repo.replace('/', '--'))
               / 'refs/main')
        if ref.exists():
            if ref.read_text().strip() != revision:
                raise RuntimeError('Cache main has changed. Restore the '
                                   'pinned revision in a fresh environment.')
        else:
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(revision)
        print(family, 'pinned files verified', flush=True)
    print('Use the recorded revisions; do not replace them with new weights.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    download(parser.parse_args().output)
