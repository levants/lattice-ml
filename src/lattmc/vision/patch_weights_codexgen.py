"""Lossless checkpoint parts with SHA-256-verified offline reassembly."""

import argparse
import hashlib
import json
from pathlib import Path

from lattmc.vision.paths_codexgen import experiment_root


ROOT = experiment_root('patch_contexts') / 'checkpoints'
FILES = [('prisma_backbone', 'open_clip_model.safetensors'),
         ('prisma_sae', 'weights.pt'),
         ('saev_register_backbone', 'model.safetensors'),
         ('saev_sae', 'sae.pt')]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def split(path):
    path = Path(path)
    entries = []
    with path.open('rb') as stream:
        for index in range(1000):
            block = stream.read(80 * 1024 * 1024)
            if not block:
                break
            part = path.with_name(f'{path.name}.part{index:03d}')
            part.write_bytes(block)
            entries.append({'path': part.name, 'bytes': len(block),
                            'sha256': hashlib.sha256(block).hexdigest()})
    manifest = {'path': path.name, 'bytes': path.stat().st_size,
                'sha256': digest(path), 'parts': entries}
    path.with_name(path.name + '.parts.json').write_text(
        json.dumps(manifest, indent=2) + '\n')


def ensure(path):
    path = Path(path)
    if path.exists():
        return path
    receipt = path.with_name(path.name + '.parts.json')
    manifest = json.loads(receipt.read_text())
    temporary = path.with_name(path.name + '.reassembling')
    with temporary.open('wb') as stream:
        for entry in manifest['parts']:
            part = path.parent / entry['path']
            if digest(part) != entry['sha256']:
                raise ValueError(f'Corrupt checkpoint part: {part}')
            stream.write(part.read_bytes())
    if (temporary.stat().st_size != manifest['bytes'] or
            digest(temporary) != manifest['sha256']):
        raise ValueError(f'Checkpoint reassembly failed: {path}')
    temporary.replace(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', action='store_true')
    args = parser.parse_args()
    for folder, filename in FILES:
        path = ROOT / folder / filename
        if args.split:
            split(path)
        else:
            ensure(path)
        print(folder, digest(path), flush=True)
