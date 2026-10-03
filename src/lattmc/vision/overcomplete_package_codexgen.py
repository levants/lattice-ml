"""Prepare checksum-verified release assets without manuscript files."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from lattmc.vision.overcomplete_fetch_codexgen import ROOT, digest


def package() -> None:
    """Package the experiment sources and cached reproducibility artifacts."""
    output = ROOT / 'releases'
    output.mkdir(exist_ok=True)
    groups = {
        'inputs': sorted((ROOT / 'dataset').glob('*')),
        'activations': sorted((ROOT / 'activations').rglob('*')),
        'codes': sorted((ROOT / 'codes').rglob('*')),
        'weights': sorted((ROOT / 'checkpoints').rglob('*'))}
    manifest, assets = [], []
    for category, candidates in groups.items():
        paths = [p for p in candidates if p.is_file()
                 and p.suffix not in ['.py', '.pth']
                 and p.name != 'weights.pt']
        batches, batch, size = [], [], 0
        for path in paths:
            if batch and size + path.stat().st_size > 450 * 2 ** 20:
                batches.append(batch)
                batch, size = [], 0
            batch.append(path)
            size += path.stat().st_size
        if batch:
            batches.append(batch)
        for index, batch in enumerate(batches):
            target = output / f'vision-overcomplete-{category}-{index:02}.zip'
            with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as archive:
                for path in batch:
                    relative = 'vision_tokens/overcomplete/' + str(
                        path.relative_to(ROOT))
                    assert path.suffix not in ['.tex', '.pdf', '.bbl', '.bib']
                    archive.write(path, relative)
                    manifest.append({'path': relative,
                                     'bytes': path.stat().st_size,
                                     'sha256': digest(path),
                                     'asset': target.name})
            with zipfile.ZipFile(target) as archive:
                assert archive.testzip() is None
            assets.append({'name': target.name, 'bytes': target.stat().st_size,
                           'sha256': digest(target)})
            print(target.name, target.stat().st_size, flush=True)
    result = {'files': manifest, 'assets': assets,
              'upstream_weights': {
                  'ra_sae': 'matybohacek/RA-SAE-DINOv2-32k',
                  'revision': '1e10a216938e112302b31e3bb2f69818e59a12a9',
                  'transcoder': 'Prisma-Multimodal/'
                  'CLIP-transcoder-topk-256-x64-all_patches_1-mlp-94',
                  'transcoder_revision':
                  '96c293a7299a99fe3d70f1f15498215849f22d96'}}
    (ROOT / 'release_manifest_codexgen.json').write_text(
        json.dumps(result, indent=2) + '\n')
    sums = '\n'.join(f"{a['sha256']}  {a['name']}" for a in assets) + '\n'
    (output / 'SHA256SUMS.txt').write_text(sums)
    (output / 'release_manifest_codexgen.json').write_text(
        json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    package()
