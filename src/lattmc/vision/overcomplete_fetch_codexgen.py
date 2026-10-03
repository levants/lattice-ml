"""Fetch pinned public inputs; keep large artifacts outside Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lattmc.vision.paths_codexgen import experiment_root


ROOT = experiment_root('overcomplete')
HF_ID = 'matybohacek/RA-SAE-DINOv2-32k'
HF_REV = '1e10a216938e112302b31e3bb2f69818e59a12a9'


def digest(path: Path) -> str:
    """Compute the SHA-256 digest of a downloaded file."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(url: str, path: Path | str) -> Path:
    """Download validated byte ranges and save a provenance receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + '.partial')
        request = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers['Content-Length'])
        start = partial.stat().st_size if partial.exists() else 0
        width = 32 * 2 ** 20
        ranges = [(lo, min(total, lo + width))
                  for lo in range(start, total, width)]

        def piece(bounds: tuple[int, int]) -> Path:
            """Download and validate one byte-range chunk."""
            lo, hi = bounds
            part = path.with_suffix(path.suffix + f'.range{lo}')
            if not part.exists() or part.stat().st_size != hi - lo:
                req = urllib.request.Request(url, headers={
                    'Range': f'bytes={lo}-{hi - 1}'})
                with urllib.request.urlopen(req, timeout=120) as src:
                    if len(ranges) == 1 and lo == 0 and src.status == 200:
                        pass
                    else:
                        assert src.status == 206, src.status
                        assert src.headers['Content-Range'].startswith(
                            f'bytes {lo}-{hi - 1}/')
                    with part.open('wb') as dst:
                        while chunk := src.read(2 ** 20):
                            dst.write(chunk)
                assert part.stat().st_size == hi - lo
            return part

        with ThreadPoolExecutor(max_workers=8) as pool:
            with partial.open('ab') as dst:
                for part in pool.map(piece, ranges):
                    with part.open('rb') as src:
                        while chunk := src.read(2 ** 20):
                            dst.write(chunk)
                    part.unlink()
                    print(path.name, dst.tell(), '/', total, flush=True)
        assert partial.stat().st_size == total
        partial.replace(path)
    receipt = {'url': url, 'bytes': path.stat().st_size,
               'sha256': digest(path)}
    path.with_suffix(path.suffix + '.json').write_text(
        json.dumps(receipt, indent=2) + '\n')
    print(path.name, receipt['bytes'], receipt['sha256'], flush=True)
    return path


def main() -> None:
    """Fetch pinned public inputs; keep large artifacts outside Git history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('item', choices=['checkpoint', 'transcoder',
                                        'pets', 'dtd'])
    item = parser.parse_args().item
    if item == 'checkpoint':
        for name in ['config.json', 'README.md', 'ra_sae.py',
                     'RA-SAE-DINOv2-32k.pth']:
            fetch(f'https://huggingface.co/{HF_ID}/resolve/{HF_REV}/{name}',
                  ROOT / 'checkpoints/pretrained' / name)
    elif item == 'transcoder':
        repo = ('Prisma-Multimodal/'
                'CLIP-transcoder-topk-256-x64-all_patches_1-mlp-94')
        revision = '96c293a7299a99fe3d70f1f15498215849f22d96'
        for name in ['config.json', 'weights.pt']:
            fetch(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',
                  ROOT / 'checkpoints/transcoder' / name)
    elif item == 'pets':
        from lattmc.vision.overcomplete_pets_codexgen import main
        main()
    else:
        url = ('https://www.robots.ox.ac.uk/~vgg/data/dtd/download/'
               'dtd-r1.0.1.tar.gz')
        fetch(url, ROOT / 'downloads/dtd-r1.0.1.tar.gz')


if __name__ == '__main__':
    main()
