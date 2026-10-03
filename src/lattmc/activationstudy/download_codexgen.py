"""Download the pinned source parquets, checking every recorded checksum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from .common_codexgen import save_json, sha256


def download(output: Path) -> None:
    """Download the registered inputs and save provenance receipts."""
    sources = json.loads(Path(__file__).with_name(
        'dataset_sources.json').read_text())
    for name, manifest in sources.items():
        folder = output / 'raw' / name
        folder.mkdir(parents=True, exist_ok=True)
        for entry in manifest['files']:
            target = folder / entry['local']
            if not target.exists():
                url = ('https://huggingface.co/datasets/' + manifest['repo']
                       + '/resolve/' + manifest['revision'] + '/'
                       + entry['file'])
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                temporary = target.with_suffix('.partial')
                with temporary.open('wb') as stream:
                    for block in response.iter_content(2 ** 20):
                        stream.write(block)
                assert sha256(temporary) == entry['sha256']
                temporary.replace(target)
            assert sha256(target) == entry['sha256']
            print(name, target.name, 'verified', flush=True)
        save_json(folder / 'source.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    download(args.output)
