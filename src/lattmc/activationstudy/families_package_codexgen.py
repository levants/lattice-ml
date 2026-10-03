"""Prepare local release assets without uploading text or model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from .common_codexgen import save_json, sha256


def package(
    output: Path,
    notebook: Path,
    protocol: Path,
    destination: Path,
) -> dict[str, int]:
    """Package reproducibility artifacts and report their byte counts."""
    destination.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).parent
    small, large = [], []
    for path in sorted(code.iterdir()):
        if path.is_file():
            small.append((path, 'src/lattmc/activationstudy/' + path.name))
    small.append((notebook, 'notebooks/sae/' + notebook.name))
    prefix = 'data/activation_studies/families_v2/'
    small.append((protocol,
                  'experiments/activation_studies/families_v2/PROTOCOL.md'))
    for path in sorted(protocol.parent.iterdir()):
        if path != protocol and path.suffix in ('.md', '.json'):
            small.append((path,
                'experiments/activation_studies/families_v2/' + path.name))
    for name in ('audit.json', 'code_sync.json', 'build_verification.json',
                 'execution_record.json', 'candidate_status.json',
                 'summary.json'):
        if (output / name).exists():
            small.append((output / name, prefix + name))
    for dataset in ('ag_news', 'dbpedia_14'):
        for path in sorted((output / dataset).iterdir()):
            name = path.name
            if (name in ('design.json', 'tfidf_vocabulary.json',
                         'tfidf_idf.npy') or name.endswith('_results.json')
                    or name.endswith('_extraction.json')
                    or name.endswith('_scores.npz')):
                small.append((path, prefix + dataset + '/' + name))
            elif name.endswith(('_max.npz', '_mean.npz', '_dense.npz')):
                large.append((path, prefix + dataset + '/' + name))
    manifests = {}
    for name, entries in [('saelens-families-results.zip', small),
                          ('saelens-families-activations.zip', large)]:
        target = destination / name
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for path, member in entries:
                assert '.local.' not in member
                archive.write(path, member)
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
        manifests[name] = dict(
            bytes=target.stat().st_size, sha256=sha256(target),
            contents={member: sha256(path) for path, member in entries})
    save_json(destination / 'release_manifest.json', manifests)
    return {name: item['bytes'] for name, item in manifests.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.notebook, args.protocol,
                             args.destination), indent=2))
