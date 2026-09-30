"""Mirror only patch experiment code and evidence; exclude paper files."""

import argparse
import json
import shutil
from pathlib import Path

from lattmc.vision.patch_weights_codexgen import digest
from lattmc.vision.paths_codexgen import repository_root


ROOT = repository_root()
PATCH = ROOT / 'vision_tokens/patch_contexts'
FORBIDDEN = {'.tex', '.bib', '.bbl', '.pdf'}


def files():
    selected = list((ROOT / 'src/lattmc/vision').glob('patch*_codexgen.py'))
    selected += [ROOT / 'notebooks/vision/patch_contexts_codexgen.ipynb']
    selected += [ROOT / 'vision_tokens/README.md',
                 PATCH / 'README.md', PATCH / '.gitignore']
    for name in ['prisma', 'saev', 'environment']:
        selected.extend(p for p in (PATCH / name).rglob('*') if p.is_file())
    for name in ['prisma_backbone', 'prisma_sae', 'saev_register_backbone',
                 'saev_sae']:
        for path in (PATCH / 'checkpoints' / name).iterdir():
            if path.is_file() and path.suffix not in {'.pt', '.safetensors'}:
                selected.append(path)
    upstream = PATCH / 'upstream'
    selected.extend(p for p in upstream.iterdir()
                    if p.suffix in {'.json', '.py'})
    allowed = {'.py', '.typed', '.html', '.js', '.css', '.elm',
               '.json', '.yaml', '.yml', '.md'}
    for name in ['prisma', 'saev']:
        package = upstream / name
        for path in (package / 'src').rglob('*'):
            if (path.is_file() and path.suffix in allowed and
                    '__pycache__' not in path.parts and
                    not any(x.endswith('.egg-info') for x in path.parts)):
                selected.append(path)
        selected.append(package / 'LICENSE')
        if name == 'prisma':
            selected.extend([package / 'setup.py', package / 'docs/README.md'])
        else:
            selected.extend([package / 'pyproject.toml',
                             package / 'README.md'])
    return sorted(set(selected))


def manifest():
    entries = []
    for path in files():
        relative = path.relative_to(ROOT)
        assert relative.parts[0] in {'src', 'notebooks', 'vision_tokens'}
        assert path.suffix not in FORBIDDEN
        assert path.stat().st_size < 100 * 1024 * 1024, relative
        entries.append({'path': relative.as_posix(),
                        'bytes': path.stat().st_size,
                        'sha256': digest(path)})
    result = {'scope': 'patch code, notebook, upstream source, and evidence',
              'excludes': ['paper', 'tex', 'pdf', 'unused checkpoints'],
              'files': entries}
    (PATCH / 'release_manifest_codexgen.json').write_text(
        json.dumps(result, indent=2) + '\n')
    return result


def mirror(target):
    target = Path(target).resolve()
    if target == ROOT:
        raise ValueError('Mirror destination must be another checkout')
    result = manifest()
    paths = [ROOT / row['path'] for row in result['files']]
    paths.append(PATCH / 'release_manifest_codexgen.json')
    for source in paths:
        destination = target / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assert digest(source) == digest(destination)
    print(f'Verified {len(paths)} source/evidence files; no paper files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirror', type=Path)
    args = parser.parse_args()
    if args.mirror:
        mirror(args.mirror)
    else:
        print(len(manifest()['files']), 'verified release entries')
