"""Package contextual evidence and optionally mirror code, never TeX files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile

from .audit_codexgen import DATA, ROOT, digest


def main(mirror: Path | None = None) -> None:
    """Package the experiment artifacts and optionally copy them to a mirror.
    """
    package = ROOT / 'src/lattmc/contextstudy'
    protocol = ROOT / 'experiments/activation_studies/context_v1'
    notebook = ROOT / 'notebooks/sae/contextual_witnesses_codexgen.ipynb'
    files = sorted(package.glob('*.py')) + [notebook]
    files += [ROOT / 'src/lattmc/latex/texttables_codexgen.py']
    files += list((ROOT / 'data/activation_studies/witnesses_v1').glob('*'))
    files += list((ROOT / 'experiments/activation_studies/witnesses_v1')
                  .glob('*.md'))
    files += [protocol / 'PROTOCOL.md', protocol / 'README.md']
    files += [protocol / name for name in ('build.json', 'verification.json')
              if (protocol / name).exists()]
    dependencies = [ROOT / 'src/lattmc' / p for p in (
        'tc/transcoder_analyzers_codexgen.py', 'tc/transcoder_utils.py',
        'tc/sparse_surrogates.py', 'tc/tokenization_utils.py',
        'sae/sae_utils.py', 'sae/nlp_sae_utils.py',
        'fca/visualization_utils.py',
    )]
    provenance = dict(
        source={str(p.relative_to(ROOT)): digest(p) for p in files},
        existing_dependencies={str(p.relative_to(ROOT)): digest(p)
                               for p in dependencies},
        inference='Executed offline by run_codexgen.py, float32 CPU.',
        notebook='Executed saved-array audit and display; no inference.',
        publication='Local archive only; not uploaded.',
    )
    if mirror:
        mirror = Path(mirror).resolve()
        for p in files:
            if p.is_relative_to(ROOT / 'data'):
                continue
            dest = mirror / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            assert digest(p) == digest(dest)
        for p in dependencies:
            assert digest(p) == digest(mirror / p.relative_to(ROOT))
        for name in ('context_v1', 'legacy'):
            dest = mirror / 'data/activation_studies' / name
            source = ROOT / 'data/activation_studies' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.is_symlink():
                assert dest.resolve() == source.resolve(), dest
            else:
                dest.symlink_to(os.path.relpath(source, dest.parent))
        provenance['mirror'] = dict(
            files_verified=sum(not p.is_relative_to(ROOT / 'data')
                               for p in files),
            dependencies_verified=len(dependencies),
            code_only=True,
        )
    (protocol / 'provenance.json').write_text(
        json.dumps(provenance, indent=2) + '\n',
    )
    if mirror:
        shutil.copy2(protocol / 'provenance.json',
                     mirror / (protocol / 'provenance.json').relative_to(ROOT))
    release = ROOT / 'artifacts/releases/activation_studies/context_v1'
    release.mkdir(parents=True, exist_ok=True)
    files += [protocol / 'provenance.json']
    files += sorted(DATA.glob('*layer*.json'))
    files += sorted(DATA.glob('*layer*.npz')) + [DATA / 'audit.json']
    target = release / 'contextual-witnesses-results.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert not any(p.endswith(('.tex', '.bib', '.pt', '.safetensors'))
                       for p in archive.namelist())
        for p in files:
            assert archive.read(str(p.relative_to(ROOT))) == p.read_bytes()
    manifest = dict(
        archive=target.name, bytes=target.stat().st_size,
        sha256=digest(target), member_count=len(files),
        members={str(p.relative_to(ROOT)): digest(p) for p in files},
        verification='CRC and every member byte comparison passed.',
    )
    (release / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n',
    )
    print('Verified archive:', target, 'members:', len(files))
    if mirror:
        print('Verified code/notebook mirror:', mirror, provenance['mirror'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirror', type=Path)
    main(parser.parse_args().mirror)
