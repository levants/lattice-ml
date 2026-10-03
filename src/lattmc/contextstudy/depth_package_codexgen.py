"""Package context-depth results and mirror source code without TeX."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile

from lattmc.activationstudy.common_codexgen import sha256, save_json
from .depth_codexgen import ROOT, OUT, OLD, PROTOCOL


def main(mirror: Path | None = None) -> None:
    """Package the experiment artifacts and optionally copy them to a mirror.
    """
    source = ROOT / 'src/lattmc'
    docs = ROOT / 'experiments/activation_studies'
    files = []
    for package in ('activationstudy', 'contextstudy'):
        files += [p for p in (source / package).iterdir()
                  if p.suffix in ('.py', '.md', '.json')]
    files += list((docs / 'contextdepth_v1').glob('*'))
    files += [docs / 'package_migration.json']
    notebook = ROOT / 'notebooks/sae/context_depth_codexgen.ipynb'
    files += [notebook]
    files += [ROOT / 'src/lattmc/latex/texttables_codexgen.py']
    files += list((ROOT / 'data/activation_studies/witnesses_v1').glob('*'))
    files += list((ROOT / 'experiments/activation_studies/witnesses_v1')
                  .glob('*.md'))
    files = [p for p in files if p.is_file() and p.name != 'provenance.json']
    # Verify depth extraction and evaluation provenance, including old inputs.
    verified = 0
    code_hash = sha256(source / 'contextstudy/depth_codexgen.py')
    for dataset in ('ag_news', 'dbpedia_14'):
        for layer in (3, 15, 27):
            path = OUT / dataset / f'smol{layer}_results.json'
            report = json.loads(path.read_text())
            assert report['source_sha256'] == code_hash
            assert report['protocol_sha256'] == sha256(PROTOCOL)
            feature = (OLD / dataset / 'smol_topk_max.npz' if layer == 15
                       else OUT / dataset / f'smol{layer}_max.npz')
            assert report['feature_sha256'] == sha256(feature)
            if layer != 15:
                path = OUT / dataset / f'smol{layer}_extraction.json'
                extraction = json.loads(path.read_text())
                assert extraction['source_sha256'] == code_hash
                assert extraction['protocol_sha256'] == sha256(PROTOCOL)
                assert extraction['design_sha256'] == sha256(
                    OLD / dataset / 'design.json')
                assert extraction['tokens_sha256'] == sha256(
                    OLD / dataset / 'smol_topk_tokens.local.npz')
                for name, digest in extraction['files'].items():
                    assert sha256(OUT / dataset / name) == digest
            verified += 1
    provenance = dict(
        source={str(p.relative_to(ROOT)): sha256(p) for p in files},
        depth_provenance_checks=verified,
        inference='Offline MPS inference; public pinned checkpoints.',
        notebook='Executed saved-array audit and complete gallery display.',
        dependency='Original families_v2 corpus, token and activation cache.',
        publication='Local release candidate only; not uploaded.',
    )
    if mirror:
        mirror = Path(mirror).resolve()
        copy_files = [p for p in files
                      if not p.is_relative_to(ROOT / 'data')]
        notebooks = (ROOT / 'notebooks').rglob('*_codexgen.ipynb')
        copy_files += [p for p in notebooks
                       if 'activationstudy' in p.read_text() or p == notebook]
        copy_files += [p for p in docs.rglob('*')
                       if p.is_file() and p.suffix in ('.md', '.json')]
        copy_files = sorted(set(copy_files))
        for path in copy_files:
            dest = mirror / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            assert sha256(path) == sha256(dest)
        for name in ('contextdepth_v1', 'families_v2'):
            dest = mirror / 'data/activation_studies' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            target = ROOT / 'data/activation_studies' / name
            if dest.exists() or dest.is_symlink():
                assert dest.resolve() == target.resolve(), dest
            else:
                dest.symlink_to(os.path.relpath(target, dest.parent))
        provenance['mirror_files_verified'] = len(copy_files)
    pp = docs / 'contextdepth_v1/provenance.json'
    save_json(pp, provenance)
    if mirror:
        shutil.copy2(pp, mirror / pp.relative_to(ROOT))
    files += [pp] + [p for p in OUT.rglob('*') if p.is_file()]
    files = sorted(set(files))
    release = ROOT / 'artifacts/releases/activation_studies/contextdepth_v1'
    release.mkdir(parents=True, exist_ok=True)
    target = release / 'context-depth-results.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(ROOT)))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert not any(p.endswith(('.tex', '.bib', '.pt', '.safetensors'))
                       for p in archive.namelist())
        for path in files:
            assert archive.read(str(path.relative_to(ROOT))) == (
                path.read_bytes())
    save_json(release / 'manifest.json', dict(
        archive=target.name, bytes=target.stat().st_size,
        sha256=sha256(target),
        members={str(p.relative_to(ROOT)): sha256(p) for p in files},
        verification='Every member byte-verified; ZIP CRC passed.'))
    print('Verified', len(files), 'members:', target)
    print('Depth provenance checks:', verified)
    if mirror:
        print('Code/notebook/doc mirror:', provenance['mirror_files_verified'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirror', type=Path)
    main(parser.parse_args().mirror)
