"""Package the full article and verify an isolated three-pass TeX build."""

from __future__ import annotations

from .paths_codexgen import (
    PAPER, repository, CACHE, RELEASE, PROTOCOL, NOTEBOOKS, TEMPLATES,
    ORGANIZATION,
)

from pathlib import Path
import subprocess
import tempfile
import zipfile

HERE = PAPER


def main() -> None:
    """Package the full article and verify an isolated three-pass TeX build."""
    paths = {HERE / 'lattconference.tex', HERE / 'references.bib',
             HERE / 'lattconference.bbl'}
    for line in (HERE / 'lattconference.fls').read_text().splitlines():
        if not line.startswith('INPUT '):
            continue
        path = Path(line[6:])
        if not path.is_absolute():
            path = HERE / path
        path = path.resolve()
        if path.is_relative_to(HERE) and path.suffix in (
                '.tex', '.bib', '.bbl', '.png', '.jpg', '.pdf', '.cls'):
            paths.add(path)
    folder = repository() / 'artifacts/papers/lattconference'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'lattconference_arxiv_source.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(HERE))
        archive.writestr('ARXIV_README.txt',
            'Compile lattconference.tex with pdfLaTeX.\n'
            'The source includes its BibLaTeX/Biber-generated .bbl file.\n'
            'References precede the appendix; combined mode is selected.\n'
            'This bundle was verified locally, not on the arXiv server.\n'
            'The artifact URL remains a declared placeholder.\n')
    directory = Path(tempfile.mkdtemp(prefix='lattconference-arxiv-check-'))
    with zipfile.ZipFile(target) as archive:
        archive.extractall(directory)
    for turn in range(3):
        with (directory / f'pass-{turn}.txt').open('w') as log:
            subprocess.run([
                'pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                '-file-line-error', 'lattconference.tex',
            ], cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                check=True)
    log = (directory / 'lattconference.log').read_text()
    bad = ('undefined', 'multiply defined', 'Overfull',
           'Please (re)run Biber', 'Rerun to get cross-references right')
    assert not any(word in log for word in bad)
    print('PASS: isolated source bundle, supplied bibliography, three passes')
    print('Bundle:', target)
    print('Verification directory:', directory)


if __name__ == '__main__':
    main()
