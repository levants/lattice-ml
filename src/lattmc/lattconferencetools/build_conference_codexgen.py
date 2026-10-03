"""Restore unmodified official template assets and build the ICLR draft."""

from __future__ import annotations

from .paths_codexgen import PAPER, TEMPLATES

import re
import subprocess
import zipfile

HERE = PAPER


def main() -> None:
    """Restore unmodified official template assets and build the ICLR draft."""
    (HERE / 'packages').mkdir(exist_ok=True)
    with zipfile.ZipFile(TEMPLATES / 'iclr2027-official.zip') as archive:
        for name in ('iclr2027_conference.sty', 'iclr2027_conference.bst'):
            target = HERE / 'packages' / name
            target.write_bytes(archive.read('iclr2027/' + name))
    with zipfile.ZipFile(TEMPLATES / 'conference-fonts.zip') as archive:
        for name in archive.namelist():
            path = HERE / 'templates' / name
            if not path.resolve().is_relative_to(HERE / 'templates'):
                raise ValueError('Unexpected font archive path.')
            if not name.endswith('/'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
    # BibTeX lacks BibLaTeX's software/online types and eprint handling.
    entries = re.split(r'(?=^@)', (HERE / 'references.bib').read_text(),
                       flags=re.MULTILINE)
    compatible = []
    for entry in entries:
        entry = re.sub(r'^@(software|online)\{', '@misc{', entry)
        eprint = re.search(r'eprint\s*=\s*\{([^}]+)\}', entry)
        if eprint:
            if not re.search(r'\bjournal\s*=', entry):
                entry = re.sub(r'^@article\{', '@misc{', entry)
            if not re.search(r'\burl\s*=', entry):
                ending = entry.rfind('}')
                prefix = entry[:ending].rstrip().rstrip(',')
                entry = (prefix + ',\n  url = {https://arxiv.org/abs/'
                         + eprint.group(1) + '}\n}\n\n')
        if entry.startswith('@misc{') and not re.search(
                r'\byear\s*=', entry):
            ending = entry.rfind('}')
            prefix = entry[:ending].rstrip().rstrip(',')
            fields = ',\n  year = {n.d.}'
            accessed = re.search(r'urldate\s*=\s*\{([^}]+)\}', entry)
            if accessed and not re.search(r'\bnote\s*=', entry):
                fields += ',\n  note = {Accessed ' + accessed.group(1) + '}'
            entry = prefix + fields + '\n}\n\n'
        compatible.append(entry)
    (HERE / 'conference_references.bib').write_text(
        '% Generated from references.bib for the official BibTeX style.\n'
        + ''.join(compatible).rstrip() + '\n')
    subprocess.run([
        'latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
        '-file-line-error', 'conference-iclr.tex',
    ], cwd=HERE, check=True)


if __name__ == '__main__':
    main()
