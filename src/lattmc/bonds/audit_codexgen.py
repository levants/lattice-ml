"""Audit all inputs, labels, citation keys, and TeX conventions."""

from __future__ import annotations
from pathlib import Path

from collections import Counter
import json
import re

from .paths_codexgen import OUT, PACKAGE, PAPER, ROOT
seen = []


def expand(path: Path) -> str:
    """Recursively expand the manuscript LaTeX input and include commands."""
    assert path not in seen, f'Repeated input: {path}'
    seen.append(path)
    source = path.read_text()
    return re.sub(
        r'\\(?:input|include)\{([^}]+)\}',
        lambda m: expand(PAPER / (m.group(1) + '.tex')), source,
    )


source = expand(PAPER / 'surrogates.tex')
labels = re.findall(r'\\label\{([^}]+)\}', source)
duplicates = [key for key, count in Counter(labels).items() if count > 1]
assert not duplicates, duplicates
refs = re.findall(r'\\(?:eqref|ref|pageref)\{([^}]+)\}', source)
assert not set(refs)-set(labels), set(refs)-set(labels)
cites = {
    key.strip() for keys in re.findall(r'\\cite\w*\{([^}]+)\}', source)
    for key in keys.split(',')
}
bib = (PAPER / 'references.bib').read_text()
keys = re.findall(r'^@\w+\{([^,]+),', bib, re.M)
assert len(keys) == len(set(keys))
assert not cites-set(keys), cites-set(keys)
assert not set(keys)-cites, set(keys)-cites

for heading in re.finditer(
    r'\\(?:section|subsection|subsubsection)\{[^}]+\}', source
):
    rest = source[heading.end():]
    assert re.match(r'\s*\\label\{', rest), heading.group()

counts = Counter()
for kind in ('equation', 'align', 'table', 'algorithm', 'theorem', 'lemma',
             'proposition', 'corollary'):
    pattern = r'\\begin\{' + kind + r'\}(.*?)\\end\{' + kind + r'\}'
    for match in re.finditer(pattern, source, re.S):
        body = match.group(1)
        assert '\\label{' in body, (kind, body[:120])
        if kind in ('theorem', 'lemma', 'proposition', 'corollary'):
            assert body.lstrip().startswith('['), body[:80]
        counts[kind] += 1

assert not re.search(r'\\[\[\]()]', source)
assert not re.search(r'\\begin\{(?:equation|align)\*\}', source)
for math in re.findall(
    r'(?<!\\)\$[^$]*?(?<!\\)\$|'
    r'\\begin\{equation\}.*?\\end\{equation\}', source, re.S
):
    assert not re.search(r'(?<!\\)[_^](?!\{)', math), math

width_files = seen + [PAPER / 'references.bib']
width_files += list(PACKAGE.glob('*.py'))
too_wide = []
for path in width_files:
    for line, text in enumerate(path.read_text().splitlines(), 1):
        if len(text) > 79:
            too_wide.append((str(path.relative_to(ROOT)), line, len(text)))
assert not too_wide, too_wide
report = dict(
    files=[str(path.relative_to(PAPER)) for path in seen],
    labels=len(labels), citations=len(cites), environments=dict(counts),
    references_resolve=True, maximum_columns=79,
)
(OUT / 'source_audit.json').write_text(
    json.dumps(report, indent=2)+'\n'
)
print(json.dumps(report, indent=2))
