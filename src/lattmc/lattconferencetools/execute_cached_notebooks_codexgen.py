"""Execute an explicitly bounded excerpt of a companion notebook offline."""

from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nbformat import NotebookNode

from .paths_codexgen import (
    PAPER, repository, CACHE, RELEASE, PROTOCOL, NOTEBOOKS, TEMPLATES,
    ORGANIZATION,
)

import argparse
import json
import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = repository()
parser = argparse.ArgumentParser()
parser.add_argument('kind', choices=['sae', 'tc'])
parser.add_argument('layer', type=int, choices=[0, 8, 11])
parser.add_argument(
    '--output', type=Path,
    default=CACHE / 'cached_notebook_checks',
)
args = parser.parse_args()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
folder = 'sae' if args.kind == 'sae' else 'transcoders'
stem = f'{args.kind}_gpt_small_tokens_places_min_acts'
source = ROOT / 'notebooks' / folder / f'{stem}_codexgen.ipynb'
original = nbformat.read(source, as_version=4)
cells = [nbformat.v4.new_markdown_cell(
    '# Offline cached-data execution check\n\n'
    f'Source: `{source.name}`. Layer: {args.layer}.\n\n'
    'The initialization and source-prompt cells below come from the '
    'companion notebook. This bounded run uses CPU, one layer, all '
    '25,600 cached sequences for retrieval, and at most two retrieved '
    'sequences for token inspection. It does not reproduce every '
    'exploratory cell or full-extent token diagnostic. Cached files '
    'are read without downloading or regenerating activations.'
)]
cells.append(nbformat.v4.new_code_cell(
    "import os\n"
    "from pathlib import Path\n"
    "project = next(\n"
    "    path for path in [Path.cwd(), *Path.cwd().parents]\n"
    "    if (path / 'pyproject.toml').is_file()\n"
    ")\n"
    f"os.chdir(project / 'notebooks' / {folder!r})\n"
    "os.environ['HF_HUB_OFFLINE'] = '1'\n"
    "os.environ['TRANSFORMERS_OFFLINE'] = '1'\n"
    "import torch\n"
    "torch.set_num_threads(4)\n"
    "torch.set_grad_enabled(False)\n"
))
for i, cell in enumerate(original.cells[:52]):
    if cell.cell_type != 'code' or not cell.source.strip():
        continue
    cell.outputs = []
    cell.execution_count = None
    cell.metadata['original_cell_index'] = i
    if i == 1:
        cell.source = '%matplotlib inline'
    elif i == 24:
        cell.source = "device = torch.device('cpu')\ndevice"
    elif i == 35:
        cell.source = cell.source.replace(
            'layers = [0, 4, 6, 8, 10, 11]', f'layers = [{args.layer}]',
        )
    elif 'tr_analyzer.fcas[0].V.shape' in cell.source:
        cell.source = '{i: f.V.shape for i, f in tr_analyzer.fcas.items()}'
    cells.append(cell)
cells.append(nbformat.v4.new_code_cell(f'layer = {args.layer}'))
cells.append(nbformat.v4.new_code_cell('''
import json
import importlib.metadata as metadata

versions = {
    name: metadata.version(name)
    for name in ['sae-lens', 'transformer-lens', 'transformers', 'torch']
}
print(versions)
print('tokens:', tuple(tr_analyzer.tokens.shape))
print('matrix:', tr_analyzer.fcas[layer].V.shape)
rows = [3457, 5411, 117, 21481, 1924, 4042]
checks = []
positions = {
    3457: [1, 2, 3], 5411: [15, 16, 17], 117: [30, 31],
    21481: [24, 25, 26], 1924: [15, 39, 94], 4042: [8, 82],
}
for row in rows:
    fresh = tr_analyzer.tr_utils.run_transcoders(
        tr_analyzer.corpus[row], [layer],
    )[layer]
    pooled = fresh.max(axis=0)
    cached = tr_analyzer.fcas[layer].V[row]
    changed = np.flatnonzero((pooled > 0) != (cached > 0))
    result = {
        'row': row,
        'max_abs_error': float(np.max(np.abs(pooled - cached))),
        'close': bool(np.allclose(pooled, cached, atol=1e-3, rtol=1e-3)),
        'support_disagreements': int(np.sum((pooled > 0) != (cached > 0))),
        'support_changes': [
            dict(feature=int(i), fresh=float(pooled[i]),
                 cached=float(cached[i])) for i in changed
        ],
    }
    checks.append(result)
    print(result)
assert all(item['close'] for item in checks), 'Activation cache mismatch'
'''.strip()))
cells.append(nbformat.v4.new_code_cell('''
# Validate all cached bounds and support floors without a second dense copy.
fca = tr_analyzer.fcas[layer]
matrix = fca.V
lower = np.full(matrix.shape[1], np.inf, dtype=matrix.dtype)
upper = np.zeros(matrix.shape[1], dtype=matrix.dtype)
floor = lower.copy()
for start in range(0, len(matrix), 256):
    block = matrix[start:start + 256]
    assert np.all(np.isfinite(block)) and np.all(block >= 0)
    lower = np.minimum(lower, block.min(axis=0))
    upper = np.maximum(upper, block.max(axis=0))
    positive_min = np.min(block, axis=0, where=block > 0, initial=np.inf)
    floor = np.minimum(floor, positive_min)
floor[np.isinf(floor)] = 0
assert np.array_equal(lower, fca.v_min.ravel())
assert np.array_equal(upper, fca.v_max.ravel())
assert np.array_equal(floor, fca.v_min_nonzeros.ravel())
print('All cached minima, maxima, and positive support floors: PASS')
'''.strip()))
cells.append(nbformat.v4.new_code_cell('''
# Exercise the source positions for every empirical example.
queries = []
for row in rows:
    analysis = ConceptAnalysis(tr_analyzer.corpus[row], tr_analyzer)
    analysis.analyze_concepts()
    analysis.gen_text(
        positions[row], layer, limit=2, min_vals=True, full_tokens=False,
    )
    row_extent = analysis.c_is[layer].A
    query = np.maximum.reduce(list(analysis.v_is[layer].values()))
    closed = fca.F(row_extent)
    assert row in row_extent
    assert np.all(closed >= query)
    assert np.array_equal(fca.G(closed), row_extent)
    queries.append(dict(row=row, positions=positions[row],
                        extent_size=len(row_extent)))
print(queries)
'''.strip()))
cells.append(nbformat.v4.new_code_cell('''
# Same support-floor NYC query as the source notebook, bounded inspection.
dets, vs = concept_an.gen_text(
    [1, 2, 3], layer, limit=2, min_vals=True, full_tokens=False,
)
extent = concept_an.c_is[layer].A
print('NYC joint extent:', len(extent))
print('First row indices:', extent[:10].tolist())
assert 3457 in extent, 'The source sequence must satisfy its floor probes.'
query = np.maximum.reduce(list(concept_an.v_is[layer].values()))
fca = tr_analyzer.fcas[layer]
closed = fca.F(extent)
assert np.array_equal(fca.G(closed), extent)
assert np.all(closed >= query)
print('Sequence-level closure and extent invariance: PASS')
'''.strip()))
report_path = OUT / f'{args.kind}_layer{args.layer}.json'
try:
    report_relative = report_path.relative_to(ROOT)
    path_code = (
        "report_path = project / " + repr(str(report_relative)) + "\n"
    )
except ValueError:
    path_code = "report_path = Path(" + repr(str(report_path)) + ")\n"
path_code += "report_path.parent.mkdir(parents=True, exist_ok=True)\n"
cells.append(nbformat.v4.new_code_cell(
    path_code +
    "report = dict(versions=versions, checks=checks, queries=queries,\n"
    "              layer=layer, cached_bounds_verified=True,\n"
    "              tokens_shape=list(tr_analyzer.tokens.shape),\n"
    "              matrix_shape=list(fca.V.shape),\n"
    "              nyc_extent=len(extent), offline=True,\n"
    "              full_notebook_execution=False)\n"
    "report_path.write_text(json.dumps(report, indent=2))\n"
    "report\n"
))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Project uv Python', 'language': 'python',
    'name': 'lattcontexts-codexgen',
}
runtime = Path(tempfile.mkdtemp(prefix='lattcontexts-jupyter-'))
kernel = runtime / 'kernels' / 'lattcontexts-codexgen'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [str(ROOT / '.venv/bin/python'), '-m', 'ipykernel_launcher',
             '-f', '{connection_file}'],
    'display_name': 'Project uv Python', 'language': 'python',
}))
os.environ['JUPYTER_PATH'] = str(runtime)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')


def progress(cell: NotebookNode, cell_index: int, **kwargs: Any) -> None:
    """Print progress when notebook execution reaches a cell."""
    print(f'Cell {cell_index}: {cell.source.splitlines()[0][:70]}',
          flush=True)


def text_chunks(value: str | list[str]) -> list[str]:
    """Preserve notebook text while keeping serialized lines short."""
    if isinstance(value, list):
        value = ''.join(value)
    chunks = []
    chunk = ''
    for char in value:
        if len(json.dumps(chunk + char, ensure_ascii=False)) > 64:
            chunks.append(chunk)
            chunk = ''
        chunk += char
    if chunk:
        chunks.append(chunk)
    return chunks


def save_notebook(notebook: NotebookNode, path: Path) -> None:
    """Save schema-valid outputs and source with a 79-column limit."""
    nbformat.validate(notebook)
    data = json.loads(nbformat.writes(notebook))
    data['metadata']['kernelspec'] = {
        'display_name': 'Python (project uv)',
        'language': 'python', 'name': 'python3',
    }
    for cell in data['cells']:
        cell['source'] = text_chunks(cell['source'])
        for output in cell.get('outputs', []):
            if 'text' in output:
                output['text'] = text_chunks(output['text'])
            for mime, value in output.get('data', {}).items():
                if mime.startswith('text/'):
                    output['data'][mime] = text_chunks(value)
    serialized = json.dumps(data, indent=1, ensure_ascii=False) + '\n'
    path.write_text(serialized)


client = NotebookClient(
    nb, timeout=600, kernel_name='lattcontexts-codexgen',
    resources={'metadata': {'path': str(source.parent)}},
    on_cell_start=progress,
)
target = (ROOT / 'notebooks' / folder /
          f'{args.kind}_layer{args.layer}_cached_check_codexgen.ipynb')
try:
    client.execute()
finally:
    save_notebook(nb, target)
    print(f'Saved {target}', flush=True)
