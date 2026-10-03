"""Create and execute the complete cached-corpus benchmark notebook."""

from __future__ import annotations

from .paths_codexgen import (
    PAPER, repository, CACHE, RELEASE, PROTOCOL, NOTEBOOKS, TEMPLATES,
    ORGANIZATION,
)

import json
import os
from pathlib import Path
import tempfile
import zipfile

import nbformat
from nbclient import NotebookClient

HERE = PAPER
ROOT = repository()


def chunks(value: str | list[str]) -> list[str]:
    """Split notebook text into serialization-sized chunks."""
    if isinstance(value, list):
        value = ''.join(value)
    result, chunk = [], ''
    for char in value:
        if len(json.dumps(chunk + char, ensure_ascii=False)) > 64:
            result.append(chunk)
            chunk = ''
        chunk += char
    if chunk:
        result.append(chunk)
    return result


def main() -> None:
    """Create and execute the complete cached-corpus benchmark notebook."""
    cells = [nbformat.v4.new_markdown_cell(
        '# Held-out cached-activation retrieval\n\n'
        'This notebook executes CONFERENCE_PROTOCOL.md in full: 32 literal '
        'phrase tasks, five source draws, six model/layer pairs, and seven '
        'retrieval comparisons. It then runs the explicitly post hoc '
        'fixed-budget ablation. No model or dataset is downloaded.\n\n'
        'These labels test lexical retrieval, not semantic interpretability. '
        'Use the repository uv environment. Large input caches must already '
        'exist at the paths checked by the experiment implementation.'
    ), nbformat.v4.new_code_cell(
        "from pathlib import Path\n"
        "import subprocess\n"
        "import os\n"
        "import sys\n"
        "project = next(\n"
        "    p for p in [Path.cwd(), *Path.cwd().parents]\n"
        "    if (p / 'pyproject.toml').exists()\n"
        ")\n"
        "os.environ['PYTHONPATH'] = str(project / 'src')\n"
        "assert Path(sys.executable).parent == project / '.venv/bin'\n"
        "def execute(script, *args):\n"
        "    module = 'lattmc.lattconferencetools.'\n"
        "    module += script.removesuffix('.py')\n"
        "    subprocess.run([sys.executable, '-m', module, *args],\n"
        "                   cwd=project, check=True)\n"
        "execute('conference_experiments_codexgen.py', 'prepare')\n"
    )]
    for kind in ('sae', 'tc'):
        for layer in (0, 8, 11):
            cells.append(nbformat.v4.new_code_cell(
                "execute('conference_experiments_codexgen.py', 'run',\n"
                f"        '--kind', {kind!r}, '--layer', {str(layer)!r})\n"
            ))
    cells.append(nbformat.v4.new_markdown_cell(
        '## Post hoc ablation and report\n\n'
        'The fixed-budget ablation was added after the first primary result '
        'inspection. It leaves the primary comparison and selection rule '
        'unchanged. The report includes negative results and all layers.'
    ))
    cells.append(nbformat.v4.new_code_cell(
        "execute('conference_ablation_codexgen.py')\n"
        "execute('summarize_conference_codexgen.py')\n"
    ))
    notebook = nbformat.v4.new_notebook(cells=cells)
    runtime = Path(tempfile.mkdtemp(prefix='conference-kernel-'))
    kernel = runtime / 'kernels' / 'conference-project'
    kernel.mkdir(parents=True)
    (kernel / 'kernel.json').write_text(json.dumps(dict(
        argv=[str(ROOT / '.venv/bin/python'), '-m', 'ipykernel_launcher',
              '-f', '{connection_file}'],
        display_name='Project uv Python', language='python',
    )))
    os.environ['JUPYTER_PATH'] = str(runtime)
    client = NotebookClient(
        notebook, timeout=600, kernel_name='conference-project',
        resources={'metadata': {'path': str(HERE)}},
        on_cell_start=lambda cell, cell_index, **kw:
            print('Executing cell', cell_index, flush=True),
    )
    target = (NOTEBOOKS /
              'conference_retrieval_codexgen.ipynb')
    try:
        client.execute()
    finally:
        nbformat.validate(notebook)
        data = json.loads(nbformat.writes(notebook))
        data['metadata']['kernelspec'] = dict(
            display_name='Python (project uv)', name='python3',
            language='python',
        )
        for cell in data['cells']:
            cell['source'] = chunks(cell['source'])
            for output in cell.get('outputs', []):
                if 'text' in output:
                    output['text'] = chunks(output['text'])
        target.write_text(json.dumps(data, indent=1) + '\n')
    RELEASE.mkdir(parents=True, exist_ok=True)
    archive = RELEASE / 'conference_retrieval_results.zip'
    with zipfile.ZipFile(archive, 'a', zipfile.ZIP_DEFLATED) as stream:
        stream.write(target, target.relative_to(ROOT))
        name = str(Path(__file__).relative_to(ROOT))
        if name not in stream.namelist():
            stream.write(__file__, name)
    print('Executed notebook saved:', target, flush=True)


if __name__ == '__main__':
    main()
