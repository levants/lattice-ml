"""Create and execute the bonds notebook in its repository location."""

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import os

from .paths_codexgen import NOTEBOOK, ROOT
nb=nbformat.v4.new_notebook(cells=[
 nbformat.v4.new_markdown_cell('''# Exact minimal bonds across surrogate layers

Uses the existing project uv environment and unmodified local activation
caches. The original notebooks remain unchanged. This notebook computes
1,584 exact implicit one-pair bonds, reproduces two historical examples,
and runs the independent finite-lattice validation. It does not rerun
neural inference or validate inherited token colors.
'''),
 nbformat.v4.new_code_cell('''from pathlib import Path
import json
import runpy
import sys

root = next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / 'pyproject.toml').is_file()
)
sys.path.insert(0, str(root))
from src.lattmc.bonds.paths_codexgen import OUT, PAPER'''),
 nbformat.v4.new_code_cell('''
from src.lattmc.bonds.bond_experiments_codexgen import run
run()'''),
 nbformat.v4.new_code_cell('''_ = runpy.run_module(
    'src.lattmc.bonds.summarize_bonds_codexgen'
)
summary_path = OUT / 'summary.json'
report = json.loads(summary_path.read_text())
report'''),
 nbformat.v4.new_code_cell('''import unittest
from src.lattmc.bonds.test_bonds_codexgen import BondChecks

suite = unittest.defaultTestLoader.loadTestsFromTestCase(BondChecks)
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()'''),
 nbformat.v4.new_markdown_cell('''
The four compressed components per bond are specified in
`data/results.json`; their arrays are in the named NPZ files
under `data/`. Every membership
query is the disjunction of the three rectangle tests in the paper.
The results describe these cached contexts, not population-level semantic
or causal performance. See `texs/crossbonds/surrogates/REVIEW.md`
for limitations.
''')],metadata=dict(kernelspec=dict(display_name='Python 3',
                                   language='python',name='python3')))
previous_directory = os.getcwd()
try:
    os.chdir(ROOT)
    shell = InteractiveShell.instance()
    for count, cell in enumerate(
            [cell for cell in nb.cells if cell.cell_type == 'code'], 1):
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(captured.stdout + captured.stderr)
        cell.execution_count = count
        cell.outputs = []
        for name, text in [('stdout', captured.stdout),
                           ('stderr', captured.stderr)]:
            if text:
                cell.outputs.append(nbformat.v4.new_output(
                    'stream', name=name, text=text
                ))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output(
                'display_data', data=output.data, metadata=output.metadata
            ))
finally:
    os.chdir(previous_directory)
nb.metadata['execution_method'] = 'Sequential in-process IPython'
nbformat.write(nb, NOTEBOOK)
print('Executed notebook successfully')
