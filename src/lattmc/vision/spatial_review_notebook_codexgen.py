"""Execute a compact review notebook from the frozen spatial audit."""

from __future__ import annotations

import nbformat as nbf
from nbclient import NotebookClient

from lattmc.vision.spatial_review_codexgen import ROOT


def main() -> None:
    """Execute a compact review notebook from the frozen spatial audit."""
    markdown = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [markdown(
        '# Spatial-query review audit\n\n'
        'This post-hoc audit reuses frozen training-selected queries.\n'
        'The reference preserves each threshold mask size within an image\n'
        'and independently randomizes their relative positions.\n'
        'Counts are descriptive, not independent semantic evaluations.\n'
        'Implementation: `src/lattmc/vision/spatial_review_codexgen.py`.'),
        code('import json\nfrom pathlib import Path\n'
             'import numpy as np\nimport pandas as pd\n\n'
             'root = Path.cwd()\n'
             'folder = root / "vision_tokens/overcomplete/results"\n'
             'folder = folder / "spatial_review"\n'
             'audit = json.loads(\n'
             '    (folder / "spatial_audit_codexgen.json").read_text())\n'
             'rows = pd.DataFrame(audit["results"])\n'
             'assert len(rows) == 420\n'
             'assert audit["reference_exhaustive_cases"] == 284\n'
             'rows.head()'),
        markdown('## Every model and threshold\n\n'
                 'The pretrained RA queries become empty at multiplier 1.5.\n'
                 'A smaller discrepancy is not a model-quality ranking.'),
        code('summary = pd.read_json(folder / "summary_codexgen.json")\n'
             'summary'),
        markdown('## Per-dataset counts at the original thresholds'),
        code('columns = ["pooled", "common", "independent_expected",\n'
             '           "separate_query_intersection"]\n'
             'fixed = rows[rows.threshold_multiplier == 1]\n'
             'fixed.groupby(["model", "dataset"])[columns].sum()'),
        markdown('## Independent consistency checks\n\n'
                 'The archived per-image arrays store pooled, common,\n'
                 'expected, and separate-query intersection values.\n'
                 'All original unscaled counts also matched prior results.'),
        code('with np.load(folder / "per_image_codexgen.npz") as arrays:\n'
             '    assert len(arrays.files) == len(rows)\n'
             '    for row in audit["results"]:\n'
             '        key = (f"{row[\'model\']}_{row[\'dataset\']}_"\n'
             '               f"{row[\'pair\']}_"\n'
             '               f"{row[\'threshold_multiplier\']}")\n'
             '        values = arrays[key]\n'
             '        assert np.all(values[:, 1] <= values[:, 3])\n'
             '        assert np.all(values[:, 3] <= values[:, 0])\n'
             '        for column, name in enumerate(columns):\n'
             '            assert np.isclose(values[:, column].sum(),\n'
             '                              row[name])\n'
             'print("All 420 saved cases agree with per-image evidence.")')]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = dict(display_name='Python 3',
                                       language='python', name='python3')
    NotebookClient(notebook, timeout=120,
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    path = ROOT / 'notebooks/vision/spatial_review_codexgen.ipynb'
    nbf.write(notebook, path)
    print(path)


if __name__ == '__main__':
    main()
