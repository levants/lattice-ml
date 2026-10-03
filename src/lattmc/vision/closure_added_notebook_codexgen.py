"""Build an executable notebook for closure-added visual requirements.

The notebook verifies all saved cases and visualizes numerical evidence.
It does not recompute model activations or depend on class labels for
query construction. Existing project kernels and dependencies are used.
"""

import nbformat as nbf
from nbclient import NotebookClient

from lattmc.vision.semantic_cache_codexgen import ROOT


def main() -> None:
    """Execute the notebook before writing its final reproducible artifact.

    Kernel execution requires local loopback access. Errors propagate and
    prevent replacing a previously executed notebook with partial output.
    """
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(
        "# Features introduced by closure, queried on their own\n\n"
        "For each old query `u`, compute `c=FG(u)`, then set\n"
        "`h=np.where(u != 0, 0, c)`. This removes whole coordinates,\n"
        "not just the original activation amount. The original extent\n"
        "is contained in `G(h)` by construction, and `G(u join h)=G(u)`.\n\n"
        "We evaluate 404 cached queries in image/token contexts and full/\n"
        "declared coordinate spaces: 1,616 cases. Empty-original cases\n"
        "use the fixed top vector and carry no observed shared evidence.\n"
        "Sources: `src/lattmc/vision/closure_added*_codexgen.py`."),
        code('import json\nfrom pathlib import Path\n'
             'import numpy as np\nimport pandas as pd\n'
             'from scipy import sparse\n'
             'from IPython.display import Image, display\n\n'
             'root = Path.cwd()\n'
             'folder = root / "vision_tokens/overcomplete/results"\n'
             'old_folder = folder / "semantic_reading"\n'
             'folder = folder / "closure_added"\n'
             'models = ["topk_k32_s0", "prisma_transcoder",\n'
             '          "pretrained_ra"]\n'
             'rows = []\n'
             'for model in models:\n'
             '    report = json.loads(\n'
             '        (folder / f"{model}_codexgen.json").read_text())\n'
             '    rows.extend({"model": model, **r}\n'
             '                for r in report["results"])\n'
             'frame = pd.DataFrame(rows)\n'
             'assert len(frame) == 1616\n'
             'frame.groupby(["model", "space", "context"])[\n'
             '    ["original", "added", "support_h"]].agg(\n'
             '        ["count", "median", "max"])'),
        md("## Complete saved-mask and support checks\n\n"
           "The separate source audit also recomputes all residual\n"
           "extents from original sparse codes and tests 6,561 finite\n"
           "matrix/query cases. Here we independently check saved masks\n"
           "and the exact support-removal rule for every case."),
        code('checked = 0\n'
             'for model in models:\n'
             '    old = {}\n'
             '    for suffix in ["", "_followup"]:\n'
             '        path = old_folder / f"{model}{suffix}_codexgen.json"\n'
             '        old.update({r["name"]: r\n'
             '                    for r in json.loads(path.read_text())\n'
             '                    ["results"]})\n'
             '    closed = sparse.load_npz(\n'
             '        folder / f"{model}_closed_codexgen.npz")\n'
             '    with np.load(folder / f"{model}_masks_codexgen.npz") as m:\n'
             '        for r in [r for r in rows if r["model"] == model]:\n'
             '            i = r["index"]\n'
             '            a, b = m[f"{i}_original"], m[f"{i}_added"]\n'
             '            assert a.sum() == r["original"]\n'
             '            assert b.sum() == r["added"]\n'
             '            assert np.all(a <= b)\n'
             '            q = old[r["name"]]\n'
             '            h = closed[r["vector_row"]].toarray()[0]\n'
             '            support = np.array(q["features"])[\n'
             '                np.array(q["query"]) != 0]\n'
             '            h[support] = 0\n'
             '            assert np.count_nonzero(h) == r["support_h"]\n'
             '            assert np.all(h[support] == 0)\n'
             '            checked += 1\n'
             'assert checked == 1616\n'
             'print(f"Verified {checked} saved cases.")'),
        md("## Inspect new returned images\n\n"
           "The final figure shows illustrative new members. Red cells\n"
           "are common witnesses; blue cells mark maxima for up to four\n"
           "limiting coordinates in a distributed query. They are not\n"
           "receptive fields or causal attributions. Initial anonymous\n"
           "readings and full/rank-sampled galleries are saved separately."),
        code('display(Image(filename=str(folder / "figures" /\n'
             '    "closure_added_examples_codexgen.png")))'),
        md("## Cross-collection follow-up\n\n"
           "After the initial reading, inspect all 21 facial-pair\n"
           "matches outside Imagewoof and Pets. This deliberately checks\n"
           "the apparent generality beyond the dominant pet collections.\n"
           "It is an exploratory follow-up, not blind validation."),
        code('for page in range(2):\n'
             '    display(Image(filename=str(folder / "figures" /\n'
             '        f"face_transfer_{page}.png")))'),
        md("## Exact added vectors and control outcomes\n\n"
           "Controls compare individual added features and 0.9/1/1.1\n"
           "threshold multipliers. These perturbations need not preserve\n"
           "the original extent; the theorem applies to the exact h."),
        code('controls = json.loads(\n'
             '    (folder / "controls_codexgen.json").read_text())\n'
             'for name in ["face6_q50:patch", "face3_q80:patch",\n'
             '             "performance3_q80:image"]:\n'
             '    item = controls["topk_k32_s0"][name]\n'
             '    print(name, "added coordinates:", len(item["features"]))\n'
             '    display(pd.DataFrame(item["scales"]))\n'
             '    if len(item["features"]) <= 10:\n'
             '        display(pd.DataFrame({"feature": item["features"],\n'
             '                              "threshold": item["values"]}))'),
        md("## Later labels and interpretation limits\n\n"
           "Label tabulations describe the same objects, not independent\n"
           "relevance judgments. Prior project familiarity is explicit.\n"
           "Broad facial relationships, weak texture/context requirements,\n"
           "and high-dimensional instance recovery are different outcomes.\n"
           "A zero residual is universal; an empty-original closure is\n"
           "vacuous. None establishes downstream causal use."),
        code('display(controls["topk_k32_s0"]["face3_q80:patch"]["labels"])\n'
             'nonempty = frame[~frame.empty_original]\n'
             'nonempty.groupby(["model", "space", "context"])[\n'
             '    ["zero_h", "same_extent", "universal_h"]].sum()')]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = dict(display_name="Python 3",
                                       language="python", name="python3")
    for cell in notebook.cells:
        assert all(len(line) <= 79 for line in cell.source.splitlines())
    NotebookClient(notebook, timeout=180,
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    target = ROOT / "notebooks/vision/closure_added_codexgen.ipynb"
    nbf.write(notebook, target)
    print(target)


if __name__ == "__main__":
    main()
