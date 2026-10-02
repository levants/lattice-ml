"""Build and execute the cached semantic-reading evidence notebook.

The notebook loads measured queries, verifies all saved masks, compares
components and closures, and displays the already inspected image panels.
It does not run a vision model or silently reinterpret class annotations.
"""

import nbformat as nbf
from nbclient import NotebookClient

from lattmc.vision.semantic_cache_codexgen import ROOT


def main() -> None:
    """Write an executed notebook using the existing project kernel.

    Kernel startup needs local loopback access. Execution failures propagate
    and do not overwrite the prior notebook with a partial result.
    """
    markdown = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [markdown(
        "# Reading cached visual query extents\n\n"
        "This exploratory study distinguishes image pooling, a common\n"
        "contextualized token, and the visible pixels at that token.\n"
        "Initial galleries withheld metadata labels; earlier project\n"
        "examples were known. It is not an independent blinded benchmark.\n\n"
        "Sources: `src/lattmc/vision/semantic_*_codexgen.py`.\n"
        "Frozen inventory: 372 queries; visual follow-ups: 32 queries.\n"
        "Sixteen threshold checks are reported separately."),
        code('import json\nfrom pathlib import Path\n'
             'import numpy as np\nimport pandas as pd\n'
             'from IPython.display import Image, display\n\n'
             'root = Path.cwd()\n'
             'folder = root / "vision_tokens/overcomplete/results"\n'
             'folder = folder / "semantic_reading"\n'
             'models = ["topk_k32_s0", "prisma_transcoder",\n'
             '          "pretrained_ra"]\n'
             'inventory = pd.read_json(\n'
             '    folder / "inventory_codexgen.json")\n'
             'assert len(inventory) == 404\n'
             'inventory.groupby(["model", "family"])[\n'
             '    ["pooled", "common", "patches"]].agg(["count", "median"])'),
        markdown("## Exact operations and component comparisons\n\n"
                 "For nonnegative codes, `G(q)` tests every threshold\n"
                 "against the image maxima. `H(q)` requires one token\n"
                 "to satisfy all thresholds. A vector join intersects\n"
                 "pooled extents; a meet can exceed their union. Closing\n"
                 "a query on this finite context preserves its extent."),
        code('records = []\n'
             'for model in models:\n'
             '    path = folder / f"{model}_followup_codexgen.json"\n'
             '    for row in json.loads(path.read_text())["results"]:\n'
             '        records.append({"model": model, **row})\n'
             'columns = ["model", "name", "features", "query",\n'
             '           "component_pooled", "pooled", "common",\n'
             '           "patch_count"]\n'
             'pd.DataFrame(records)[columns]'),
        markdown("## Independent saved-mask checks\n\n"
                 "All 404 outcomes, including failures, are checked.\n"
                 "The source modules additionally recompute extents and\n"
                 "closures from the original sparse activation caches."),
        code('checked = 0\n'
             'for model in models:\n'
             '    for suffix in ["", "_followup"]:\n'
             '        stem = f"{model}{suffix}"\n'
             '        path = folder / f"{stem}_codexgen.json"\n'
             '        rows = json.loads(path.read_text())["results"]\n'
             '        path = folder / f"{stem}_masks_codexgen.npz"\n'
             '        with np.load(path) as masks:\n'
             '            for row in rows:\n'
             '                name = row["name"]\n'
             '                patch = masks[f"{name}__patch"]\n'
             '                common = masks[f"{name}__common"]\n'
             '                pooled = masks[f"{name}__pooled"]\n'
             '                assert patch.sum() == row["patch_count"]\n'
             '                assert common.sum() == row["common"]\n'
             '                assert pooled.sum() == row["pooled"]\n'
             '                assert np.array_equal(patch.any(1), common)\n'
             '                assert np.all(common <= pooled)\n'
             '                checked += 1\n'
             'assert checked == 404\n'
             'print(f"Verified {checked} query outcomes.")'),
        markdown("## Visual readings and numerical witnesses\n\n"
                 "Red cells are common witnesses. Numbered blue cells\n"
                 "are separate coordinate maxima. The whole image is\n"
                 "needed: a tiger response lies above the face, while\n"
                 "a texture query responds to vegetation beside a vehicle.\n"
                 "Margins near one and alternative explanations matter."),
        code('display(Image(filename=str(folder / "figures" /\n'
             '    "semantic_witnesses_codexgen.png")))\n'
             'display(Image(filename=str(folder / "figures" /\n'
             '    "semantic_source_meet_codexgen.png")))\n'
             'display(Image(filename=str(folder / "figures" /\n'
             '    "semantic_counterexamples_codexgen.png")))'),
        markdown("## Threshold sensitivity and full-dictionary closure\n\n"
                 "The native S0/S7 TopK meet retains feature 825 and\n"
                 "returns 66 images; its 24-coordinate projection is\n"
                 "zero and returns all 547. Raising the native threshold\n"
                 "by 1.5 retains seven instrument photographs, a post-hoc\n"
                 "observation. Full closure can add coordinates outside\n"
                 "a projection without establishing their visual meaning."),
        code('sensitivity = pd.read_json(\n'
             '    folder / "sensitivity_codexgen.json")\n'
             'display(sensitivity.drop(columns="common_ids"))\n'
             'intents = json.loads(\n'
             '    (folder / "full_intents_codexgen.json").read_text())\n'
             'intents["topk_k32_s0:face6_q50"]["patch"]'),
        markdown("## Labels revealed after the first reading\n\n"
                 "These counts are a separate description of the same\n"
                 "records, not independent relevance annotations. A\n"
                 "tench label, for example, omits the pictured person.\n"
                 "The initial label-withheld interpretations are preserved\n"
                 "in `initial_reading_codexgen.json`."),
        code('path = folder / "labels_after_reading_codexgen.json"\n'
             'labels = json.loads(path.read_text())\n'
             'display(labels["topk_k32_s0"]["performance3_q80"])\n'
             'display(labels["prisma_transcoder"]["repetition2_q80"])\n'
             'display(labels["pretrained_ra"]["person_scene3_q50"])')]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = dict(display_name="Python 3",
                                       language="python", name="python3")
    NotebookClient(notebook, timeout=180,
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    target = ROOT / "notebooks/vision/semantic_reading_codexgen.ipynb"
    nbf.write(notebook, target)
    print(target)


if __name__ == "__main__":
    main()
