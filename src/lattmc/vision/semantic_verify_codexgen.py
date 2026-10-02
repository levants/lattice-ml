"""Verify semantic-reading provenance and reproducibility of cached queries.

This audit compares the earlier paired-query counts, re-evaluates every
saved witness against its thresholds, and checks the source/extent algebra.
It distinguishes these numerical checks from any semantic evaluation.
"""

import ast
import json
from pathlib import Path

import nbformat
import numpy as np

from lattmc.vision.semantic_cache_codexgen import (
    DATA, MODELS, OUT, ROOT, digest,
)


def main() -> None:
    """Check source, cache, and notebook evidence and save the audit counts."""
    checked, legacy, witness_count = 0, 0, 0
    for model in MODELS:
        base = json.loads((OUT / f"{model}_codexgen.json").read_text())
        for relative, expected in base["provenance"].items():
            assert digest(ROOT / relative) == expected, relative
        mapping = base["mapping"]
        for suffix in ("", "_followup"):
            records = json.loads((OUT / f"{model}{suffix}_codexgen.json")
                                 .read_text())["results"]
            mask_path = OUT / f"{model}{suffix}_masks_codexgen.npz"
            with np.load(mask_path) as arrays:
                for row in records:
                    name = row["name"]
                    patch = arrays[name + "__patch"]
                    common = arrays[name + "__common"]
                    pooled = arrays[name + "__pooled"]
                    assert patch.shape[0] == len(mapping) == 547
                    assert np.array_equal(patch.any(1), common)
                    assert int(pooled.sum()) == row["pooled"]
                    assert int(patch.sum()) == row["patch_count"]
                    assert np.all(common <= pooled)
                    q = np.array(row["query"])
                    components = np.array(row["components"])
                    if row["operation"] == "join":
                        assert np.array_equal(components.max(0), q)
                        assert row["pooled"] == row["component_intersection"]
                    if row["operation"] == "meet":
                        assert np.array_equal(components.min(0), q)
                        assert row["pooled"] >= row["component_union"]
                    for item in row["witnesses"]:
                        index, site = item["image"], item["site"]
                        exact_patch = (np.array(item["code"]) >= q).all()
                        exact_image = (np.array(item["maxima"]) >= q).all()
                        assert exact_patch == patch[index, site]
                        assert exact_image == pooled[index]
                        witness_count += 1
                    if name.startswith("existing_"):
                        _, pair, operation = name.split("_")
                        for dataset in {m[0] for m in mapping}:
                            path = DATA / f"results/{model}"
                            path = path / f"{dataset}_codexgen.json"
                            old = json.loads(path.read_text())
                            reference = old["queries"][int(pair)][
                                "operations"][operation]
                            mask = np.array([m[0] == dataset for m in mapping])
                            assert pooled[mask].sum() == reference[
                                "pooled_test"]
                            assert common[mask].sum() == reference[
                                "common_test"]
                            legacy += 1
                    checked += 1
    modules = list((ROOT / "src/lattmc/vision").glob("semantic*_codexgen.py"))
    for path in modules:
        source = path.read_text()
        assert all(len(line) <= 79 for line in source.splitlines()), path
        tree = ast.parse(source)
        assert ast.get_docstring(tree), path
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.returns and ast.get_docstring(node), node.name
                assert all(arg.annotation for arg in node.args.args)
    notebook_path = ROOT / "notebooks/vision/semantic_reading_codexgen.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    executed = 0
    for cell in notebook.cells:
        assert all(len(line) <= 79 for line in cell.source.splitlines())
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert all(out.output_type != "error" for out in cell.outputs)
            executed += 1
    result = {"queries": checked, "legacy_dataset_comparisons": legacy,
              "numerical_witnesses": witness_count,
              "typed_documented_modules": len(modules),
              "executed_notebook_cells": executed}
    (OUT / "verification_codexgen.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
