"""Independently verify stored evidence and manuscript source conventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from lattmc.vision.contexts_codexgen import VectorContext, spatial_extents
from lattmc.vision.experiment_codexgen import benchmark
from lattmc.vision.paths_codexgen import experiment_root, load_digit_cache


def verify_results(paper: Path) -> dict[str, int]:
    """Verify cached query identities and the recorded artifact hashes."""
    folder = experiment_root()
    report = json.loads((folder / "results/results_codexgen.json").read_text())
    digits = load_digit_cache(folder, 17)
    digest = hashlib.sha256(digits["data"].tobytes()
                            + digits["labels"].tobytes()).hexdigest()
    assert digest == report["data_sha256"]
    for name, expected in report["artifact_sha256"].items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        assert actual == expected
    verified = 0
    for seed in report["seeds"]:
        cache = load_digit_cache(folder, seed)
        train, cal, test = (
            cache[key] for key in ("train", "calibration", "test"))
        assert not set(train) & set(test)
        assert not set(train) & set(cal)
        assert not set(cal) & set(test)
        assert len(set(train) | set(cal) | set(test)) == len(digits["labels"])
        assert np.array_equal(cache["labels"], digits["labels"])
        rows, queries, sources, scores = benchmark(
            cache["patches"], cache["dense"], cache["labels"],
            train, cal, test, seed)
        expected = [r for r in report["retrieval"] if r["seed"] == seed]
        assert rows == expected
        assert np.array_equal(queries, cache["queries"])
        assert np.array_equal(sources, cache["sources"])
        assert np.array_equal(scores, cache["scores"])
        context = VectorContext(cache["patches"][train].max(axis=1))
        for index, row in enumerate(rows):
            query = queries[index]
            image, site = spatial_extents(cache["patches"][test], query)
            assert np.all(~site | image)
            assert np.array_equal(scores[index, 0] >= 1, image)
            truth = digits["labels"][test] == row["digit"]
            assert f1_score(truth, image) == row["graded_f1"]
            ap = average_precision_score(truth, scores[index, 0])
            assert ap == row["graded_ap"]
            assert np.all(digits["labels"][sources[index]] == row["digit"])
            assert set(sources[index]) <= set(train)
            closed = context.close_query(query)
            assert np.array_equal(
                context.extent(query), context.extent(closed))
            verified += 1
    return {"verified_queries": verified, "verified_artifacts":
            len(report["artifact_sha256"])}


def audit_sources(paper: Path, source: Path) -> dict[str, int]:
    """Audit LaTeX source labels, citations, and bibliography entries."""
    paper = Path(paper)
    paths = sorted(paper.rglob("*.tex"))
    combined = "\n".join(p.read_text() for p in paths)
    labels = re.findall(r"\\label\{([^}]+)\}", combined)
    assert len(labels) == len(set(labels)), "Duplicate labels"
    references = re.findall(r"\\(?:[cC]?ref|eqref)\{([^}]+)\}", combined)
    assert all(key.strip() in labels for group in references
               for key in group.split(",")), "Missing reference"
    bibliography = (paper / "references.bib").read_text()
    keys = re.findall(r"@\w+\{([^,]+),", bibliography)
    assert len(keys) == len(set(keys)), "Duplicate citation keys"
    citations = re.findall(r"\\cite\{([^}]+)\}", combined)
    assert all(key.strip() in keys for group in citations
               for key in group.split(",")), "Missing citation"
    assert r"\(" not in combined and r"\)" not in combined
    # Remove text subscripts escaped as \_ and keys before script checks.
    math_source = re.sub(r"\\(?:label|cite|input|includegraphics|[cC]?ref)"
                         r"(?:\[[^\]]*\])?\{[^}]*\}", "", combined)
    assert not re.search(r"(?<!\\)[_^](?!\{)", math_source)
    environments = ("equation", "align", "gather", "theorem", "lemma",
                    "proposition", "definition", "example", "table", "figure")
    for env in environments:
        for block in re.findall(r"\\begin\{" + env + r"\}(.*?)\\end\{"
                                + env + r"\}", combined, flags=re.S):
            assert r"\label{" in block, f"Unlabeled {env}"
            if env in ("theorem", "lemma", "proposition"):
                assert block.lstrip().startswith("["), "Unnamed result"
    for heading in re.finditer(r"\\(?:sub)*section\{[^}]+\}", combined):
        assert re.match(r"\s*\\label\{", combined[heading.end():])
    for path in paths + [paper / "references.bib"] + list(
            Path(source).glob("*_codexgen.py")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            assert len(line) <= 79, f"Width: {path}:{number} ({len(line)})"
    return {"tex_files": len(paths), "labels": len(labels),
            "bibliography_entries": len(keys)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent)
    arguments = parser.parse_args()
    print(verify_results(arguments.paper))
    print(audit_sources(arguments.paper, arguments.source))
