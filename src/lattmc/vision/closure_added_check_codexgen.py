"""Check added-coordinate algebra, numerical controls, and later labels.

Small finite matrices test the construction independently of the sparse
implementation. Cached outputs retain full membership masks. Selected
case controls compare single added coordinates and threshold changes;
metadata labels are tabulated only after visual notes have been recorded.
"""

from collections import Counter
import itertools
import json

import numpy as np
from scipy import sparse

from lattmc.vision.closure_added_codexgen import DEST, extent
from lattmc.vision.closure_added_visual_codexgen import CASES
from lattmc.vision.semantic_cache_codexgen import (
    DATA, MODELS, load_evidence,
)
from lattmc.vision.semantic_report_codexgen import read_results

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def finite_check() -> int:
    """Exhaust 729 three-item, two-coordinate matrices and nine queries.

    Compare sparse retrieval with direct dense inequalities; assert
    support removal, extensivity of the returned set, and recovery by
    rejoining u. Empty extents close to the fixed upper vector (2, 2).
    """
    checked = 0
    for values in itertools.product(range(3), repeat=6):
        z = np.array(values, dtype=float).reshape(3, 2)
        for query in itertools.product(range(3), repeat=2):
            u = np.array(query, dtype=float)
            original = (z >= u).all(1)
            closed = z[original].min(0) if original.any() else np.full(2, 2.)
            h = np.where(u != 0, 0, closed)
            added = (z >= h).all(1)
            assert np.all(original <= added)
            assert np.array_equal((z >= np.maximum(u, h)).all(1), original)
            assert np.array_equal(extent(sparse.csr_matrix(z), h), added)
            assert np.all(h[u != 0] == 0)
            checked += 1
    return checked


def main() -> None:
    """Verify saved outcomes and write controls after the anonymous reading."""
    assert (DEST / "initial_reading_codexgen.json").is_file()
    controls, checked = {}, 0
    for model in MODELS:
        evidence = load_evidence(model)
        report = json.loads((DEST / f"{model}_codexgen.json").read_text())
        rows = report["results"]
        originals = read_results(model)
        closed = sparse.load_npz(DEST / f"{model}_closed_codexgen.npz")
        n, sites = evidence.codes.shape[:2]
        pooled = sparse.vstack([
            evidence.native[i:i + sites].max(0).tocsr()
            for i in range(0, evidence.native.shape[0], sites)], format="csr")
        floors = {"image": pooled.min(0).toarray()[0],
                  "patch": evidence.native.min(0).toarray()[0]}
        metadata = {dataset: json.loads(
            (DATA / f"dataset/{dataset}_codexgen.json").read_text())
            for dataset, _ in evidence.mapping}
        labels = [str(metadata[d][i]["label"]) for d, i in evidence.mapping]
        controls[model] = {}
        with np.load(DEST / f"{model}_masks_codexgen.npz") as masks:
            for r in rows:
                index = r["index"]
                original = masks[f"{index}_original"]
                added = masks[f"{index}_added"]
                common = masks[f"{index}_common"]
                image = masks[f"{index}_pooled"]
                assert original.sum() == r["original"]
                assert added.sum() == r["added"]
                assert np.all(original <= added)
                assert common.sum() == r["h_common_images"]
                assert image.sum() == r["h_pooled_images"]
                if r["context"] == "patch":
                    assert np.array_equal(added.reshape(n, sites).any(1),
                                          common)
                q = originals[r["name"]]
                support = np.array(q["features"])[np.array(q["query"]) != 0]
                h = closed[r["vector_row"]].toarray()[0]
                h[support] = 0
                assert np.count_nonzero(h) == r["support_h"]
                matrix = evidence.native if r["context"] == "patch" else pooled
                # Recompute every saved residual mask from original codes.
                assert np.array_equal(extent(matrix, h), added)
                r["above_global_floor"] = int(np.sum(
                    h > floors[r["context"]]))
                checked += 1
                if (r["name"], r["context"]) not in CASES[model]:
                    continue
                if r["space"] != "native":
                    continue
                case = {"result": r, "features": np.flatnonzero(h).tolist(),
                        "values": h[h > 0].tolist(), "scales": [],
                        "individual": [], "labels": {}, "new_labels": {}}
                for scale in (.9, 1., 1.1):
                    p = extent(evidence.native, h * scale).reshape(n, sites)
                    g = extent(pooled, h * scale)
                    case["scales"].append({
                        "scale": scale, "patches": int(p.sum()),
                        "common": int(p.any(1).sum()), "pooled": int(g.sum())})
                if np.count_nonzero(h) <= 12:
                    for coordinate in np.flatnonzero(h):
                        single = np.zeros(len(h))
                        single[coordinate] = h[coordinate]
                        p = extent(evidence.native, single).reshape(n, sites)
                        case["individual"].append({
                            "feature": int(coordinate),
                            "threshold": float(h[coordinate]),
                            "patches": int(p.sum()),
                            "images": int(p.any(1).sum())})
                old_images = (original.reshape(n, sites).any(1)
                              if r["context"] == "patch" else original)
                shown = common if r["context"] == "patch" else image
                for key, mask in (("labels", shown),
                                  ("new_labels", shown & ~old_images)):
                    case[key] = dict(Counter(labels[i]
                                             for i in np.flatnonzero(mask)))
                controls[model][f"{r['name']}:{r['context']}"] = case
        # Annotate all cases with the count exceeding unconditional floors.
        (DEST / f"{model}_codexgen.json").write_text(
            json.dumps(report, indent=2) + "\n")
    verification = {"finite_cases": finite_check(),
                    "recomputed_context_projection_cases": checked}
    (DEST / "controls_codexgen.json").write_text(
        json.dumps(controls, indent=2) + "\n")
    (DEST / "verification_codexgen.json").write_text(
        json.dumps(verification, indent=2) + "\n")
    logging.info(verification)


if __name__ == "__main__":
    main()
