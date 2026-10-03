"""Evaluate closure-introduced coordinates on frozen visual contexts.

For every saved semantic query, close in image and patch contexts, then
zero all coordinates active in the original query. Full dictionaries and
the original declared query projections are reported separately. Empty
extents use the fixed training/test maximum bound and are flagged vacuous.
No labels, training, or backbone inference enter this computation.
"""

import hashlib
import json
import numpy as np
from scipy import sparse

from lattmc.vision.semantic_cache_codexgen import (
    MODELS, OUT, ROOT, BoolArray, FloatArray, digest, load_evidence,
)
from lattmc.vision.semantic_report_codexgen import read_results


DEST = OUT.parent / "closure_added"


def extent(matrix: sparse.csr_matrix, query: FloatArray) -> BoolArray:
    """Test all nonzero thresholds exactly against nonnegative CSR rows.

    Matrix shape is (items, native features), query shape is (features,).
    Zero queries return all rows. Sparse zeros are treated as exact zeros;
    no activation epsilon or floating-point comparison tolerance is used.
    """
    ids = np.flatnonzero(query)
    if not len(ids):
        return np.ones(matrix.shape[0], dtype=bool)
    selected = matrix[:, ids].copy()
    selected.data = (selected.data >= query[ids][selected.indices])
    return np.asarray(selected.sum(axis=1)).ravel() == len(ids)


def close(matrix: sparse.csr_matrix, mask: BoolArray,
          bound: FloatArray) -> FloatArray:
    """Return the coordinatewise infimum, using bound for an empty extent."""
    if not mask.any():
        return bound.copy()
    return matrix[mask].min(axis=0).toarray()[0].astype(np.float64)


def input_records(model: str) -> tuple[list[dict], dict[str, BoolArray]]:
    """Read the complete previous query inventory and saved witness masks."""
    results = list(read_results(model).values())
    masks = {}
    for suffix in ("", "_followup"):
        with np.load(OUT / f"{model}{suffix}_masks_codexgen.npz") as saved:
            masks.update({name: saved[name] for name in saved.files})
    return results, masks


def experiment(model: str) -> None:
    """Save both contexts and projections for every prior query of a model.

    CSR vector row numbers and NPZ membership keys are recorded in JSON.
    Each vector row stores FG(u); h is recovered by masking u's support.
    Query-projection rows have zeros outside their declared coordinates.
    Failures of extent preservation or inclusion raise AssertionError.
    """
    evidence = load_evidence(model)
    queries, old_masks = input_records(model)
    n, sites = evidence.codes.shape[:2]
    native = evidence.native
    pooled = sparse.vstack([
        native[i:i + sites].max(0).tocsr()
        for i in range(0, native.shape[0], sites)], format="csr")
    bound = np.maximum(native.max(0).toarray()[0],
                       evidence.native_training.max(0).toarray()[0])
    bound = bound.astype(np.float64)
    arrays, vectors, records = {}, [], []
    vector_rows: dict[str, int] = {}
    # Reuse identical extents, without conflating their query supports.
    cache: dict[tuple[str, bytes], FloatArray] = {}
    for number, query in enumerate(queries):
        u = np.zeros(native.shape[1], dtype=np.float64)
        u[query["features"]] = query["query"]
        assert np.all((u >= 0) & (u <= bound))
        original_patch = old_masks[query["name"] + "__patch"]
        original_image = old_masks[query["name"] + "__pooled"]
        assert np.array_equal(extent(native, u), original_patch.ravel())
        assert np.array_equal(extent(pooled, u), original_image)
        for context, matrix, original in (
                ("patch", native, original_patch.ravel()),
                ("image", pooled, original_image)):
            key = context, original.tobytes()
            if key not in cache:
                cache[key] = close(matrix, original, bound)
            full_closed = cache[key]
            for space in ("native", "query_projection"):
                closed = full_closed.copy()
                if space == "query_projection":
                    keep = np.zeros(len(u), dtype=bool)
                    keep[query["features"]] = True
                    closed[~keep] = 0
                h = np.where(u != 0, 0, closed)
                assert np.all(u <= closed)
                assert np.array_equal(extent(matrix, closed), original)
                added_patch = extent(native, h).reshape(n, sites)
                added_image = extent(pooled, h)
                added_common = added_patch.any(1)
                added = (added_patch.ravel() if context == "patch"
                         else added_image)
                assert np.all(original <= added)
                assert np.array_equal(extent(matrix, np.maximum(u, h)),
                                      original)
                assert np.all(added_common <= added_image)
                index = len(records)
                signature = hashlib.sha256(closed.tobytes()).hexdigest()
                if signature not in vector_rows:
                    vector_rows[signature] = len(vectors)
                    vectors.append(sparse.csr_matrix(closed[None]))
                record = {
                    "index": index, "name": query["name"],
                    "vector_row": vector_rows[signature],
                    "family": query["family"],
                    "operation": query["operation"], "context": context,
                    "space": space, "original": int(original.sum()),
                    "added": int(added.sum()),
                    "new": int((added & ~original).sum()),
                    "zero_h": not bool(np.any(h)),
                    "empty_original": not bool(original.any()),
                    "same_extent": bool(np.array_equal(original, added)),
                    "universal_h": bool(added.all()),
                    "support_u": int(np.count_nonzero(u)),
                    "support_closed": int(np.count_nonzero(closed)),
                    "support_h": int(np.count_nonzero(h)),
                    "raised_original": int(np.sum((u > 0) & (closed > u))),
                    "original_images": int(original_patch.any(1).sum())
                        if context == "patch" else int(original.sum()),
                    "h_common_images": int(added_common.sum()),
                    "h_pooled_images": int(added_image.sum()),
                    "h_patches": int(added_patch.sum()),
                }
                records.append(record)
                arrays[f"{index}_original"] = original
                arrays[f"{index}_added"] = added
                arrays[f"{index}_common"] = added_common
                arrays[f"{index}_pooled"] = added_image
        if (number + 1) % 25 == 0:
            print(model, number + 1, "/", len(queries), flush=True)
    provenance = dict(evidence.provenance)
    for suffix in ("", "_followup"):
        for tail in ("_codexgen.json", "_masks_codexgen.npz"):
            path = OUT / f"{model}{suffix}{tail}"
            provenance[str(path.relative_to(ROOT))] = digest(path)
    report = {"model": model, "images": n, "sites": sites,
              "width": native.shape[1], "mapping": evidence.mapping,
              "provenance": provenance, "results": records}
    (DEST / f"{model}_codexgen.json").write_text(
        json.dumps(report, indent=2) + "\n")
    sparse.save_npz(DEST / f"{model}_closed_codexgen.npz",
                    sparse.vstack(vectors, format="csr"))
    np.savez_compressed(DEST / f"{model}_masks_codexgen.npz", **arrays)
    print(model, len(records), "context/projection cases passed", flush=True)


def main() -> None:
    """Run all models using only the existing frozen activation caches."""
    DEST.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        experiment(model)


if __name__ == "__main__":
    main()
