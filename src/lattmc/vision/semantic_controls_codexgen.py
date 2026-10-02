"""Audit full intents, sensitivity, and labels after visual interpretation.

The label reveal is a descriptive cross-check on the same records, not
independent semantic validation. Original dictionaries remain separate.
"""

from collections import Counter
import json

import numpy as np
from scipy import sparse

from lattmc.vision.semantic_cache_codexgen import (
    DATA, MODELS, OUT, load_evidence, projected_codes,
)
from lattmc.vision.semantic_queries_codexgen import (
    Query, evaluate, memberships,
)


DETAILS = {
    "topk_k32_s0": ["source_0_7_native_meet", "face3_q80",
                    "face6_q50", "structure2_q50", "contrast2_q50",
                    "performance3_q80"],
    "prisma_transcoder": ["repetition2_q80", "pattern3_q80",
                          "source_0_1_2_native_meet",
                          "source_0_1_2_3_4_5_6_7_native_meet"],
    "pretrained_ra": ["person_scene3_q50", "source_0_7_native_meet"],
}


def main() -> None:
    """Verify all saved masks and reveal labels after notes are recorded."""
    assert (OUT / "initial_reading_codexgen.json").is_file()
    labels, intents, sensitivity, inventory = {}, {}, [], []
    for model in MODELS:
        evidence = load_evidence(model)
        metadata = {name: json.loads((DATA / f"dataset/{name}_codexgen.json")
                                     .read_text())
                    for name in {pair[0] for pair in evidence.mapping}}
        all_labels = [str(metadata[dataset][row]["label"])
                      for dataset, row in evidence.mapping]
        labels[model] = {}
        results, masks = [], {}
        for suffix in ("", "_followup"):
            path = OUT / f"{model}{suffix}_codexgen.json"
            results.extend(json.loads(path.read_text())["results"])
            with np.load(OUT / f"{model}{suffix}_masks_codexgen.npz") as f:
                masks.update({key: f[key] for key in f.files})
        p = evidence.codes.shape[1]
        pooled_native = sparse.vstack([
            evidence.native[i:i + p].max(0).tocsr()
            for i in range(0, evidence.native.shape[0], p)], format="csr")
        for result in results:
            name = result["name"]
            patch = masks[f"{name}__patch"]
            common = masks[f"{name}__common"]
            pooled = masks[f"{name}__pooled"]
            assert int(patch.sum()) == result["patch_count"]
            assert np.array_equal(patch.any(1), common)
            assert int(common.sum()) == result["common"]
            assert int(pooled.sum()) == result["pooled"]
            assert np.all(common <= pooled)
            counts = {}
            for scope, mask in (("pooled", pooled), ("common", common)):
                counts[scope] = dict(Counter(all_labels[i]
                                            for i in np.flatnonzero(mask)))
                counts[f"{scope}_datasets"] = dict(Counter(
                    evidence.mapping[i][0] for i in np.flatnonzero(mask)))
            labels[model][name] = counts
            inventory.append({"model": model, "name": name,
                              "family": result["family"],
                              "positive": result["positive_coordinates"],
                              "pooled": int(pooled.sum()),
                              "common": int(common.sum()),
                              "patches": int(patch.sum())})
            if name not in DETAILS[model]:
                continue
            key = f"{model}:{name}"
            intents[key] = {}
            for scope, mask, matrix in (
                    ("patch", patch.ravel(), evidence.native),
                    ("image", pooled, pooled_native)):
                if not mask.any():
                    intents[key][scope] = {"empty": True}
                    continue
                closed = matrix[mask].min(0).toarray()[0]
                active = np.flatnonzero(closed > 0)
                original = set(np.array(result["features"])[
                    np.array(result["query"]) > 0].tolist())
                closed_codes = projected_codes(evidence, active)
                checked = memberships(closed_codes, closed[active])[
                    0 if scope == "patch" else 2].ravel()
                assert np.array_equal(checked, mask)
                intents[key][scope] = {
                    "features": active.tolist(),
                    "values": closed[active].tolist(),
                    "added": sorted(set(active.tolist()) - original)}
            if name in ("face3_q80", "face6_q50", "repetition2_q80",
                        "source_0_7_native_meet"):
                scales = ([.9, 1., 1.1, 1.5] if model == "topk_k32_s0"
                          and name == "source_0_7_native_meet"
                          else [.9, 1., 1.1])
                for scale in scales:
                    values = np.array(result["query"]) * scale
                    query = Query(f"{name}_scale_{scale}", "sensitivity",
                                  np.array(result["features"]), values,
                                  values[None], "identity", [])
                    # The scale can exceed the original context bound.
                    # Membership remains defined; do not close such queries.
                    codes = projected_codes(evidence, query.features)
                    a, b, c = memberships(codes, values)
                    sensitivity.append({"model": model, "name": name,
                                        "scale": scale,
                                        "patches": int(a.sum()),
                                        "common": int(b.sum()),
                                        "pooled": int(c.sum()),
                                        "common_ids": np.flatnonzero(b)
                                            .tolist()})
        # Source-prefix meets must expand and joins contract in both scopes.
        for scope in ("native", "projected"):
            for operation in ("meet", "join"):
                previous = None
                for count in (2, 3, 4, 6, 8):
                    key = "source_" + "_".join(map(str, range(count)))
                    key += f"_{scope}_{operation}"
                    current = masks[f"{key}__pooled"]
                    if previous is not None:
                        assert np.all((previous <= current) if operation
                                      == "meet" else (current <= previous))
                    previous = current
        print(model, "full-intent and mask checks passed", flush=True)
    outputs = {"labels_after_reading": labels, "full_intents": intents,
               "sensitivity": sensitivity, "inventory": inventory}
    for name, result in outputs.items():
        (OUT / f"{name}_codexgen.json").write_text(
            json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
