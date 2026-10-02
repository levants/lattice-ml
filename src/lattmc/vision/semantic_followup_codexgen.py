"""Run targeted follow-ups motivated by the label-withheld image atlases.

These are explicitly exploratory choices, separate from the complete
fixed-seed query inventory. Quantiles still use training patches only.
"""

import json

import numpy as np

from lattmc.vision.semantic_cache_codexgen import (
    MODELS, OUT, load_evidence, projected_codes,
)
from lattmc.vision.semantic_gallery_codexgen import (
    extent_sheets, query_gallery,
)
from lattmc.vision.semantic_queries_codexgen import Query, evaluate


TARGETS = {
    "topk_k32_s0": {
        "face2": [329, 507], "face3": [329, 507, 1271],
        "face4": [329, 507, 1271, 1430],
        "face6": [329, 507, 1271, 1430, 871, 468],
        "structure2": [282, 798], "contrast2": [329, 798],
        "contrast3": [329, 798, 282],
        "contrast4": [329, 798, 282, 942],
        "performance3": [54, 585, 942],
    },
    "prisma_transcoder": {
        "texture2": [18417, 40486],
        "pattern3": [32929, 41201, 18417],
        "repetition2": [13769, 5792],
        "contrast2": [19489, 48542],
    },
    "pretrained_ra": {
        "person_scene3": [30521, 22522, 8811],
        "contrast2": [4271, 11318],
        "background3": [4093, 21444, 26994],
    },
}

INSPECT = {
    "topk_k32_s0": [("source_0_7_native_meet", "common", False),
                    ("structure2_q50", "common", True),
                    ("performance3_q80", "pooled", True)],
    "prisma_transcoder": [("repetition2_q80", "common", True),
                          ("pattern3_q80", "pooled", True),
                          ("source_0_1_2_native_meet", "common", False)],
    "pretrained_ra": [("person_scene3_q50", "pooled", True)],
}


def main() -> None:
    """Save both median/upper-quantile follow-ups and selected galleries."""
    inspection = {}
    for model in MODELS:
        evidence = load_evidence(model)
        results, arrays = [], {}
        for name, indices in TARGETS[model].items():
            features = np.array(indices, dtype=np.int64)
            native = evidence.native_training[:, features].toarray()
            for level in (.5, .8):
                values = np.array([np.quantile(x[x > 0], level)
                                   for x in native.T])
                key = f"{name}_q{int(level * 100)}"
                query = Query(key, "visual_followup", features, values,
                              np.diag(values), "join", [])
                result, masks = evaluate(evidence, query)
                result["panels"] = query_gallery(
                    evidence, projected_codes(evidence, features),
                    values, f"{model}_{key}")
                results.append(result)
                arrays.update({f"{key}__{k}": v for k, v in masks.items()})
        report = {"model": model, "selection": "post-atlas exploratory",
                  "results": results}
        (OUT / f"{model}_followup_codexgen.json").write_text(
            json.dumps(report, indent=2) + "\n")
        np.savez_compressed(OUT / f"{model}_followup_masks_codexgen.npz",
                            **arrays)
        original = json.loads((OUT / f"{model}_codexgen.json").read_text())
        names = ["source_0_7_native_meet", "source_0_7_projected_meet",
                 "source_0_1_2_native_meet", "coactive_8_1_q80",
                 "existing_0_meet", "existing_0_join"]
        for result in original["results"]:
            if result["name"] not in names:
                continue
            features = np.array(result["features"], dtype=np.int64)
            query_gallery(evidence, projected_codes(evidence, features),
                          np.array(result["query"]),
                          f"{model}_{result['name']}")
        for name, mode, followup in INSPECT[model]:
            candidates = results if followup else original["results"]
            result = next(r for r in candidates if r["name"] == name)
            codes = projected_codes(evidence, np.array(result["features"]))
            inspection[f"{model}_{name}_{mode}"] = extent_sheets(
                evidence, codes, np.array(result["query"]),
                f"{model}_{name}", mode)
        print(model, len(results), "targeted queries", flush=True)
    (OUT / "inspection_items_codexgen.json").write_text(
        json.dumps(inspection, indent=2) + "\n")


if __name__ == "__main__":
    main()
