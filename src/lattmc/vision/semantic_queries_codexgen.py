"""Audit label-free image and patch retrieval beyond paired features.

Queries use frozen training statistics or actual training-token codes.
All declared combinations are retained, including empty and zero meets.
Closure is computed on the fixed held-out context, not on a new sample.
"""

import itertools
import json
from dataclasses import dataclass

import numpy as np

from lattmc.vision.semantic_cache_codexgen import (
    DATA, MODELS, OUT, BoolArray, Evidence, FloatArray, IntArray,
    load_evidence, projected_codes,
)


@dataclass
class Query:
    """Represent a query in an explicitly named coordinate projection."""

    name: str
    family: str
    features: IntArray
    values: FloatArray
    components: FloatArray
    operation: str
    sources: list[dict]


def memberships(codes: FloatArray, query: FloatArray
                ) -> tuple[BoolArray, BoolArray, BoolArray]:
    """Return patch, same-site image, and max-pooled image masks.

    Codes have shape (N, P, J), query has shape (J,), and zero
    requirements hold automatically for these nonnegative codes.
    """
    patch = (codes >= query).all(2)
    return patch, patch.any(1), (codes.max(1) >= query).all(1)


def coordinate_queries(evidence: Evidence) -> list[Query]:
    """Construct all singles and 30 larger training-designed projections.

    At each arity 2, 3, 4, 6, 8, use two greedy coactivation chains,
    two greedy low-correlation chains, and two seeded random sets.
    Each combination is evaluated at positive training quantiles .5/.8.
    Greedy chains start at the first two variance-ranked coordinates.
    """
    training = evidence.training.reshape(-1, len(evidence.features))
    quantiles = np.array([np.quantile(x[x > 0], [.5, .8])
                         for x in training.T])
    correlation = np.corrcoef(training.T)
    queries = []
    for j, feature in enumerate(evidence.features):
        queries.append(Query(f"single_{feature}", "single",
                             np.array([feature]), quantiles[j, 1:2],
                             quantiles[j, 1:2].reshape(1, 1),
                             "identity", []))
    rng = np.random.default_rng(20261003)
    for size in (2, 3, 4, 6, 8):
        for family, replicate in itertools.product(
                ("coactive", "lowcorr", "random"), range(2)):
            if family == "random":
                selected = rng.choice(24, size, replace=False).tolist()
            else:
                selected = [replicate]
                while len(selected) < size:
                    remaining = [j for j in range(24) if j not in selected]
                    scores = correlation[np.ix_(remaining, selected)].mean(1)
                    chosen = (np.argmax(scores) if family == "coactive"
                              else np.argmin(scores))
                    selected.append(remaining[int(chosen)])
            for level in range(2):
                values = quantiles[selected, level]
                queries.append(Query(
                    f"{family}_{size}_{replicate}_q{[50, 80][level]}",
                    family, evidence.features[selected], values,
                    np.diag(values), "join", []))
    return queries


def source_queries(evidence: Evidence) -> list[Query]:
    """Compare meets/joins of actual tokens before and after projection.

    Source images are 12 seeded training draws. Each token maximizes the
    norm on the pre-existing 24-coordinate projection. Fixed prefixes of
    length 2, 3, 4, 6, 8 and the pair (S0, S7) are retained. Native queries
    include every coordinate active at any source token.
    """
    ids = np.random.default_rng(20261003).choice(200, 12, replace=False)
    sites = np.linalg.norm(evidence.training[ids], axis=2).argmax(1)
    p = evidence.training.shape[1]
    native = evidence.native_training[ids * p + sites]
    result = []
    sets = [list(range(n)) for n in (2, 3, 4, 6, 8)] + [[0, 7]]
    for selection in sets:
        records = [{"source": int(s), "training_index": int(ids[s]),
                    "row": int(evidence.training_rows[ids[s]]),
                    "site": int(sites[s])} for s in selection]
        selected = native[selection]
        full_features = np.unique(selected.indices).astype(np.int64)
        for scope in ("native", "projected"):
            features = (full_features if scope == "native"
                        else evidence.features)
            components = selected[:, features].toarray().astype(np.float64)
            for operation in ("meet", "join"):
                values = (components.min(0) if operation == "meet"
                          else components.max(0))
                name = "source_" + "_".join(map(str, selection))
                name += f"_{scope}_{operation}"
                result.append(Query(name, "source", features, values,
                                    components, operation, records))
    return result


def existing_queries(evidence: Evidence) -> list[Query]:
    """Retain all earlier paired queries for direct count comparisons."""
    path = DATA / f"results/{evidence.model}/queries_codexgen.json"
    result = []
    for index, row in enumerate(json.loads(path.read_text())["queries"]):
        components = np.array([row["u"], row["v"]])
        for operation in ("u", "v", "meet", "join"):
            values = {"u": components[0], "v": components[1],
                      "meet": components.min(0),
                      "join": components.max(0)}[operation]
            result.append(Query(f"existing_{index}_{operation}",
                                "existing", np.array(row["features"]),
                                values, components, operation, []))
    return result


def evaluate(evidence: Evidence, query: Query
             ) -> tuple[dict, dict[str, BoolArray]]:
    """Compute exact extents, closures, component controls, and witnesses.

    Returned masks preserve all matches in a compressed companion file.
    Query bounds include training and held-out maxima, so frozen queries
    lie in the declared fixed product lattice, including empty extents.
    """
    codes = projected_codes(evidence, query.features)
    patch, common, pooled = memberships(codes, query.values)
    image_codes = codes.max(1)
    train_max = evidence.native_training[:, query.features].max(0)
    bound = np.maximum(image_codes.max(0), train_max.toarray()[0])
    assert np.all(query.values <= bound)
    patch_closed = codes[patch].min(0) if patch.any() else bound
    image_closed = image_codes[pooled].min(0) if pooled.any() else bound
    assert np.array_equal(memberships(codes, patch_closed)[0], patch)
    assert np.array_equal(memberships(codes, image_closed)[2], pooled)
    component_masks = [memberships(codes, q) for q in query.components]
    h_intersection = np.logical_and.reduce([m[1] for m in component_masks])
    g_intersection = np.logical_and.reduce([m[2] for m in component_masks])
    g_union = np.logical_or.reduce([m[2] for m in component_masks])
    if query.operation == "join":
        assert np.array_equal(pooled, g_intersection)
        assert np.all(common <= h_intersection)
    if query.operation == "meet":
        assert np.all(g_union <= pooled)
    assert np.all(common <= pooled)
    positive = query.values > 0
    ratios = codes[:, :, positive] / query.values[positive]
    common_margin = (ratios.min(2).max(1) if positive.any()
                     else np.ones(len(codes)))
    pooled_margin = (ratios.max(1).min(1) if positive.any()
                     else np.ones(len(codes)))
    strong = np.argsort(-common_margin, kind="stable")
    strong = strong[common[strong]]
    choices = list(dict.fromkeys(np.concatenate([
        strong[:4], strong[-2:][::-1],
        np.flatnonzero(pooled & ~common)[:2]]).tolist()))
    witness = []
    for index in choices:
        site = (int(ratios[index].min(1).argmax())
                if positive.any() else 0)
        witness.append({"image": index, "site": site,
                        "code": codes[index, site].tolist(),
                        "maxima": image_codes[index].tolist(),
                        "argmax_sites": codes[index].argmax(0).tolist(),
                        "common_margin": float(common_margin[index]),
                        "pooled_margin": float(pooled_margin[index])})
    result = {"name": query.name, "family": query.family,
              "features": query.features.tolist(),
              "query": query.values.tolist(),
              "components": query.components.tolist(),
              "operation": query.operation, "sources": query.sources,
              "positive_coordinates": int(positive.sum()),
              "patch_count": int(patch.sum()), "common": int(common.sum()),
              "pooled": int(pooled.sum()),
              "component_pooled": [int(m[2].sum()) for m in component_masks],
              "component_common": [int(m[1].sum()) for m in component_masks],
              "component_union": int(g_union.sum()),
              "component_intersection": int(g_intersection.sum()),
              "separate_common_intersection": int(h_intersection.sum()),
              "image_closure": image_closed.tolist(),
              "patch_closure": patch_closed.tolist(),
              "closure_bound": bound.tolist(), "witnesses": witness}
    masks = {"patch": patch, "common": common, "pooled": pooled}
    return result, masks


def main() -> None:
    """Write the complete prespecified numerical audit, without labels."""
    OUT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        evidence = load_evidence(model)
        definitions = (existing_queries(evidence)
                       + coordinate_queries(evidence)
                       + source_queries(evidence))
        results, arrays = [], {}
        for query in definitions:
            result, masks = evaluate(evidence, query)
            results.append(result)
            arrays.update({f"{query.name}__{key}": value
                           for key, value in masks.items()})
        report = {"model": model, "images": len(evidence.images),
                  "sites": evidence.codes.shape[1],
                  "mapping": evidence.mapping,
                  "training_rows": evidence.training_rows.tolist(),
                  "provenance": evidence.provenance,
                  "selection_seed": 20261003, "results": results}
        (OUT / f"{model}_codexgen.json").write_text(
            json.dumps(report, indent=2) + "\n")
        np.savez_compressed(OUT / f"{model}_masks_codexgen.npz", **arrays)
        print(model, len(results), "queries verified", flush=True)


if __name__ == "__main__":
    main()
