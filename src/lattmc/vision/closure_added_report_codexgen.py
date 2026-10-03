"""Export tables and measured visual comparisons for added-only queries.

The optional paper directory receives only generated tables and one final
figure. Numerical outputs and intermediate panels remain under data.
Explicit final choices are illustrative, not a semantic evaluation set.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse

from lattmc.vision.closure_added_codexgen import DEST
from lattmc.vision.closure_added_visual_codexgen import panel
from lattmc.vision.semantic_cache_codexgen import MODELS, load_evidence
from lattmc.vision.semantic_report_codexgen import read_results, table


DETAILS = [
    (0, "face6_q50", "patch", "TopK facial six"),
    (0, "face3_q80", "patch", "TopK facial triple"),
    (1, "repetition2_q80", "patch", "TC texture pair"),
    (1, "repetition2_q80", "image", "TC texture pair"),
    (0, "source_0_7_native_meet", "patch", "TopK source meet"),
    (0, "source_0_7_native_meet", "image", "TopK source meet"),
    (0, "performance3_q80", "image", "TopK performance"),
    (2, "person_scene3_q50", "image", "RA person--scene"),
    (0, "contrast2_q50", "patch", "TopK contrast pair"),
    (1, "source_0_1_2_native_meet", "patch", "TC source triple"),
]


def tables(reports: dict[str, list[dict]]) -> dict[str, str]:
    """Build selected-case and complete-inventory tables from saved counts."""
    rows = []
    for model, name, context, title in DETAILS:
        r = next(r for r in reports[MODELS[model]] if r["name"] == name
                 and r["context"] == context and r["space"] == "native")
        counts = [r[key] for key in ("support_h", "original", "added",
                                    "h_common_images", "h_pooled_images")]
        rows.append(title + " & " + ("T" if context == "patch" else "I")
                    + " & " + " & ".join(map(str, counts)))
    result = {"closure_added_cases_codexgen": table(
        "Added-only retrieval in full dictionaries. T denotes token\n"
        "    context; I denotes image context. The last two columns\n"
        "    evaluate the same $h$ with common-site and pooled witnesses.",
        "tab:closure-added-cases", "llrrrrr",
        r"Original query & Context & $|\mathrm{supp}(h)|$"
        "\n    " + r"& $|G(u)|$ & $|G(h)|$ & $|H(h)|$"
        r" & $|G_{\mathrm{img}}(h)|$", rows)}
    rows = []
    for model, title in zip(MODELS, ("TopK", "TC", "RA")):
        for space, short in (("native", "Full"),
                             ("query_projection", "Proj.")):
            for context, letter in (("patch", "T"), ("image", "I")):
                all_rows = [r for r in reports[model] if r["space"] == space
                            and r["context"] == context]
                rs = [r for r in all_rows if not r["empty_original"]]
                values = [len(rs), sum(r["zero_h"] for r in rs),
                          sum(r["same_extent"] for r in rs),
                          sum(r["universal_h"] for r in rs)]
                rows.append(f"{title} & {short} & {letter} & "
                            + " & ".join(map(str, values)))
    result["closure_added_inventory_codexgen"] = table(
        "All 404 prior queries, conditioned on a nonempty original\n"
        "    extent. Proj. retains the original declared coordinates.\n"
        "    Columns overlap: zero $h$ is universal, and a universal\n"
        "    original extent can also be unchanged.",
        "tab:closure-added-inventory", "lllrrrr",
        r"Model & Space & Context & Nonempty & $h=0$"
        r" & Unchanged & Universal", rows)
    return result


def figures(reports: dict[str, list[dict]]) -> None:
    """Render final examples and a declared cross-collection follow-up.

    The follow-up displays every facial-pair match outside the two large
    pet collections, after initial notes. It tests the narrower initial
    cat/dog interpretation and is not presented as label-blind selection.
    """
    choices = [(0, "face6_q50", "patch", [431, 436, 54]),
               (1, "repetition2_q80", "patch", [505, 414, 359]),
               (0, "source_0_7_native_meet", "image", [76, 40, 85])]
    canvas = Image.new("RGB", (840, 3 * 345), "white")
    evidence_by_model = {}
    all_panels = {}
    for rownum, (model_index, name, context, ids) in enumerate(choices):
        model = MODELS[model_index]
        if model not in evidence_by_model:
            evidence_by_model[model] = load_evidence(model)
        evidence = evidence_by_model[model]
        r = next(r for r in reports[model] if r["name"] == name
                 and r["context"] == context and r["space"] == "native")
        vectors = sparse.load_npz(DEST / f"{model}_closed_codexgen.npz")
        h = vectors[r["vector_row"]].toarray()[0]
        q = read_results(model)[name]
        h[np.array(q["features"])[np.array(q["query"]) != 0]] = 0
        title = (f"{name}, {context}: {r['original']} -> {r['added']}; "
                 f"{r['support_h']} added coordinates")
        ImageDraw.Draw(canvas).text(
            (5, rownum * 345), title, fill="black",
            font=ImageFont.load_default(size=16))
        all_panels[name] = []
        for col, index in enumerate(ids):
            picture, record = panel(evidence, h, index, False, context)
            assert record["common_margin" if context == "patch"
                          else "pooled_margin"] >= 1
            all_panels[name].append(record)
            canvas.paste(picture, (col * 280, rownum * 345 + 30))
        if rownum != 0:
            continue
        with np.load(DEST / f"{model}_masks_codexgen.npz") as masks:
            found = np.flatnonzero(masks[f"{r['index']}_common"])
            original = masks[f"{r['index']}_original"].reshape(547, -1).any(1)
        found = found[(found < 100) | (found >= 400)]
        all_panels["facial_cross_collection_followup"] = []
        for start in range(0, len(found), 18):
            block = found[start:start + 18]
            sheet = Image.new("RGB", (840, ((len(block) + 2) // 3) * 312))
            for number, index in enumerate(block):
                picture, record = panel(evidence, h, int(index),
                                         bool(original[index]), context)
                all_panels["facial_cross_collection_followup"].append(record)
                sheet.paste(picture, ((number % 3) * 280,
                                     (number // 3) * 312))
            sheet.save(DEST / "figures" / f"face_transfer_{start // 18}.png")
    canvas.save(DEST / "figures/closure_added_examples_codexgen.png")
    (DEST / "final_panels_codexgen.json").write_text(
        json.dumps(all_panels, indent=2) + "\n")


def main() -> None:
    """Write generated reports and optionally export manuscript artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path)
    args = parser.parse_args()
    reports = {m: json.loads((DEST / f"{m}_codexgen.json").read_text())[
        "results"] for m in MODELS}
    figures(reports)
    if args.paper:
        for name, source in tables(reports).items():
            (args.paper / f"tables/{name}.tex").write_text(source)
        shutil.copy2(DEST / "figures/closure_added_examples_codexgen.png",
                     args.paper / "figures")


if __name__ == "__main__":
    main()
