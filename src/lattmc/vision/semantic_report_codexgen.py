"""Export measured semantic-reading tables and selected visual evidence.

The images are cached observations, never generated activation art.
Optional paper export copies only final selected figures and numeric TeX
tables; ordinary experiment outputs remain independent of the manuscript.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lattmc.vision.semantic_cache_codexgen import (
    FIGURES, MODELS, OUT, load_evidence, projected_codes,
)
from lattmc.vision.semantic_gallery_codexgen import tile


def read_results(model: str) -> dict[str, dict]:
    """Index the fixed and exploratory query records by stable name."""
    rows = []
    for suffix in ("", "_followup"):
        path = OUT / f"{model}{suffix}_codexgen.json"
        rows.extend(json.loads(path.read_text())["results"])
    return {row["name"]: row for row in rows}


def visual_evidence() -> None:
    """Render three explicitly selected comparison figures with witnesses."""
    cases = [("topk_k32_s0", "face6_q50", [426, 293, 205]),
             ("topk_k32_s0", "performance3_q80", [52, 53, 54]),
             ("prisma_transcoder", "repetition2_q80", [495, 413, 441])]
    canvas = Image.new("RGB", (840, 3 * 338), "white")
    evidence = None
    for row, (model, name, choices) in enumerate(cases):
        if evidence is None or evidence.model != model:
            evidence = load_evidence(model)
        result = read_results(model)[name]
        z = projected_codes(evidence, np.array(result["features"]))
        q = np.array(result["query"])
        grid = round(z.shape[1] ** .5)
        text = (f"{name}: {result['patch_count']} tokens, "
                f"H={result['common']}, G={result['pooled']}")
        ImageDraw.Draw(canvas).text((5, row * 338), text, fill="black",
                                   font=ImageFont.load_default(size=18))
        for col, index in enumerate(choices):
            ratios = z[index] / q
            h = float(ratios.min(1).max())
            g = float(ratios.max(0).min())
            sites = ratios.argmax(0).tolist() if h < 1 else None
            site = sites[0] if sites else int(ratios.min(1).argmax())
            panel = tile(evidence.images[index], site, grid,
                         f"I{index:03d} H={h:.3f}, G={g:.3f}", sites)
            canvas.paste(panel, (col * 280, row * 338 + 28))
    canvas.save(FIGURES / "semantic_witnesses_codexgen.png")
    evidence = load_evidence("topk_k32_s0")
    result = read_results(evidence.model)["source_0_7_native_meet"]
    canvas = Image.new("RGB", (840, 3 * 308), "white")
    for number, record in enumerate(result["sources"]):
        panel = tile(evidence.training_images[record["training_index"]],
                     record["site"], 16,
                     f"S{record['source']}: row {record['row']} "
                     f"p{record['site']}")
        canvas.paste(panel, (number * 280, 0))
    choices = [51, 52, 204, 546, 55, 162, 80]
    values = projected_codes(evidence, np.array([825]))[:, :, 0]
    threshold = max(result["query"])
    for number, index in enumerate(choices, 2):
        site = int(values[index].argmax())
        panel = tile(evidence.images[index], site, 16,
                     f"I{index:03d} a={values[index, site]:.4f} "
                     f"r={values[index, site] / threshold:.3f}")
        canvas.paste(panel, ((number % 3) * 280, (number // 3) * 308))
    canvas.save(FIGURES / "semantic_source_meet_codexgen.png")
    source = FIGURES / "topk_k32_s0_structure2_q50_common_0.png"
    shutil.copy2(source, FIGURES / "semantic_counterexamples_codexgen.png")


def table(header: str, label: str, columns: str,
          heading: str, rows: list[str]) -> str:
    """Build a small labeled booktabs table from already formatted rows."""
    return "\n".join([
        r"\begin{table}[tbp]", r"  \centering\small",
        "  \\caption{" + header + "}", "  \\label{" + label + "}",
        "  \\begin{tabular}{" + columns + "}", r"    \toprule",
        "    " + heading + r"\\", r"    \midrule",
        *["    " + row + r"\\" for row in rows],
        r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])


def export_tables() -> dict[str, str]:
    """Create case, arity, and fixed-inventory tables from saved results."""
    results = {model: read_results(model) for model in MODELS}
    rows = []
    for name in ("face2", "face3", "face4", "face6"):
        a = results[MODELS[0]][f"{name}_q50"]
        b = results[MODELS[0]][f"{name}_q80"]
        size = len(a["features"])
        rows.append(f"{size} & {a['pooled']} & {a['common']} & "
                    f"{a['patch_count']} & {b['pooled']} & "
                    f"{b['common']} & {b['patch_count']}")
    output = {"semantic_arity_codexgen": table(
        "Nested facial queries at positive training quantiles .5 and .8.\n"
        "    Counts refer to 547 images; $T$ counts returned patch tokens.",
        "tab:semantic-arity", "rrrrrrr",
        r"Coordinates & $G_{.5}$ & $H_{.5}$ & $T_{.5}$"
        r" & $G_{.8}$ & $H_{.8}$ & $T_{.8}$", rows)}
    rows = []
    names = [(MODELS[0], "structure2_q50", "TopK structural pair"),
             (MODELS[0], "contrast2_q50", "TopK contrasting pair"),
             (MODELS[0], "performance3_q80", "TopK performance triple"),
             (MODELS[1], "repetition2_q80", "TC texture pair"),
             (MODELS[1], "pattern3_q80", "TC pattern triple"),
             (MODELS[2], "person_scene3_q50", "RA person--scene triple"),
             (MODELS[2], "contrast2_q50", "RA contrasting pair")]
    for model, name, title in names:
        r = results[model][name]
        singles = ",".join(map(str, r["component_pooled"]))
        rows.append(f"{title} & {singles} & {r['pooled']} & "
                    f"{r['common']} & {r['patch_count']}")
    output["semantic_cases_codexgen"] = table(
        "Selected follow-ups and their single-coordinate controls.\n"
        "    Names describe exploratory readings, not validated detectors.",
        "tab:semantic-cases", "llrrr",
        r"Query & Individual $|G|$ & $|G|$ & $|H|$ & $|T|$", rows)
    rows = []
    for model, title in zip(MODELS, ("TopK", "Transcoder", "RA-SAE")):
        path = OUT / f"{model}_codexgen.json"
        records = json.loads(path.read_text())["results"]
        with np.load(OUT / f"{model}_masks_codexgen.npz") as arrays:
            unique = [len({arrays[r["name"] + "__" + key].tobytes()
                           for r in records}) for key in ("pooled", "common")]
        counts = [sum(r[key] == value for r in records)
                  for value in (0, 547) for key in ("pooled", "common")]
        rows.append(title + " & " + " & ".join(map(str, counts + unique)))
    output["semantic_inventory_codexgen"] = table(
        "Complete fixed inventory: 124 queries per dictionary.\n"
        "    Empty and universal extents recur; distinct queries can agree.",
        "tab:semantic-inventory", "lrrrrrr",
        r"Model & Empty $G$ & Empty $H$ & All $G$ & All $H$"
        r" & Distinct $G$ & Distinct $H$", rows)
    prefix = "source_0_1_2_3_4_5_6_7_native_meet"
    records = [results[model][prefix]["sources"] for model in MODELS]
    rows = [f"S{i} & {records[0][i]['row']} & "
            + " & ".join(str(r[i]["site"]) for r in records)
            for i in range(8)]
    output["semantic_sources_codexgen"] = table(
        "Exact training sources for the nested source-code operations.\n"
        "    Rows index the existing Imagenette cache; sites are zero-based.",
        "tab:semantic-sources", "lrrrr",
        "Source & Image row & TopK site & TC site & RA site", rows)
    rows = []
    for model, title in zip(MODELS, ("TopK", "TC", "RA")):
        for size in (2, 3, 4, 6, 8):
            name = "source_" + "_".join(map(str, range(size)))
            r = results[model][name + "_native_meet"]
            rows.append(f"{title} & {size} & {r['positive_coordinates']} & "
                        f"{r['pooled']} & {r['common']} & "
                        f"{r['patch_count']}")
    output["semantic_prefixes_codexgen"] = table(
        "All native source-prefix meets. Every corresponding native join\n"
        "    has an empty held-out extent. Zero meets return all tokens.",
        "tab:semantic-prefixes", "lrrrrr",
        r"Model & Sources & Positive & $|G|$ & $|H|$ & Tokens", rows)
    rows = []
    for model, title in zip(MODELS, ("TopK", "TC", "RA")):
        for size in (2, 3, 4, 6, 8):
            cells = []
            for family in ("coactive", "lowcorr", "random"):
                pair = [results[model][f"{family}_{size}_{i}_q80"]
                        for i in range(2)]
                ranges = []
                for key in ("pooled", "common"):
                    values = [r[key] for r in pair]
                    ranges.append(f"{min(values)}--{max(values)}")
                cells.append(" / ".join(ranges))
            rows.append(f"{title} & {size} & " + " & ".join(cells))
    output["semantic_controls_codexgen"] = table(
        "Range of image counts over two fixed combinations per design,\n"
        "    at upper training quantiles. Each cell gives $G$ / $H$ ranges;\n"
        "    these descriptive controls are not matched semantic trials.",
        "tab:semantic-controls", "lrlll",
        "Model & Size & Coactive & Low correlation & Random", rows)
    return output


def main() -> None:
    """Generate figures and optionally export selected manuscript assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path)
    args = parser.parse_args()
    visual_evidence()
    tables = export_tables()
    if args.paper:
        import textwrap

        for name, content in tables.items():
            lines = [textwrap.fill(line, 79, break_long_words=False,
                                   break_on_hyphens=False)
                     for line in content.splitlines()]
            (args.paper / "tables" / f"{name}.tex").write_text(
                "\n".join(lines) + "\n")
        for name in ("witnesses", "source_meet", "counterexamples"):
            filename = f"semantic_{name}_codexgen.png"
            shutil.copy2(FIGURES / filename, args.paper / "figures" / filename)


if __name__ == "__main__":
    main()
