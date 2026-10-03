"""Render anonymous outputs of closure-added feature queries.

Selection uses numerical ranks and earlier declared case names, not class
labels. Each panel keeps the full crop and identifies whether an image
was already returned. Patch boxes indicate contextualized token sites,
not receptive fields. Detailed activation evidence is saved alongside.
"""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse

from lattmc.vision.closure_added_codexgen import DEST
from lattmc.vision.semantic_cache_codexgen import (
    MODELS, Evidence, FloatArray, load_evidence,
)
from lattmc.vision.semantic_report_codexgen import read_results


CASES = {
    "topk_k32_s0": [
        ("face6_q50", "patch"), ("face3_q80", "patch"),
        ("performance3_q80", "image"),
        ("source_0_7_native_meet", "image"),
        ("contrast2_q50", "patch")],
    "prisma_transcoder": [("repetition2_q80", "patch"),
                          ("source_0_1_2_native_meet", "patch")],
    "pretrained_ra": [("person_scene3_q50", "image")],
}


def measurements(evidence: Evidence, h: FloatArray) -> tuple[
        FloatArray, FloatArray]:
    """Return common and pooled minimum ratios per image without labels."""
    n, sites = evidence.codes.shape[:2]
    active = np.flatnonzero(h)
    common, pooled = np.ones(n), np.ones(n)
    for i in range(n):
        z = evidence.native[i * sites:(i + 1) * sites, active].toarray()
        if len(active):
            ratio = z / h[active]
            common[i] = ratio.min(1).max()
            pooled[i] = ratio.max(0).min()
    return common, pooled


def panel(evidence: Evidence, h: FloatArray, index: int,
          old: bool, context: str) -> tuple[Image.Image, dict]:
    """Render exact witness positions and return full numerical evidence.

    Common witnesses have one red cell. Distributed image matches show up
    to four bottleneck-feature maxima in blue, without claiming those cells
    witness every coordinate. A zero h raises ValueError because no
    numerical witness should be fabricated for a universal zero query.
    """
    sites = evidence.codes.shape[1]
    grid = round(sites ** .5)
    active = np.flatnonzero(h)
    if not len(active):
        raise ValueError("A zero query has no feature-specific witness.")
    z = evidence.native[index * sites:(index + 1) * sites,
                        active].toarray().astype(np.float64)
    ratios = z / h[active]
    maxima = ratios.max(0)
    g, common = float(maxima.min()), float(ratios.min(1).max())
    position = int(ratios.min(1).argmax())
    args = ratios.argmax(0)
    distributed = common < 1
    marked_features = np.argsort(maxima, kind="stable")[:4]
    marks = (args[marked_features].tolist() if distributed else [position])
    view = Image.fromarray(evidence.images[index]).resize((280, 280))
    draw = ImageDraw.Draw(view)
    for number, site in enumerate(marks):
        y, x = divmod(site, grid)
        box = (x * 280 / grid, y * 280 / grid,
               (x + 1) * 280 / grid, (y + 1) * 280 / grid)
        draw.rectangle(box, outline="#008bff" if distributed else "red",
                       width=2)
        if distributed:
            draw.text((box[0], box[1]), str(number + 1), fill="yellow")
    canvas = Image.new("RGB", (280, 312), "white")
    canvas.paste(view)
    text = (f"I{index:03d} {'old' if old else 'new image'} "
            f"H={common:.3f} G={g:.3f}")
    ImageDraw.Draw(canvas).text((3, 283), text, fill="black",
                               font=ImageFont.load_default(size=13))
    record = {"image": index, "old_image": old, "context": context,
              "common_margin": common, "pooled_margin": g,
              "best_site": position, "features": active.tolist(),
              "thresholds": h[active].tolist(),
              "best_site_values": z[position].tolist(),
              "maxima": z.max(0).tolist(), "argmax_sites": args.tolist(),
              "marked_features": active[marked_features].tolist()
                  if distributed else active.tolist()}
    return canvas, record


def main() -> None:
    """Save complete small extents and rank samples of larger extents."""
    figures = DEST / "figures"
    figures.mkdir(exist_ok=True)
    inspections = {}
    for model in MODELS:
        evidence = load_evidence(model)
        report = json.loads((DEST / f"{model}_codexgen.json").read_text())
        originals = read_results(model)
        vectors = sparse.load_npz(DEST / f"{model}_closed_codexgen.npz")
        with np.load(DEST / f"{model}_masks_codexgen.npz") as saved:
            masks = {name: saved[name] for name in saved.files}
        for name, context in CASES[model]:
            row = next(r for r in report["results"] if r["name"] == name
                       and r["context"] == context and r["space"] == "native")
            h = vectors[row["vector_row"]].toarray()[0]
            q = originals[name]
            active_u = np.array(q["features"])[np.array(q["query"]) != 0]
            h[active_u] = 0
            assert h.any(), (name, context)
            common, pooled = measurements(evidence, h)
            score = common if context == "patch" else pooled
            original = masks[f"{row['index']}_original"]
            if context == "patch":
                original = original.reshape(len(score), -1).any(1)
            ids = np.argsort(-score, kind="stable")
            ids = ids[score[ids] >= 1]
            complete = len(ids) <= 80
            if not complete:
                chosen = ids[np.linspace(0, len(ids) - 1, 36).astype(int)]
                ids = np.array(list(dict.fromkeys(
                    np.concatenate([ids[original[ids]][:3], chosen])
                    .tolist())))
            stem = f"{model}_{name}_{context}"
            inspected = []
            for start in range(0, len(ids), 18):
                block = ids[start:start + 18]
                canvas = Image.new("RGB", (840, ((len(block) + 2) // 3)
                                          * 312 + 38), "white")
                title = (f"{stem}: {row['original']} -> {row['added']}; "
                         f"h has {row['support_h']} coordinates")
                ImageDraw.Draw(canvas).text(
                    (5, 5), title, fill="black",
                    font=ImageFont.load_default(size=14))
                for k, i in enumerate(block):
                    item, record = panel(evidence, h, int(i),
                                         bool(original[i]), context)
                    inspected.append(record)
                    canvas.paste(item, ((k % 3) * 280,
                                        (k // 3) * 312 + 38))
                canvas.save(figures / f"{stem}_{start // 18}.png")
            inspections[stem] = {
                "complete": complete, "result": row, "items": inspected}
    (DEST / "inspection_codexgen.json").write_text(
        json.dumps(inspections, indent=2) + "\n")


if __name__ == "__main__":
    main()
