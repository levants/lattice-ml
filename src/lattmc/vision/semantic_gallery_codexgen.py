"""Render label-withheld feature atlases and reproducible query witnesses.

Boxes identify cached token sites, not isolated receptive fields. Panels
retain the whole image; a small inset magnifies the selected token cell.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lattmc.vision.semantic_cache_codexgen import (
    FIGURES, MODELS, Evidence, FloatArray, load_evidence,
)


def tile(pixels: np.ndarray, site: int, grid: int,
         title: str, separate_sites: list[int] | None = None
         ) -> Image.Image:
    """Render an RGB crop, its token cell, and a numerical caption.

    The input is a uint8 (224, 224, 3) array. Site indices are row-major;
    the lower-right inset displays only the indicated cell, enlarged.
    """
    result = Image.new("RGB", (280, 308), "white")
    view = Image.fromarray(pixels).resize((280, 280))
    draw = ImageDraw.Draw(view)
    y, x = divmod(site, grid)
    left, top = x * 280 / grid, y * 280 / grid
    right, bottom = (x + 1) * 280 / grid, (y + 1) * 280 / grid
    draw.rectangle((left, top, right, bottom), outline="#ff3300", width=2)
    for number, position in enumerate(separate_sites or []):
        sy, sx = divmod(position, grid)
        box = (sx * 280 / grid, sy * 280 / grid,
               (sx + 1) * 280 / grid, (sy + 1) * 280 / grid)
        draw.rectangle(box, outline="#00bfff", width=2)
        draw.text((box[0] + 1, box[1] + 1), str(number + 1),
                  fill="#ff0055", font=ImageFont.load_default(size=12))
    crop = Image.fromarray(pixels).crop((
        x * 224 // grid, y * 224 // grid,
        (x + 1) * 224 // grid, (y + 1) * 224 // grid))
    view.paste(crop.resize((60, 60)), (218, 218))
    ImageDraw.Draw(view).rectangle((217, 217, 279, 279),
                                   outline="#ff3300", width=2)
    result.paste(view, (0, 0))
    ImageDraw.Draw(result).text((3, 283), title, fill="black",
                               font=ImageFont.load_default(size=14))
    return result


def atlas(evidence: Evidence) -> None:
    """Save three feature atlases without dataset names or class labels.

    Each feature shows the two largest image maxima and the median-ranked
    positive image, with stable row-order ties. This is an exploratory
    display of the existing training-selected 24 coordinates.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    z = evidence.codes
    grid = round(z.shape[1] ** 0.5)
    for page in range(3):
        canvas = Image.new("RGB", (840, 8 * 340 + 35), "#dddddd")
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 5), f"{evidence.model}; page {page + 1}",
                  fill="black", font=ImageFont.load_default(size=18))
        for row in range(8):
            j = page * 8 + row
            values = z[:, :, j].max(1)
            order = np.argsort(-values, kind="stable")
            order = order[values[order] > 0]
            choices = [order[0], order[min(1, len(order) - 1)],
                       order[len(order) // 2]]
            draw.text((5, 35 + row * 340),
                      f"f={evidence.features[j]}; positive={len(order)}",
                      fill="black", font=ImageFont.load_default(size=16))
            for col, index in enumerate(choices):
                site = int(z[index, :, j].argmax())
                caption = f"I{index:03d} p{site} a={values[index]:.4f}"
                panel = tile(evidence.images[index], site, grid, caption)
                canvas.paste(panel, (col * 280, row * 340 + 62))
        canvas.save(FIGURES / f"{evidence.model}_atlas_{page}.png")


def sources(evidence: Evidence) -> None:
    """Show 12 fixed random training images and their peak-norm tokens.

    The random seed and numerical selection are fixed before visual
    interpretation. The same image indices are used for every dictionary.
    """
    ids = np.random.default_rng(20261003).choice(200, 12, replace=False)
    canvas = Image.new("RGB", (4 * 280, 3 * 308), "white")
    grid = round(evidence.training.shape[1] ** 0.5)
    for number, index in enumerate(ids):
        site = int(np.linalg.norm(evidence.training[index], axis=1).argmax())
        title = f"S{number} train row {evidence.training_rows[index]} p{site}"
        panel = tile(evidence.training_images[index], site, grid, title)
        canvas.paste(panel, ((number % 4) * 280, (number // 4) * 308))
    canvas.save(FIGURES / f"{evidence.model}_sources.png")


def query_gallery(evidence: Evidence, codes: FloatArray,
                  query: FloatArray, name: str,
                  chosen: list[int] | None = None) -> list[dict]:
    """Save strongest, boundary, and distributed matches for a fixed query.

    Parameters are projected codes (N, P, J), positive thresholds (J),
    and an optional explicit list of global image IDs. Return the exact
    witness values for the panels. Empty extents are rendered explicitly.
    """
    positive = query > 0
    ratios = codes[:, :, positive] / query[positive]
    if not positive.any():
        ratios = np.ones((*codes.shape[:2], 1))
    common = ratios.min(2).max(1)
    pooled = ratios.max(1).min(1)
    if chosen is None:
        strong = np.argsort(-common, kind="stable")
        strong = strong[common[strong] >= 1]
        boundary = strong[-2:][::-1]
        distributed = np.flatnonzero((pooled >= 1) & (common < 1))
        chosen = list(dict.fromkeys(np.concatenate(
            [strong[:4], boundary, distributed[:2]]).tolist()))
    canvas = Image.new("RGB", (4 * 280, 2 * 308 + 40), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), f"{name}: G={sum(pooled >= 1)}, H={sum(common >= 1)}",
              fill="black", font=ImageFont.load_default(size=19))
    records = []
    grid = round(codes.shape[1] ** 0.5)
    for number, index in enumerate(chosen[:8]):
        separate = (ratios[index].argmax(0).tolist()
                    if common[index] < 1 else None)
        site = (int(separate[0]) if separate
                else int(ratios[index].min(1).argmax()))
        title = (f"I{index:03d} p{site} H={common[index]:.3f} "
                 f"G={pooled[index]:.3f}")
        panel = tile(evidence.images[index], site, grid, title, separate)
        canvas.paste(panel, ((number % 4) * 280,
                            (number // 4) * 308 + 40))
        records.append({"image": index, "site": site,
                        "values": codes[index, site].tolist(),
                        "maxima": codes[index].max(0).tolist(),
                        "argmax_sites": codes[index].argmax(0).tolist(),
                        "common_margin": float(common[index]),
                        "pooled_margin": float(pooled[index])})
    if not chosen:
        draw.text((20, 80), "Empty extent: no representative match.",
                  fill="black", font=ImageFont.load_default(size=22))
    canvas.save(FIGURES / f"{name}.png")
    return records


def extent_sheets(evidence: Evidence, codes: FloatArray,
                  query: FloatArray, name: str, mode: str = "common"
                  ) -> list[int]:
    """Render all small extents or 36 ranked quantiles of a larger extent.

    Extents of at most 80 images are shown in full. Larger extents use
    equally spaced ranks, including both endpoints, without labels.
    Three-column pages retain readable whole-image context.
    """
    positive = query > 0
    ratios = codes[:, :, positive] / query[positive]
    if not positive.any():
        ratios = np.ones((*codes.shape[:2], 1))
    score = (ratios.min(2).max(1) if mode == "common"
             else ratios.max(1).min(1))
    ids = np.argsort(-score, kind="stable")
    ids = ids[score[ids] >= 1]
    if len(ids) > 80:
        ids = ids[np.linspace(0, len(ids) - 1, 36).astype(int)]
    grid = round(codes.shape[1] ** 0.5)
    for start in range(0, len(ids), 18):
        block = ids[start:start + 18]
        rows = (len(block) + 2) // 3
        canvas = Image.new("RGB", (840, rows * 308 + 30), "white")
        ImageDraw.Draw(canvas).text(
            (5, 3), f"{name} {mode}: {start + 1}-{start + len(block)}",
            fill="black", font=ImageFont.load_default(size=16))
        for number, index in enumerate(block):
            separate = (ratios[index].argmax(0).tolist()
                        if ratios[index].min(1).max() < 1 else None)
            site = (int(separate[0]) if separate
                    else int(ratios[index].min(1).argmax()))
            caption = f"I{index:03d} p{site} margin={score[index]:.4f}"
            panel = tile(evidence.images[index], site, grid, caption,
                         separate)
            canvas.paste(panel, ((number % 3) * 280,
                                (number // 3) * 308 + 30))
        canvas.save(FIGURES / f"{name}_{mode}_{start // 18}.png")
    return ids.tolist()


def main() -> None:
    """Generate label-withheld atlases for existing dictionaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    for model in ([args.model] if args.model else MODELS):
        evidence = load_evidence(model)
        atlas(evidence)
        sources(evidence)
        print(model, "label-withheld atlases saved", flush=True)


if __name__ == "__main__":
    main()
