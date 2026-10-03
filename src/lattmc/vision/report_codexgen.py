"""Generate paper tables and scientific figures from saved pilot results."""

from __future__ import annotations
from collections.abc import Sequence

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lattmc.vision.paths_codexgen import experiment_root, load_digit_cache


def table(
    path: Path,
    caption: str,
    label: str,
    headings: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    """Write a captioned LaTeX results table."""
    lines = [r"\begin{table}[tbp]", r"\centering", r"\small",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{" + "l" + "r" * (len(headings) - 1) + "}",
             r"\toprule", " & ".join(headings) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def generate(paper: Path) -> None:
    """Generate the digit pilot's paper tables from cached results."""
    paper = Path(paper)
    folder = experiment_root()
    result_path = folder / "results/results_codexgen.json"
    report = json.loads(result_path.read_text())
    seeds = report["seeds"]
    records = report["retrieval"]
    table(paper / "tables/reconstruction.tex",
          "Held-out classifier and surrogate diagnostics.",
          "tab:reconstruction",
          ["Seed", "Accuracy", "Replaced", r"$R^{2}_{\rm tr}$",
           "Site nnz", "Image nnz", "Alive"],
          [[r["seed"], f'{100*r["accuracy"]:.2f}',
            f'{100*r["reconstructed_accuracy"]:.2f}',
            f'{r["reconstruction_r2_train_mean"]:.3f}',
            f'{r["site_l0"]:.2f}', f'{r["image_l0"]:.2f}',
            r["alive_training_latents"]] for r in report["models"]])
    methods = [("Graded conjunction", "graded_ap"),
               ("Presence baseline", "binary_ap"),
               ("Sparse-code cosine", "sae_cosine_ap"),
               ("Dense-code cosine", "dense_cosine_ap"),
               ("Random sources", "random_sources_ap")]
    values = np.array([[100 * np.mean([r[key] for r in records
                                      if r["seed"] == seed])
                        for seed in seeds] for _, key in methods])
    table(paper / "tables/retrieval.tex",
          "Mean test average precision (percent), 50 queries per seed.",
          "tab:retrieval", ["Method"] + [str(s) for s in seeds] + ["Mean"],
          [[name] + [f"{v:.2f}" for v in row] + [f"{row.mean():.2f}"]
           for (name, _), row in zip(methods, values)])
    spatial = []
    for seed in seeds:
        rows = [r for r in records if r["seed"] == seed]
        delta = np.mean([1 - r["same_site_extent"] / r["test_extent"]
                         for r in rows if r["test_extent"]])
        spatial.append([
            seed, f'{np.mean([r["graded_f1"] for r in rows]):.3f}',
            f'{np.mean([r["binary_f1"] for r in rows]):.3f}',
            sum(r["test_extent"] for r in rows),
            sum(r["same_site_extent"] for r in rows), f"{delta:.3f}",
        ])
    table(paper / "tables/spatial.tex",
          "Thresholded retrieval and spatial witnesses on test images.",
          "tab:spatial", ["Seed", r"Graded $F_{1}$", r"Binary $F_{1}$",
                          "Pooled", "Same-site", r"Mean $\Delta$"], spatial)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), layout="constrained")
    for i, (name, _) in enumerate(methods):
        axes[0].scatter(values[i], [i] * len(seeds), s=32)
    axes[0].set_yticks(range(len(methods)), [m[0] for m in methods])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean average precision (%)")
    axes[0].set_xlim(0, 100)
    axes[0].grid(axis="x", alpha=0.25)
    for seed in seeds:
        rows = [r for r in records if r["seed"] == seed]
        axes[1].scatter([r["active_coordinates"] for r in rows],
                        [1-r["same_site_extent"]/r["test_extent"]
                         for r in rows], label=str(seed), alpha=0.5, s=18)
    axes[1].set_xlabel("Positive coordinates in query")
    axes[1].set_ylabel("Witness discrepancy")
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].legend(title="Model seed", fontsize=8)
    fig.savefig(paper / "figures/retrieval_codexgen.pdf")
    plt.close(fig)
    # Exact constructed example: clearly distinguished from learned codes.
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.2), layout="constrained")
    arrays = [np.array([[2, 0], [0, 2]]), np.array([[2, 2]]),
              np.array([[False], [False]])]
    titles = ["Two site codes", "Pooled summary", "Same-site match"]
    for ax, array, title in zip(axes, arrays, titles):
        ax.imshow(array, cmap="Blues", vmin=0, vmax=2, aspect="auto")
        for index in np.ndindex(array.shape):
            ax.text(index[1], index[0], str(int(array[index])),
                    ha="center", va="center",
                    color="white" if array[index] > 1
                    else "black")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(paper / "figures/spatial_example_codexgen.pdf")
    plt.close(fig)

    cache = load_digit_cache(folder, 17)
    images = cache["images"]
    sources = cache["sources"][0]
    ranking = np.argsort(-cache["scores"][0, 0], kind="stable")[:3]
    selected = cache["test"][ranking]
    fig, axes = plt.subplots(2, 3, figsize=(7, 4), layout="constrained")
    for index, ax in enumerate(axes[0]):
        ax.imshow(images[sources[index]], cmap="gray_r", vmin=0, vmax=16)
        ax.set_title(f"Source {sources[index]}: digit 0", fontsize=10)
        ax.axis("off")
    for index, ax in enumerate(axes[1]):
        image_id = selected[index]
        label = int(cache["labels"][image_id])
        value = cache["scores"][0, 0, ranking[index]]
        ax.imshow(images[image_id], cmap="gray_r", vmin=0, vmax=16)
        ax.set_title(f"Test {image_id}: digit {label}; score {value:.2f}",
                     fontsize=9)
        ax.axis("off")
    fig.savefig(paper / "figures/image_retrieval_codexgen.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    generate(parser.parse_args().paper)
