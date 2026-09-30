"""Activation-grounded galleries, tuning profiles, and lattice queries."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

from lattmc.vision.contexts_codexgen import graded_score, spatial_extents
from lattmc.vision.natural_codexgen import CLASSES
from lattmc.vision.paths_codexgen import experiment_root, repository_root


ASSOCIATIONS = (2, 3, 5, 7, 8, 9)


def make_examples(paper=None):
    folder = experiment_root("cifar10_resnet34")
    if paper is None:
        paper = repository_root() / "texs/sparsesurrs/visionlattices"
    paper = Path(paper)
    dataset = np.load(folder / "dataset/cifar10_sample_codexgen.npz")
    activations = np.load(folder / "activations/resnet34_codes_codexgen.npz")
    patches = activations["patches"]
    pooled = patches.max(axis=1).astype(np.float64)
    images, labels = dataset["images"], dataset["labels"]
    train, test = dataset["train"], dataset["test"]
    spread = np.maximum(pooled[train].std(axis=0), 1e-12)
    scale = np.maximum(np.sqrt((pooled[train] ** 2).mean(axis=0)), 1e-12)
    records, chosen = [], []
    for target in ASSOCIATIONS:
        positive = train[labels[train] == target]
        negative = train[labels[train] != target]
        contrast = (pooled[positive].mean(0)
                    - pooled[negative].mean(0)) / spread
        ordered = np.argsort(-contrast, kind="stable")
        feature = next(int(j) for j in ordered if int(j) not in chosen)
        chosen.append(feature)
        order = np.argsort(-pooled[test, feature], kind="stable")
        top_rows = test[order[:3]]
        records.append({
            "class": CLASSES[target], "class_id": target, "feature": feature,
            "train_contrast": float(contrast[feature]),
            "test_ap": float(average_precision_score(
                labels[test] == target, pooled[test, feature])),
            "top_rows": top_rows.tolist(),
            "original_test_indices":
                dataset["original_index"][top_rows].tolist(),
            "top_labels": [CLASSES[i] for i in labels[top_rows]],
            "top_activations": pooled[top_rows, feature].tolist(),
        })
    for group, start in (("animals", 0), ("transport", 3)):
        fig, axes = plt.subplots(3, 6, figsize=(12, 6.6),
                                 layout="constrained")
        for row, record in enumerate(records[start:start + 3]):
            feature = record["feature"]
            vmax = float(patches[test, :, feature].max())
            for column, image_id in enumerate(record["top_rows"]):
                raw, heat = axes[row, 2 * column:2 * column + 2]
                raw.imshow(images[image_id], interpolation="nearest")
                raw.set_title(
                    f'{CLASSES[labels[image_id]]} '
                    f'#{dataset["original_index"][image_id]}', fontsize=12)
                heat.imshow(patches[image_id, :, feature].reshape(6, 6),
                            cmap="magma", vmin=0, vmax=vmax,
                            interpolation="nearest")
                heat.set_title(f'max {pooled[image_id, feature]:.2f}',
                               fontsize=12)
                raw.set_xticks([])
                raw.set_yticks([])
                heat.set_xticks([])
                heat.set_yticks([])
                if column == 0:
                    raw.set_ylabel(f'F{feature}\n{record["class"]} selection',
                                   fontsize=13)
        fig.savefig(paper / f"figures/feature_{group}_codexgen.pdf")
        plt.close(fig)
    tuning = np.array([[pooled[test[labels[test] == c], feature].mean()
                        / scale[feature] for c in range(10)]
                       for feature in chosen])
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    picture = ax.imshow(tuning, cmap="viridis", aspect="auto")
    ax.set_xticks(range(10), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(6), [f'F{r["feature"]} ({r["class"]})'
                           for r in records])
    for row in range(6):
        for col in range(10):
            ax.text(col, row, f"{tuning[row, col]:.2f}", ha="center",
                    va="center", fontsize=10,
                    color="white" if tuning[row, col] < tuning.max()/2
                    else "black")
    fig.colorbar(picture, ax=ax, label="Mean activation / training RMS")
    fig.savefig(paper / "figures/feature_tuning_codexgen.pdf")
    plt.close(fig)
    # Chosen before inspecting test images: cat/dog and ship/truck.
    cases, arrays = [], {}
    for first, second in ((1, 2), (4, 5)):
        pair = [records[first], records[second]]
        source_rows = []
        for record in pair:
            candidates = train[labels[train] == record["class_id"]]
            source_rows.append(int(candidates[np.argmax(
                pooled[candidates, record["feature"]])]))
        coordinates = [r["feature"] for r in pair]
        queries = []
        for source in source_rows:
            query = np.zeros(pooled.shape[1])
            query[coordinates] = 0.5 * pooled[source, coordinates]
            queries.append(query)
        u, v = queries
        name = pair[0]["class"] + "_" + pair[1]["class"]
        queries += [np.minimum(u, v), np.maximum(u, v)]
        case = {"name": name, "source_rows": source_rows,
                "coordinates": coordinates, "operations": []}
        arrays[name + "_queries"] = np.array(queries)
        fig, axes = plt.subplots(4, 4, figsize=(8, 7.6),
                                 layout="constrained")
        for index, (operation, query) in enumerate(zip(
                ("u", "v", "meet", "join"), queries)):
            extent, same_site = spatial_extents(patches[test], query)
            score = graded_score(pooled[test], query)
            ranking = np.argsort(-score, kind="stable")
            selected = [int(i) for i in ranking if extent[i]][:3]
            item = {"operation": operation,
                    "thresholds": query[coordinates].tolist(),
                    "extent_size": int(extent.sum()),
                    "same_site_size": int(same_site.sum()),
                    "class_counts": np.bincount(labels[test[extent]],
                                                minlength=10).tolist(),
                    "display_rows": test[selected].tolist()}
            case["operations"].append(item)
            arrays[name + "_" + operation + "_extent"] = extent
            info = axes[index, 0]
            info.axis("off")
            thresholds = ", ".join(f"{x:.2f}" for x in query[coordinates])
            info.text(0, 0.6,
                      f"{operation}: ({thresholds})\n"
                      f"pooled = {extent.sum()}\n"
                      f"same site = {same_site.sum()}", fontsize=13)
            for column, ax in enumerate(axes[index, 1:]):
                ax.axis("off")
                if column >= len(selected):
                    ax.text(0.5, 0.5, "No further match", ha="center")
                    continue
                image_id = int(test[selected[column]])
                ax.imshow(images[image_id], interpolation="nearest")
                ax.set_title(f'{CLASSES[labels[image_id]]} '
                             f'#{dataset["original_index"][image_id]}',
                             fontsize=12)
        fig.savefig(paper / f"figures/query_{name}_codexgen.pdf")
        plt.close(fig)
        assert np.array_equal(arrays[name + "_join_extent"],
                              arrays[name + "_u_extent"]
                              & arrays[name + "_v_extent"])
        assert np.all(~(arrays[name + "_u_extent"]
                        | arrays[name + "_v_extent"])
                      | arrays[name + "_meet_extent"])
        cases.append(case)
    np.savez_compressed(folder / "retrieval/visual_queries_codexgen.npz",
                        **arrays)
    result = {"selection": "training standardized class contrast, unique",
              "feature_examples": records, "test_tuning": tuning.tolist(),
              "lattice_cases": cases}
    (folder / "results/visual_examples_codexgen.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path)
    make_examples(parser.parse_args().paper)
