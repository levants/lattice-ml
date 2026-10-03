"""Reproduce the offline digits pilot and export its complete evidence."""

from __future__ import annotations
from typing import Any

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

from lattmc.vision.audit_codexgen import audit
from lattmc.vision.contexts_codexgen import (
    VectorContext,
    graded_score,
    select_query,
    spatial_extents,
)
from lattmc.vision.models_codexgen import DigitCNN, TopKSAE
from lattmc.vision.paths_codexgen import prepare_folders


def cosine(rows: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Compute cosine similarity with a nonzero denominator floor."""
    denominator = np.linalg.norm(rows, axis=1) * np.linalg.norm(source)
    return rows @ source / np.maximum(denominator, 1e-12)


def fit_models(
    images: torch.Tensor,
    labels: torch.Tensor,
    train: np.ndarray,
    seed: int,
) -> (
    tuple[DigitCNN, TopKSAE, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray]
):
    """Train the digit model and surrogate and return their cached outputs."""
    torch.manual_seed(seed)
    model = DigitCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(25):
        order = torch.tensor(train)[torch.randperm(len(train))]
        for indices in order.split(64):
            loss = nn.functional.cross_entropy(model(images[indices]),
                                               labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        hidden = model.features(images)
        sites = hidden.permute(0, 2, 3, 1).reshape(-1, 16, 32)
    training = sites[train].reshape(-1, 32)
    surrogate = TopKSAE()
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.001)
    for _ in range(20):
        for indices in torch.randperm(len(training)).split(512):
            batch = training[indices]
            loss = nn.functional.mse_loss(surrogate(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            surrogate.normalize_decoder()
    surrogate.eval()
    with torch.no_grad():
        codes = surrogate.encode(sites)
        reconstructed = surrogate.decoder(codes)
        restored = reconstructed.reshape(-1, 4, 4, 32).permute(0, 3, 1, 2)
        logits = model.head(hidden.flatten(1))
        reconstructed_logits = model.head(restored.flatten(1))
    return (model, surrogate, sites.numpy(), codes.numpy(),
            reconstructed.numpy(), logits.numpy(),
            reconstructed_logits.numpy())


def benchmark(
    patches: np.ndarray,
    dense: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate retrieval methods and collect held-out query results."""
    pooled = patches.max(axis=1).astype(np.float64)
    dense = dense.max(axis=1).astype(np.float64)
    scale = np.sqrt(np.mean(pooled[train] ** 2, axis=0))
    rng = np.random.default_rng(seed + 1000)
    rows, saved_queries, source_ids, scores = [], [], [], []
    context = VectorContext(pooled[train])
    for digit in range(10):
        positive = train[labels[train] == digit]
        truth_cal = labels[calibration] == digit
        truth_test = labels[test] == digit
        for draw in range(5):
            sources = rng.choice(positive, 3, replace=False)
            candidates = []
            support_candidates = []
            for budget in (1, 4, 16, 128):
                for alpha in (0.25, 0.5, 0.75, 1.0):
                    query = select_query(pooled[sources], scale, budget, alpha)
                    cal_score = graded_score(pooled[calibration], query)
                    quality = f1_score(truth_cal, cal_score >= 1)
                    candidates.append((quality, query, budget, alpha))
                query = select_query(pooled[sources], scale, budget, 1.0)
                active = query > 0
                if active.any():
                    support = (pooled[calibration][:, active] > 0).mean(axis=1)
                else:
                    support = np.ones(len(calibration))
                quality = f1_score(truth_cal, support == 1)
                support_candidates.append((quality, active, budget))
            # Stable tie break: first budget and alpha in the declared grid.
            _, query, budget, alpha = max(candidates, key=lambda item: item[0])
            _, active, binary_budget = max(support_candidates,
                                          key=lambda item: item[0])
            graded = graded_score(pooled[test], query)
            binary = ((pooled[test][:, active] > 0).mean(axis=1)
                      if active.any() else np.ones(len(test)))
            sae_cos = cosine(pooled[test], pooled[sources].mean(axis=0))
            dense_cos = cosine(dense[test], dense[sources].mean(axis=0))
            random_sources = rng.choice(train, 3, replace=False)
            random_query = select_query(pooled[random_sources], scale,
                                        budget, alpha)
            random_score = graded_score(pooled[test], random_query)
            image, same_site = spatial_extents(patches[test], query)
            selected_count = int(image.sum())
            row = {
                "seed": seed, "digit": digit, "draw": draw,
                "budget": budget, "alpha": alpha,
                "active_coordinates": int((query > 0).sum()),
                "binary_budget": binary_budget,
                "test_extent": selected_count,
                "same_site_extent": int(same_site.sum()),
                "graded_f1": float(f1_score(truth_test, image)),
                "binary_f1": float(f1_score(truth_test, binary == 1)),
                "closure_preserved": bool(np.array_equal(
                    context.extent(query),
                    context.extent(context.close_query(query)))),
            }
            methods = (graded, binary, sae_cos, dense_cos, random_score)
            names = ("graded", "binary", "sae_cosine", "dense_cosine",
                     "random_sources")
            for name, values in zip(names, methods):
                row[name + "_ap"] = float(
                    average_precision_score(truth_test, values))
            rows.append(row)
            saved_queries.append(query)
            source_ids.append(sources)
            scores.append(methods)
    return (rows, np.array(saved_queries), np.array(source_ids),
            np.array(scores))


def run(output: Path) -> dict[str, Any]:
    """Train and evaluate the offline digit pilot and save its artifacts."""
    output = Path(output)
    prepare_folders(output)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    digits = load_digits()
    labels = digits.target
    indices = np.arange(len(labels))
    train, remaining = train_test_split(indices, test_size=0.4,
                                        stratify=labels, random_state=2026)
    calibration, test = train_test_split(remaining, test_size=0.5,
                                        stratify=labels[remaining],
                                        random_state=2026)
    images = torch.tensor(digits.images[:, None] / 16, dtype=torch.float32)
    target = torch.tensor(labels, dtype=torch.long)
    np.savez_compressed(
        output / "dataset/digits_codexgen.npz", images=digits.images,
        data=digits.data, labels=labels, train=train,
        calibration=calibration, test=test)
    report = {
        "scope": "Offline digits pilot; no foundation model evaluated",
        "seeds": [17, 29, 43], "split_seed": 2026,
        "split_sizes": [len(train), len(calibration), len(test)],
        "cnn_epochs": 25, "sae_epochs": 20,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "sklearn": sklearn.__version__,
        "data_sha256": hashlib.sha256(digits.data.tobytes()
                                       + labels.tobytes()).hexdigest(),
        "audit": audit(), "models": [], "retrieval": [],
    }
    for seed in report["seeds"]:
        print(f"Training model and surrogate for seed {seed}", flush=True)
        model, sae, hidden, patches, restored, logits, rec_logits = fit_models(
            images, target, train, seed)
        test_hidden = hidden[test]
        error = np.sum((restored[test] - test_hidden) ** 2)
        mean = hidden[train].reshape(-1, 32).mean(axis=0)
        total = np.sum((test_hidden - mean) ** 2)
        metrics = {
            "seed": seed,
            "accuracy": float((logits[test].argmax(1) == labels[test]).mean()),
            "reconstructed_accuracy": float(
                (rec_logits[test].argmax(1) == labels[test]).mean()),
            "reconstruction_r2_train_mean": float(1 - error / total),
            "site_l0": float((patches[test] > 0).sum(axis=2).mean()),
            "image_l0": float((patches[test].max(1) > 0).sum(axis=1).mean()),
            "alive_training_latents": int(
                (patches[train].max(axis=(0, 1)) > 0).sum()),
        }
        rows, queries, sources, scores = benchmark(
            patches, hidden, labels, train, calibration, test, seed)
        report["models"].append(metrics)
        report["retrieval"].extend(rows)
        torch.save({"cnn": model.state_dict(), "sae": sae.state_dict()},
                   output / "checkpoints" / f"digits_seed_{seed}_codexgen.pt")
        np.savez_compressed(
            output / "activations" / f"digits_seed_{seed}_codexgen.npz",
            patches=patches, dense=hidden)
        np.savez_compressed(
            output / "retrieval" / f"digits_seed_{seed}_codexgen.npz",
            queries=queries, sources=sources, scores=scores)
        print(metrics, flush=True)
    artifacts = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.parent.name == "results":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts[str(path.relative_to(output))] = digest
    report["artifact_sha256"] = artifacts
    (output / "results/results_codexgen.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output)
