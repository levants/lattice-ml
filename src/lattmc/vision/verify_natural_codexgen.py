"""Check the natural-image examples against models and stored arrays."""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from lattmc.vision.contexts_codexgen import spatial_extents
from lattmc.vision.models_codexgen import TopKSAE
from lattmc.vision.natural_codexgen import feature_model, preprocess, sha256
from lattmc.vision.paths_codexgen import experiment_root


def verify_natural(inference: bool = True) -> dict[str, int | bool]:
    """Verify natural-image artifacts and optionally recompute model outputs.
    """
    folder = experiment_root("cifar10_resnet34")
    report = json.loads((folder / "results/natural_codexgen.json").read_text())
    visual = json.loads((folder / "results/visual_examples_codexgen.json")
                        .read_text())
    for name, expected in report["artifact_sha256"].items():
        assert sha256(folder / name) == expected
    data = np.load(folder / "dataset/cifar10_sample_codexgen.npz")
    cache = np.load(folder / "activations/resnet34_codes_codexgen.npz")
    query_cache = np.load(folder / "retrieval/visual_queries_codexgen.npz")
    train, calibration, test = (data[k] for k in
                                ("train", "calibration", "test"))
    assert len(set(train) | set(calibration) | set(test)) == 1500
    assert sum(map(len, (train, calibration, test))) == 1500
    assert np.all(data["original_split"][test] == "test")
    assert np.all(data["original_split"][train] == "train")
    ids = list(zip(data["original_split"], data["original_index"]))
    assert len(set(ids)) == 1500
    patches = cache["patches"]
    pooled = patches.max(1).astype(np.float64)
    labels = data["labels"]
    spread = np.maximum(pooled[train].std(0), 1e-12)
    used = []
    for record in visual["feature_examples"]:
        target, feature = record["class_id"], record["feature"]
        pos = train[labels[train] == target]
        neg = train[labels[train] != target]
        contrast = (pooled[pos].mean(0) - pooled[neg].mean(0)) / spread
        ranked = np.argsort(-contrast, kind="stable")
        selected = next(int(j) for j in ranked if j not in used)
        assert selected == feature
        used.append(selected)
        top = test[np.argsort(-pooled[test, feature], kind="stable")[:3]]
        assert top.tolist() == record["top_rows"]
        ap = average_precision_score(labels[test] == target,
                                     pooled[test, feature])
        assert ap == record["test_ap"]
    for case in visual["lattice_cases"]:
        name = case["name"]
        queries = query_cache[name + "_queries"]
        coordinates = case["coordinates"]
        for i, source in enumerate(case["source_rows"]):
            assert source in train
            expected = np.zeros(512)
            expected[coordinates] = 0.5 * pooled[source, coordinates]
            np.testing.assert_array_equal(queries[i], expected)
        np.testing.assert_array_equal(queries[2], np.minimum(*queries[:2]))
        np.testing.assert_array_equal(queries[3], np.maximum(*queries[:2]))
        for query, item in zip(queries, case["operations"]):
            extent, same = spatial_extents(patches[test], query)
            assert int(extent.sum()) == item["extent_size"]
            assert int(same.sum()) == item["same_site_size"]
            counts = np.bincount(labels[test[extent]], minlength=10)
            assert counts.tolist() == item["class_counts"]
            saved = query_cache[name + "_" + item["operation"] + "_extent"]
            np.testing.assert_array_equal(extent, saved)
    if inference:
        torch.set_num_threads(4)
        backbone = feature_model(
            folder / "checkpoints/resnet34_imagenet1k_v1_codexgen.pt")
        sae = TopKSAE(width=256, latents=512, k=16).eval()
        sae.load_state_dict(torch.load(folder / "checkpoints/sae_codexgen.pt",
                                       map_location="cpu", weights_only=True))
        with torch.no_grad():
            # Same batches as extraction; all 1,500 rows are verified.
            for start in range(0, len(labels), 32):
                dense = backbone(preprocess(data["images"][start:start + 32]))
                dense = dense.permute(0, 2, 3, 1).flatten(1, 2)
                np.testing.assert_array_equal(
                    dense.numpy(), cache["dense"][start:start + 32])
            dense = torch.from_numpy(cache["dense"])
            codes = sae.encode(dense / torch.from_numpy(cache["scale"]))
            np.testing.assert_array_equal(codes.numpy(), patches)
            restored = sae.decoder(codes) * torch.from_numpy(cache["scale"])
            mean = dense[train].mean(dim=(0, 1))
            r2 = 1 - ((restored[test] - dense[test]).square().sum()
                      / (dense[test] - mean).square().sum())
            assert float(r2) == report["test_reconstruction_r2"]
    return {"images": 1500, "feature_galleries": 6, "lattice_queries": 8,
            "full_model_inference": inference}


if __name__ == "__main__":
    print(verify_natural())
