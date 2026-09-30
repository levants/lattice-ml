"""Measured CIFAR-10 feature examples using a frozen pretrained ResNet34."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as functional
from torchvision.datasets import CIFAR10
from torchvision.models import resnet34

from lattmc.vision.models_codexgen import TopKSAE
from lattmc.vision.paths_codexgen import experiment_root, prepare_folders


CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
           "horse", "ship", "truck")


def preprocess(images):
    values = torch.as_tensor(images).permute(0, 3, 1, 2).float() / 255
    values = functional.interpolate(values, size=(96, 96), mode="bilinear",
                                    align_corners=False, antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (values - mean) / std


def feature_model(weights):
    model = resnet34(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu",
                                    weights_only=True))
    # Includes the complete layer3, before layer4 and the ImageNet head.
    return nn.Sequential(*list(model.children())[:7]).eval()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(dataset_root=None, weights=None, folder=None):
    folder = prepare_folders(folder or experiment_root("cifar10_resnet34"))
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    sample_path = folder / "dataset/cifar10_sample_codexgen.npz"
    if dataset_root is not None:
        training = CIFAR10(dataset_root, train=True, download=False)
        testing = CIFAR10(dataset_root, train=False, download=False)
        selected_train = np.concatenate([
            np.flatnonzero(np.array(training.targets) == c)[:125]
            for c in range(10)])
        selected_test = np.concatenate([
            np.flatnonzero(np.array(testing.targets) == c)[:25]
            for c in range(10)])
        images = np.concatenate([training.data[selected_train],
                                 testing.data[selected_test]])
        labels = np.concatenate([np.array(training.targets)[selected_train],
                                 np.array(testing.targets)[selected_test]])
        train = np.concatenate([np.arange(c * 125, c * 125 + 100)
                                for c in range(10)])
        calibration = np.concatenate([np.arange(c * 125 + 100, (c + 1) * 125)
                                      for c in range(10)])
        test = np.arange(1250, 1500)
        np.savez_compressed(folder / "dataset/cifar10_sample_codexgen.npz",
                            images=images, labels=labels, train=train,
                            calibration=calibration, test=test,
                            original_index=np.r_[selected_train,
                                                 selected_test],
                            original_split=np.array(["train"] * 1250
                                                    + ["test"] * 250))
    else:
        with np.load(sample_path, allow_pickle=False) as sample:
            images, labels = sample["images"], sample["labels"]
            train = sample["train"]
            calibration, test = sample["calibration"], sample["test"]
    backbone_path = folder / "checkpoints/resnet34_imagenet1k_v1_codexgen.pt"
    if weights is not None:
        if Path(weights).resolve() != backbone_path.resolve():
            shutil.copy2(weights, backbone_path)
        assert sha256(weights) == sha256(backbone_path)
    if not backbone_path.exists():
        raise FileNotFoundError("Supply the documented ResNet34 weights")
    backbone = feature_model(backbone_path)
    batches = []
    with torch.no_grad():
        for start in range(0, len(images), 32):
            activations = backbone(preprocess(images[start:start + 32]))
            batches.append(activations.permute(0, 2, 3, 1).flatten(1, 2))
            if start % 320 == 0:
                print(f"Natural-image extraction: {start}/{len(images)}",
                      flush=True)
    hidden = torch.cat(batches)
    scale = hidden[train].square().mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
    scaled = hidden / scale
    train_sites = scaled[train].flatten(0, 1)
    sae = TopKSAE(width=256, latents=512, k=16)
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
    for epoch in range(12):
        for batch_ids in torch.randperm(len(train_sites)).split(512):
            batch = train_sites[batch_ids]
            loss = functional.mse_loss(sae(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
        value = float(loss.detach())
        print(f"Natural-image SAE epoch {epoch + 1}: {value:.4f}",
              flush=True)
    sae.eval()
    with torch.no_grad():
        codes = sae.encode(scaled)
        reconstructed = sae.decoder(codes) * scale
    np.savez_compressed(folder / "activations/resnet34_codes_codexgen.npz",
                        dense=hidden.numpy(), patches=codes.numpy(),
                        scale=scale.numpy())
    torch.save(sae.state_dict(), folder / "checkpoints/sae_codexgen.pt")
    train_mean = hidden[train].mean(dim=(0, 1))
    r2 = 1 - ((reconstructed[test] - hidden[test]).square().sum()
              / (hidden[test] - train_mean).square().sum())
    report = {
        "backbone": "torchvision ResNet34 IMAGENET1K_V1",
        "backbone_url": "https://download.pytorch.org/models/"
                        "resnet34-b627a593.pth",
        "backbone_sha256": sha256(backbone_path),
        "hook": "layer3 output, after residual addition and ReLU",
        "input_size": [96, 96], "site_grid": [6, 6],
        "preprocessing": "RGB /255, bilinear resize, ImageNet mean/std",
        "sae_width": 256, "sae_latents": 512, "sae_topk": 16,
        "sae_epochs": 12, "seed": 2026, "batch_size": 512,
        "learning_rate": 0.001, "split_sizes": [1000, 250, 250],
        "selection": "first 125 train and 25 test images per class",
        "classes": list(CLASSES), "torch": str(torch.__version__),
        "torchvision": torchvision.__version__, "numpy": np.__version__,
        "test_reconstruction_r2": float(r2),
        "test_site_l0": float((codes[test] > 0).sum(-1).float().mean()),
        "training_alive": int((codes[train].amax(dim=(0, 1)) > 0).sum()),
        "claim": "Illustrative feature study; no CIFAR classifier benchmark",
    }
    report["artifact_sha256"] = {
        str(p.relative_to(folder)): sha256(p)
        for p in sorted(folder.rglob("*")) if p.is_file()
        and p.parent.name != "results"}
    (folder / "results/natural_codexgen.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("test_reconstruction_r2", "test_site_l0",
                       "training_alive")}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.dataset_root, args.weights, args.output)
