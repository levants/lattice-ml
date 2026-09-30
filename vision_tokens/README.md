# Vision feature evidence

Data and code for *Lattice-Theoretic Formal Concepts for Sparse Surrogate
Feature Analysis in Computer Vision*. The same relative directories are
mirrored at https://github.com/levants/lattice-ml.

## Organization

Each experiment has five purpose-based subdirectories:

- `dataset/`: image pixels, labels, original indices, and split indices.
- `checkpoints/`: trained surrogate and backbone weights.
- `activations/`: measured dense and sparse site codes.
- `retrieval/`: query vectors, source rows, extents, and ranking scores.
- `results/`: metrics, protocol metadata, and content hashes.

`digits/` contains the three trained digit CNN/SAE pairs and all 150 queries.
`cifar10_resnet34/` contains the 1,500-image natural-image sample, frozen
ResNet34, trained SAE, six feature galleries, and eight lattice queries.
`provenance/` records the verified migration and final release manifest.

The old digit cache was split into these directories without changing any
of its 27 arrays. The raw digit images were also saved, so plots and cached
verification no longer depend on a later dataset-loader version.

## Run with uv

From the repository root, use its existing environment:

```sh
export PYTHONPATH=src
export UV_CACHE_DIR=/private/tmp/vision-uv
export MPLCONFIGDIR=/private/tmp/vision-mpl
export XDG_CACHE_HOME=/private/tmp/vision-cache
uv run --offline --no-sync python -m lattmc.vision.verify_codexgen \
  texs/sparsesurrs/visionlattices
uv run --offline --no-sync python -m lattmc.vision.verify_natural_codexgen
uv run --offline --no-sync python -m lattmc.vision.report_codexgen \
  texs/sparsesurrs/visionlattices
uv run --offline --no-sync python -m lattmc.vision.examples_codexgen
uv run --offline --no-sync python -m lattmc.vision.notebook_codexgen
```

For a fresh checkout, the lightweight vision-specific environment is in
`vision_tokens/environment/`. Install it with `uv sync --project` pointing
to that directory, then run the commands above with the same `--project`
argument. Set `PYTHONPATH=src` from the repository root. This avoids requiring
unrelated language-model packages to inspect the vision results.

To retrain the digit pilot in a separate folder:

```sh
uv run --offline --no-sync python -m lattmc.vision.experiment_codexgen \
  --output /private/tmp/digits-reproduction
```

To rerun natural-image extraction and SAE training from the released sample
and backbone weights, without the complete CIFAR download:

```sh
uv run --offline --no-sync python -m lattmc.vision.natural_codexgen
uv run --offline --no-sync python -m lattmc.vision.examples_codexgen
```

This last command sequence intentionally refreshes the natural experiment.
Copy that experiment directory first to preserve its original evidence.
`--dataset-root` accepts an existing official torchvision CIFAR-10 cache;
`--weights` accepts the documented ResNet34 checkpoint. These are optional
when the released subset and checkpoint are present.

The three executed notebooks live in `notebooks/vision/`. Every new Python
module lives in `src/lattmc/vision/` and carries the `_codexgen` suffix.
See `THIRD_PARTY.md` for dataset and pretrained-weight provenance.


## Higher-resolution images and a vision transformer

[Imagenette extension](IMAGENETTE.md) documents the CNN/DINOv2 experiments,
Imagewoof transfer images, gradient-based input optimization, and controlled
response probes. Its shared image data are in `imagenette_imagewoof/`;
model-specific evidence is in `imagenette_resnet34/` and `imagenette_dinov2/`.
