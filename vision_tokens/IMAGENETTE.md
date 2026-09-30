# Imagenette, Imagewoof, and feature visualization

This extension compares frozen ResNet34 and DINOv2 ViT-S/14 backbones with
separately trained Top-32 sparse autoencoders. It adds real photographs,
optimized inputs, controlled shape stimuli, and rotation response curves.

## Data and model layout

- `imagenette_imagewoof/dataset/`: cropped pixels, original JPEG archives,
  original member paths, labels, and split assignments.
- `imagenette_imagewoof/results/`: selection protocol and upstream hashes.
- `imagenette_resnet34/`: the CNN surrogate, sharded activations, queries,
  optimized stimuli, probes, and result records.
- `imagenette_dinov2/`: the same structure for the transformer, including
  the original pretrained checkpoint and its upstream license.

The ResNet experiment reuses the released checkpoint in
`cifar10_resnet34/checkpoints/`, without duplicating it.

The Imagenette subset takes the lexically first 25 training and ten
validation JPEG paths per class from the official 320-pixel archive.
Twenty training images per class train the SAE; five are reserved for
calibration. The 100 selected official validation images are the test set.
Imagewoof contributes the first ten validation paths per class (100 total),
used only for transfer evaluation. Reserved calibration images are not
used to choose a checkpoint, threshold, or coordinate.

Preprocessing converts to RGB, resizes the shorter edge to 256 with PIL
bicubic interpolation, and center-crops to 224. Original JPEG bytes are
retained to verify this step. Both models then use ImageNet normalization.
ResNet layer3 has 14x14 sites; DINOv2's final normalized patch output has
16x16 sites, excluding CLS. This DINOv2 checkpoint has no register tokens.

## Reproduction

Run from the repository root with its existing uv environment, or use the
separate environment as described in `README.md`:

```sh
export PYTHONPATH=src
export UV_CACHE_DIR=/private/tmp/vision-uv
export MPLCONFIGDIR=/private/tmp/vision-mpl
export XDG_CACHE_HOME=/private/tmp/vision-cache
uv run --offline --no-sync python -m lattmc.vision.verify_featureviz_codexgen
uv run --offline --no-sync python -m lattmc.vision.feature_tables_codexgen
uv run --offline --no-sync python -m lattmc.vision.feature_report_codexgen
uv run --offline --no-sync python -m lattmc.vision.featureviz_notebook_codexgen
```

To repeat extraction, SAE training, and input optimization:

```sh
uv run --offline --no-sync python -m lattmc.vision.imagenette_codexgen resnet34
uv run --offline --no-sync python -m lattmc.vision.imagenette_codexgen dinov2
uv run --offline --no-sync python -m lattmc.vision.featureviz_codexgen resnet34
uv run --offline --no-sync python -m lattmc.vision.featureviz_codexgen dinov2
```

These four commands overwrite their experiment's outputs. Copy the
experiment directories first if preserving an exact existing run matters.
They use the released images and backbone weights without network access.
To reconstruct the subset from official downloads, run
`lattmc.vision.datasets_codexgen imagenette` and the analogous command with
`imagewoof`. That optional step downloads the official archives to a
temporary directory and preserves their hashes and selected JPEG bytes.

## Interpretation

The selected coordinates maximize training class contrast for English
springer and church. Selection is independent per backbone. Applying the
springer coordinate to Imagewoof does not imply breed specificity.

Fourier-parameterized optimization uses a smooth pre-Top-k response to
avoid initially zero gradients. Reported image responses use the actual
Top-32 code. Both initializations and their before/after codes are saved;
no visually preferred restart is selected. Synthetic curves, lines, and
corners have matched mean and RMS contrast. Rotated photographs also
change crop boundaries and interpolation.

This follows the investigative spirit of Distill's *Zoom In*, without
claiming to recover its InceptionV1 curve or dog-head circuits. Input
optimization tests response control; it does not demonstrate how a
feature is used downstream. Imagenette overlaps ImageNet, so these are
held-out SAE examples, not an unseen-data backbone benchmark. The
Imagewoof subset is small and deterministically selected as well.

## Meet and join galleries

Four additional figures show English springer and garbage truck queries
for both backbones. For each class separately, select its two highest
training-contrast coordinates. Choose a class-matching training image
maximizing the first coordinate, then a distinct training image maximizing
the second. Ties use coordinate index or dataset row. Queries retain half
of both source-image values; all other coordinates are zero.

Evaluation uses all 100 Imagenette test images, without Imagewoof. No test
image selects a coordinate, source, or threshold. This is an exploratory
extension on existing caches, not a preregistered benchmark. The four rows
show the two sources, their coordinatewise minimum (meet), and maximum
(join). Each displays up to three matches ranked by the minimum positive
coordinate satisfaction ratio, with ties resolved by dataset row.
Counts and common-site tests use all test images and full precision.

The top-right inset in the ResNet truck figure shows an additional meet
match outside both source extents. Its ratios for each source are below
one, while its meet ratio exceeds one. Other figures keep the top-right
position for a reading guide. Labels are dataset labels, not feature names.

```sh
uv run --offline --no-sync python -m lattmc.vision.query_galleries_codexgen
uv run --offline --no-sync python -m \
  lattmc.vision.verify_query_galleries_codexgen
uv run --offline --no-sync python -m lattmc.vision.featureviz_notebook_codexgen
```

Each model stores `retrieval/query_springer_codexgen.npz` and
`retrieval/query_truck_codexgen.npz`, with source rows, features, thresholds,
full scores, pooled/same-site masks, displayed rows, and meet-extra masks.
`results/query_galleries_codexgen.json` contains counts, original source
paths, and class histograms. The verifier independently recomputes feature
selection and every query from the cached site activations.
