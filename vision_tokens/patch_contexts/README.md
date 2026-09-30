# Pretrained sparse features and exact patch contexts

Executed Prisma CLIP ViT-B/32 and SAEV DINOv2 ViT-B/14 experiments on the
same 200 training and 100 test Imagenette photographs as the earlier
Imagenette study. This release contains Python source, an executed
notebook, upstream package sources, checkpoints, and experimental evidence.
It contains no manuscript, TeX, bibliography, or PDF files.

## Results and interpretation

- Prisma: test reconstruction R2 0.9850; 1,410.75 active codes per patch.
- SAEV: test reconstruction R2 0.9295; 726.87 active codes per patch.
- Ten join queries per model: 86 pooled / 51 common-patch matches for
  Prisma, and 82 / 57 for SAEV. Totals count query-image pairs.
- All 80 vector-query extent-preservation checks pass. Twenty projected
  downset concepts are computed exactly; 16 are nonprincipal.
- All six selected identical-pixel patches lose their join match when
  isolated. Reflection correspondence exceeds the shuffled mean for
  nine class queries in each model; the chain-saw case has no original
  same-site test match. No significance test is claimed.

These results concern different dictionaries and training-selected
queries. They are not a model-quality ranking or proof that coordinates
have identical semantic meanings. A token position is not its causal
receptive field. Gray ablations are out of distribution; the six selected
cases do not estimate a population frequency. Exact downset descriptions
usually retrieve no test photograph and are not claimed to improve
retrieval. Closure describes the full 300-image context; held-out counts
use the original training-selected queries.

## Files

- `checkpoints/`: four matching backbones/SAEs, immutable HF metadata,
  original model cards, hashes, and exact checkpoint byte parts.
- `upstream/`: source-only snapshots of the executed packages, upstream
  MIT licenses, revision metadata, checkpoint-era SAEV encoder and
  preprocessing sources, and extracted normalization constants.
- `prisma/`, `saev/`: model-specific crops, full dense residuals, lossless
  full SAE codes in CSR format, selected codes, queries, closed intents,
  downset generators, controls, figures, and result/verification JSON.
- `environment/`: recorded runtime, installed versions, and a focused
  pinned environment specification. Vendored source formatting is intact.
- `release_manifest_codexgen.json`: hashes of every published file in this
  extension, excluding the manifest itself. It has an explicit whitelist;
  it cannot copy manuscript material.

Original JPEG bytes and source identifiers are reused from
`../imagenette_imagewoof/dataset/`. Both backbones use RGB central
224-pixel crops. Prisma resizes the shorter edge to 224 with bicubic
interpolation; SAEV uses 256 with bilinear interpolation, matching the
checkpoint-era preprocessing. Different crops and grids are not aligned
across models. Registers and CLS never enter a patch context.

Each `activations/chunk_*_codexgen.npz` contains dense residuals and the
full sparse matrix as `data`, `indices`, `indptr`, and `shape`. It also
records image positions, pooled codes, and reconstruction sufficient
statistics. `contexts/class_*` saves all four queries, exact patch
extents, pooled/image projections, closed intents, source patches, and
maximal downset generators. No activation truncation is used for storage.

## Pinned provenance and compatibility

Prisma source revision:
`46d21f0bb1a4e23aaad60ff777e5fa3243e09383`.
SAEV source revision:
`9715c45ffc5aa822bfdac9610b62664503ed1a71`.
Full backbone and dictionary revisions are in each checkpoint's
`metadata.json`; the original weight hashes are in `weights_sha256.json`.

- [ViT-Prisma](https://github.com/Prisma-Multimodal/ViT-Prisma).
- [Prisma model collection](https://huggingface.co/Prisma-Multimodal).
- [SAEV](https://github.com/Imageomics/saev).
- [SAEV dictionary][saev-weights].
- [DINOv2 register backbone][dino-weights].
- [DataComp CLIP backbone][clip-weights].

[saev-weights]: https://huggingface.co/osunlp/SAE_DINOv2_24K_ViT-B-14_IN1K
[dino-weights]: https://huggingface.co/facebook/dinov2-with-registers-base
[clip-weights]:
  https://huggingface.co/laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K

Prisma uses residual-post block 11 and 49 spatial tokens, with its native
encoder. SAEV uses block 10 of the register backbone and 256 patch tokens.
Block indices are zero-based. Its checkpoint predates the current encoder
convention. The adapter calls the native encoder on `x - b_dec`, matching
`relu((x - b_dec) @ W_enc + b_enc)` in the retained training-era source.
It uses symmetric clipping at 100000, the published 768-dimensional mean,
and scalar 2.0181241035461426. Current notebook clipping at -1e-5 is
recorded as a sensitivity check, not used for the retained experiment.

On one fixed training image per class, compatible normalization/encoding
has MSE 0.07229. Changing only clipping gives 0.68802; omitting encoder
centering gives 44.60590. These diagnostics use normalized activation
units. Source history and first diagnostic receipts are in
`upstream/normalization_provenance.json`; the balanced ten-image check is
in `saev/results/controls_codexgen.json`. None of the discarded incompatible
activation caches is part of this release.

## Run using uv

The experiments were run in the existing project `.venv`, with CPU and
four Torch threads. All new modules use the `_codexgen` suffix. From the
repository root, use the existing environment:

```sh
export PYTHONPATH=src
export UV_CACHE_DIR=/private/tmp/vision-uv
export MPLCONFIGDIR=/private/tmp/vision-mpl
export XDG_CACHE_HOME=/private/tmp/vision-cache
uv pip install --python .venv/bin/python --no-deps \
  -e vision_tokens/patch_contexts/upstream/prisma \
  -e vision_tokens/patch_contexts/upstream/saev
uv run --offline --no-sync python -m lattmc.vision.patch_weights_codexgen
uv run --offline --no-sync python -m lattmc.vision.patch_verify_codexgen \
  prisma --native
uv run --offline --no-sync python -m lattmc.vision.patch_verify_codexgen \
  saev --native
uv run --offline --no-sync python -m lattmc.vision.patch_notebook_codexgen
```

A fresh checkout can provision the focused environment with
`uv sync --project vision_tokens/patch_contexts/environment`. Install the
two vendored packages with `--no-deps` into that environment's Python,
then use `uv run --project vision_tokens/patch_contexts/environment`
in place of `uv run` above. The focused dependency set covers this
inference workflow; it does not install every optional upstream training
or web-application extra. First-time provisioning needs package access.

Checkpoint files exceed GitHub's individual-file limit. Exact 80 MiB
parts preserve their original bytes; `patch_weights_codexgen` reassembles
and validates SHA-256 before native inference. The model adapter also
reassembles missing files automatically. Reassembled originals are ignored
by Git. No lossy conversion of model weights is used.

To regenerate the experiment, first copy the model-specific evidence
folders elsewhere, then remove only their activation chunks to request
fresh extraction. Existing chunks are treated as immutable resume points;
changing preprocessing or weights requires a fresh output directory.
For each name (`prisma`, `saev`), run these modules in order:

```sh
uv run --offline --no-sync python -m lattmc.vision.patch_extract_codexgen NAME
uv run --offline --no-sync python -m lattmc.vision.patch_contexts_codexgen NAME
uv run --offline --no-sync python -m lattmc.vision.patch_controls_codexgen NAME
uv run --offline --no-sync python -m lattmc.vision.patch_verify_codexgen NAME \
  --native
uv run --offline --no-sync python -m lattmc.vision.patch_figures_codexgen
uv run --offline --no-sync python -m lattmc.vision.patch_notebook_codexgen
```

The notebook and default renderer use PNG evidence and have no dependency
on paper files. Python, package, and experiment directories mirror the
source repository's structure. Upstream code retains its original license
and formatting; new authored code and notebook source use 79 columns.
