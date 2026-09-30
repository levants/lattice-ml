# Overcomplete and pretrained sparse vision experiments

This experiment extends the lattice feature analysis with native
Overcomplete 0.3.0 models and a native Prisma transcoder. Image classes and
patch positions do not constrain feature sharing. Mathematical concepts
record common numerical requirements; visual meanings remain hypotheses.

## Evidence and scope

- 36 local fits: TopK, BatchTopK, JumpReLU, relaxed archetypal TopK,
  fixed-L1 ReLU, and a failed adaptive-L1 diagnostic; two settings,
  three seeds, 12 epochs, 1,536 coordinates.
- A pinned pretrained 32K relaxed archetypal DINOv2 SAE. Its native class
  is BatchTopKSAE. Inference uses the wrapper threshold 0.829, not a
  post-hoc fixed Top-5 rule. The native encoder uses LayerNorm.
- A pinned pretrained CLIP block-1 TopK transcoder, width 49,152, k=256,
  with an explicitly checked dense skip contribution.
- Imagenette, Imagewoof, Oxford-IIIT Pets, PartImageNet, DTD, 32 exploratory
  shape drawings, and 48 fixed-contrast grating probes.
- Native-vs-explicit checks, reconstruction, actual sparsity, seed matching,
  frozen meet/join queries, annotation references, position permutations,
  image bootstrap intervals, and measured visual examples.

Quantitative coordinate selection uses only Imagenette training images.
The foreground gallery is separately exploratory: it ranks pretrained
coordinates by pet foreground occupancy times normalized spatial entropy,
requiring responses in ten pet, five Imagewoof, and five PartImageNet test
images. The frequency gallery is also exploratory. Neither supplies an
unbiased semantic-accuracy estimate. Dataset-balanced pictures are the
strongest example *within each column*, not a global top-image ranking.

The adaptive ReLU controller failed to produce a useful reconstruction
tradeoff. Its checkpoints and history are retained. Fixed-L1 ReLU is much
denser than the hard-budget runs. The shared archetypal landmarks and
identity mixing initialization constrain interpretation of seed stability.
The test subsets are small and deterministic. Image-bootstrap intervals
are conditional on these subsets, not full-population guarantees.
Cross-dataset duplicate images and pretraining overlap are not generally
excluded. Part IDs are retained numerically; ID 40 is background. Pet
trimaps use 1 for foreground, 2 for background, and 3 for boundary.

## Layout

- `dataset/`: 224-pixel crops, masks, and exact source/split records.
- `checkpoints/`: native local weights, training histories, upstream
  checkpoint metadata, and the verified compact pretrained dictionary.
- `activations/`: exact frozen states with backbone and crop metadata.
- `codes/`: lossless CSR codes and per-image error/baseline contributions.
- `results/`: full numeric scores, queries, controls, and verification.
- `figures/`: measured figures and panel-level provenance.
- `environment/`: pinned runtime dependencies.
- `releases/`: upload bundles; excluded from ordinary Git history.

The same relative paths are used in `my_papers` and `lattice-ml`. No paper
or TeX file belongs in the experiment mirror. Large caches and weights
belong in release assets or their original Hugging Face repositories.
The original 4.39 GB pretrained checkpoint remains on Hugging Face; the
verified compact derivative avoids repeating its quadratic mixing matrix.

## Environment and reproduction

Use the project's existing uv environment. The focused dependency record
is in `environment/requirements_codexgen.txt`. Prisma source is the pinned
copy already used by `vision_tokens/patch_contexts`; it is not replaced.

```sh
export PYTHONPATH=src:vision_tokens/patch_contexts/upstream/prisma/src
export MPLCONFIGDIR=/private/tmp/vision-mpl
export UV_CACHE_DIR=/private/tmp/vision-uv
```

All commands below run from the repository root. To install the recorded
runtime in a fresh environment, use `uv pip install --python .venv/bin/python
-r vision_tokens/overcomplete/environment/requirements_codexgen.txt` on
one shell line. Existing project users should retain their uv environment.

1. Restore the release assets at repository root. Each archive preserves
   `vision_tokens/overcomplete/...` paths. Verify the release SHA-256 list.
   The existing `patch_weights_codexgen` helper restores backbone weight
   chunks from the earlier experiment when necessary.
2. For a fresh extraction, fetch the pinned upstream SAE and transcoder:

```sh
uv run --offline --no-sync python -m \
  lattmc.vision.overcomplete_fetch_codexgen checkpoint
uv run --offline --no-sync python -m \
  lattmc.vision.overcomplete_fetch_codexgen transcoder
```

The module's network fetches remain online; uv itself uses the existing
environment without dependency resolution. Raw inputs can be reconstructed
with `overcomplete_data_codexgen` for Imagenette/Imagewoof, the `dtd`
fetch and data commands, `overcomplete_parts_codexgen` followed by the
`parts` data command, and `overcomplete_pets_codexgen`. The pet script uses
a pinned public viewer subset plus original trimaps; its JPEG bytes may
differ from the original archive. The PartImageNet downloader validates
ZIP CRCs and downloads only selected members. Original image archives for
Imagenette/Imagewoof are part of the preceding experiment.

3. Cache DINOv2 states with `overcomplete_extract_codexgen DATASET`.
   Train with `overcomplete_train_codexgen --help`; native MPS and CPU
   backends are supported. Training in this release used MPS, four CPU
   threads for evaluation, and the final epoch without test selection.
4. Run `overcomplete_evaluate_codexgen` for local checkpoints, and
   `overcomplete_pretrained_codexgen DATASET` or
   `overcomplete_transcoder_codexgen DATASET` for pretrained extraction.
   The latter two save codes; use `queries` and `evaluate` from the
   evaluation module to produce their query results. Always process
   Imagenette before transcoder transfer data to fit its output mean.
5. `overcomplete_annotations_codexgen MODEL pets` (or `parts`) evaluates
   annotation association. `overcomplete_evaluate_codexgen --stability`
   compares the three seeds. `overcomplete_report_codexgen` aggregates
   scores and 2,000 image-bootstrap replicates.
6. `overcomplete_figures_codexgen MODEL` renders training-selected
   examples; `--queries` adds the frozen TopK meet/join example.
   `--foreground-gallery` and `--transfer-gallery` are exploratory options
   for the pretrained RA model. `overcomplete_probe_codexgen` creates the
   gratings; extract both pretrained models on `gratings`, then call its
   `plot()` function. Shape probes are descriptive only: rotation fill and
   contrast differ, so they are not controlled curvature tests.
7. `overcomplete_verify_codexgen` checks caches and algebraic identities.
   `overcomplete_notebook_codexgen` executes the review notebook in
   `notebooks/vision/overcomplete_features_codexgen.ipynb`.

Each module invocation uses the same `uv run --offline --no-sync python -m`
prefix as above. Source and notebook lines stay within 79 columns.

## Provenance and attribution

Upstream checkpoint revisions and SHA-256 hashes are included in the
manifest and adjacent metadata. Primary resources:

- https://github.com/KempnerInstitute/overcomplete
- https://huggingface.co/matybohacek/RA-SAE-DINOv2-32k
- https://github.com/Prisma-Multimodal/ViT-Prisma
- https://huggingface.co/Prisma-Multimodal
- https://github.com/fastai/imagenette
- https://www.robots.ox.ac.uk/~vgg/data/pets/
- https://www.robots.ox.ac.uk/~vgg/data/dtd/
- https://github.com/TACJu/PartImageNet
- https://huggingface.co/datasets/timm/oxford-iiit-pet
- https://huggingface.co/datasets/turkeyju/PartImageNet

The compact pretrained RA checkpoint is a derivative of the upstream
CC-BY-SA-4.0 checkpoint; attribute Bohacek and collaborators and preserve
that license. Upstream software and dataset terms remain applicable;
repository code licensing does not relicense source photographs. Source
identifiers accompany the research crops and figures. No claim is made
that all image rights belong to the experiment author.
