# Vision lattices

Main manuscript: `visionlattices.tex`; compiled output: `visionlattices.pdf`.
The source is independent of the companion paper's build tree.

- `config/`: minimal preamble and author metadata.
- `sections/`: eight readable, single-level manuscript inputs.
- `tables/`, `figures/`: generated experimental artifacts.
- `vision_tokens/`: datasets, checkpoints, caches, and results.
  The canonical location is `vision_tokens/` at the repository root.
- `LIBRARIES.md`: researched vision SAE/transcoder tools.
- `REVIEW.md`: scientific readiness and venue assessment.
- `VERIFICATION.md`: final verification record.

Python code is in `src/lattmc/vision/` at the repository root.
Executed notebooks are `visionlattices_codexgen.ipynb` and
`feature_examples_codexgen.ipynb`, and
`imagenette_features_codexgen.ipynb`, all in `notebooks/vision/`.
Every added Python source and notebook has the `_codexgen` suffix.
The existing empty `__init__.py` is unchanged. The complete vision
source, notebooks, data, and manuscript are mirrored at
https://github.com/levants/lattice-ml with the same relative paths.
See `vision_tokens/README.md` for natural-image reproduction commands.

## Reproduce the experiment

Run from the repository root using its existing uv environment:

```sh
export PYTHONPATH=src
export UV_CACHE_DIR=/private/tmp/visionlattices-uv
export MPLCONFIGDIR=/private/tmp/visionlattices-mpl
export XDG_CACHE_HOME=/private/tmp/visionlattices-cache
uv run --offline --no-sync python -m lattmc.vision.experiment_codexgen \
  --output /private/tmp/vision-reproduction
```

The output location above preserves the delivered evidence. To regenerate
the delivered experiment deliberately, change it to
`vision_tokens/digits`.
The run trains three classifiers and three SAEs, then stores the exact
query scores, source IDs, splits, weights, and code arrays. No network or
external foundation-model weights are needed for the digits pilot.

```sh
uv run --offline --no-sync python -m lattmc.vision.verify_codexgen \
  texs/sparsesurrs/visionlattices
uv run --offline --no-sync python -m lattmc.vision.report_codexgen \
  texs/sparsesurrs/visionlattices
uv run --offline --no-sync python -m lattmc.vision.notebook_codexgen
```

The notebook uses the active Python executable as its kernel. Its execution
requires local Jupyter sockets. It rechecks cached experiments and inference
rather than retraining models every time it is opened.

## Build the manuscript

From this paper directory:

```sh
mkdir -p /private/tmp/visionlattices-build
mkdir -p /private/tmp/visionlattices-biber
export PAR_TMPDIR=/private/tmp/visionlattices-biber
/usr/bin/perl /Library/TeX/texbin/latexmk -pdf \
  -outdir=/private/tmp/visionlattices-build \
  -interaction=nonstopmode -halt-on-error visionlattices.tex
```

Copy the verified PDF from the isolated build to this directory. On another
system, use its normal `latexmk` executable and temporary directories.
The supplied `tac.cls` was present initially and is unused; this article
uses the standard article class, as does the source manuscript.


## Imagenette and transformer extension

See `vision_tokens/IMAGENETTE.md` at the repository root for the matched
ResNet34/DINOv2 protocol, Imagewoof transfer examples, optimized stimuli,
and controlled response probes. Six additional figures and three tables
are generated from that evidence. The frozen transformer checkpoint is
included, so extraction and feature visualization can run offline.
