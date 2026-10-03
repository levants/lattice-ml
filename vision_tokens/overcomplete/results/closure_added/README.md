# Closure-added visual features

This cached-only extension evaluates exactly the support-removal rule
`h = where(u != 0, 0, FG(u))`, followed by `G(h)`. It does not subtract
the old activation amounts, retrain a model, or run backbone inference.

## Scope and mathematical guarantees

All 404 previous semantic queries, including single-feature/component
controls, are retained. Image and token contexts are kept separate.
Each is evaluated in its full dictionary and its originally declared
coordinate projection: 1,616 context/projection cases in total.

The same 547 held-out crops and sparse codes are used. TopK and RA-SAE
have 256 sites per image; the CLIP transcoder has 49. Missing coordinates
are exactly zero; comparisons use saved precision without an epsilon.
Empty extents close to the fixed training/test upper bound. Their added
features are vacuous consequences, not observations of shared features.

The identities `G(u) <= G(h)` and `G(u join h) = G(u)` hold by construction.
The meaningful empirical question is the structure of new matches, not
whether original matches survive. Equal extents are equivalent to equal
closed intents. Zero h returns the whole universe. Nonzero h can also be
universal when its thresholds lie below unconditional context minima.

## Numerical and visual findings

- Facial six, TopK patch context: h keeps coordinates 399 and 515 at
  2.269981622695923 and 1.625866413116455. Retrieval expands from six
  tokens in five images to 1,089 tokens in 221 images. These include
  cross-species facial configurations and human performers. The original
  contextual tiger response remains above the face. Individual features
  retrieve 4,901 and 12,150 tokens; h is more restrictive than either.
- Facial triple, TopK patch context: ten added coordinates expand three
  tokens in three images to 61 tokens in 35 images. All 35 were inspected,
  showing dogs and muzzle/nose witnesses. This is an exploratory visual
  pattern, not an independently evaluated semantic detector.
- TC texture pair, patch context: seven added coordinates expand 24 tokens
  in 21 images to 382 tokens in 228 images. Coordinate 6198 accounts for
  much of the selectivity; the other six occur in 545--547 images each.
  Patterned surfaces persist, but smoother surfaces and context also enter.
- TopK S0/S7 meet: patch closure adds nothing, so h is zero. Image closure
  adds seven coordinates and expands 66 images to 380, with no common
  token witness. Broad background/context requirements are plausible;
  the inspected outputs do not establish one shared semantic meaning.
- Performance and person--fish image queries: 261 and 73 added coordinates
  exactly recover the original three and four images. A 10% increase in
  all thresholds empties both sets. Singleton controls similarly recover
  their own token using many remaining coordinates. Instance specificity
  and finite-sample redundancy must be distinguished from generalization.

The eight initial cases are named in closure_added_visual_codexgen.py.
All extents of at most 80 images were shown; larger extents used 36 evenly
spaced ranks plus up to three old matches. New panels show whole crops and
numerical witnesses. Initial notes precede the new label tabulations,
but earlier project images were known: this is not blinded validation.
A subsequent follow-up displays all 21 facial-pair matches outside
Imagewoof/Pets, after the initial reading. It includes other mammals and
human faces and is explicitly exploratory. Selected final illustrations
do not constitute a random semantic-accuracy sample.

Threshold multipliers 0.9, 1, and 1.1 and individual-coordinate controls
are saved. Perturbed thresholds need not preserve the original extent.
Coordinates exceeding the global context minimum are counted separately.
Later class labels are descriptive comparisons on the same images.

## Reproduction

Use the existing project uv environment and original cached activations:

```sh
export PYTHONPATH=src
uv run --offline --no-sync python -m lattmc.vision.closure_added_codexgen
uv run --offline --no-sync python \
  -m lattmc.vision.closure_added_visual_codexgen
uv run --offline --no-sync python \
  -m lattmc.vision.closure_added_check_codexgen
uv run --offline --no-sync python \
  -m lattmc.vision.closure_added_report_codexgen
uv run --offline --no-sync python \
  -m lattmc.vision.closure_added_notebook_codexgen
```

Initial reading notes are an interpretation record, not generated labels.
The checker requires those notes before its later label comparison. The
report's optional `--paper` argument exports two tables and one figure;
manuscript files are not part of the lattice-ml mirror.

## Saved evidence

- Model JSON: every query name, context, space, vector row index, counts,
  degenerate-case flags, exact item mapping, and input hashes.
- `*_closed_codexgen.npz`: CSR rows containing FG(u). Recover u from the
  corresponding prior semantic query; zero its support to reconstruct h.
  JSON `vector_row` indexes deduplicated vectors; `index` indexes masks.
  A projected row is stored with zero outside its declared coordinates.
- `*_masks_codexgen.npz`: original and residual extent masks plus common
  and pooled image masks. Patch masks use image-major, row-major site order.
- `inspection_codexgen.json`: every inspected item and its exact thresholds,
  best-token values, pooled maxima, and coordinate argmax sites.
- `controls_codexgen.json`: exact selected h vectors, individual-feature
  counts, threshold controls, and later label tallies.
- `final_panels_codexgen.json`: selected final and cross-collection panels.
- `verification_codexgen.json`: 6,561 finite matrix/query cases and all
  1,616 residual extents recomputed from original sparse codes.

Five source modules and the executed six-cell notebook are mirrored with
the complete local evidence. Compact records and selected final figures
can be versioned in Git; intermediate galleries are reproducible from the
original caches. Original large-cache/weight release upload remains
pending. No new weights or large activations are created by this extension.
