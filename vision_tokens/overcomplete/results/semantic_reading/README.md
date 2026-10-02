# Cached visual semantic reading

This follow-up reads actual image/patch extents before consulting class
labels. It uses three existing dictionaries and the same 547 held-out
images. It does not fit a model or recompute backbone activations.

## What was examined

- 372 fixed queries: 124 per dictionary, including 16 earlier operations,
  24 single features, 60 coordinate combinations, and 24 source operations.
- Larger coordinate combinations: sizes 2, 3, 4, 6, and 8, with training
  coactivation, low-correlation, and seeded random controls, at two positive
  training quantiles. Source operations use actual training-token codes.
- 32 atlas-motivated follow-ups, explicitly exploratory, plus 16 threshold
  checks reported separately from the main query inventory.
- Full native closed intents for 12 detailed cases, including requirements
  outside the original 24-coordinate projection.

The initial displays omit dataset names, labels, filenames, and annotation
masks. They retain the whole image and numerical item IDs. Earlier project
galleries were known; this is label-withheld reading, not an independent
blinded semantic evaluation. The first interpretations were recorded in
`initial_reading_codexgen.json` before the metadata label comparison.

## Findings and limits

The six-coordinate TopK facial query at median positive thresholds returns
306 pooled images but six common patch witnesses in five images. Those
images include different animals, with a tiger's witness above its face.
The full patch closure also requires coordinates 399 and 515. These are
finite-context implications, not independently identified visual concepts.

A three-coordinate performance query returns three images of brass players
with evidence distributed across different sites. The later French-horn
labels agree with the scene reading. A pretrained RA triple returns four
person-with-fish scenes; its tench labels do not describe the person role.
Neither query has a common token witness.

The transcoder pair (13769, 5792) at upper quantiles returns 21 common-site
images including lace, grass, cracks, scales, a grid, and scene vegetation.
All 21 were inspected. A broad texture/context relationship is plausible,
but neither a unique material nor a specific curvature mechanism follows.

The native TopK meet of source S0 (training row 7, site 30) and S7 (row 144,
site 153) keeps only feature 825 at 1.748754859, returning 66 images. All 66
were inspected, including instrument scenes, animals, patterns, and sky.
Its projection to the previous 24 features is zero and returns all images.
A post-hoc 1.5 threshold multiplier retains seven instrument photographs.
The unscaled set is not a coherent instrument-only class.

Counterexamples remain: a structurally motivated pair returns a cat beside
reflective objects and a puppy with a red neck accessory. Larger source
meets can become nearly universal. Many joins are empty. None of these
results proves that unrelated-looking images have no common relationship,
or that every numerical dependency has a single semantic interpretation.

## Operations and identifiers

`G(q)` compares q with each image's coordinatewise maxima. `H(q)` requires
one contextualized token to dominate q. `G_patch(q)` returns token IDs.
A vector join conjoins numerical requirements, while a vector meet weakens
them and can retrieve more than the union of component extents. Concept
joins instead close extent unions. `FG` closes a query in a declared finite
context without changing its extent there.

IDs I000--I546 map to (dataset, original row) pairs in each model JSON.
The source S-number mapping and token positions are in every source-query
record. Site grids are 16 by 16 for the DINOv2 dictionaries and 7 by 7 for
the CLIP transcoder. A grid cell is not the token's receptive field.

Requirements always hold within an individual returned image or token.
Different returned images are never pooled to satisfy one image query.
Source queries use training codes outside the held-out context; their
retrieval is transfer of a frozen query, not test-set closure of the sources.

## Reproduction

From the repository root, use the existing uv environment and caches:

```sh
export PYTHONPATH=src
uv run --offline --no-sync python -m lattmc.vision.semantic_gallery_codexgen
uv run --offline --no-sync python -m lattmc.vision.semantic_queries_codexgen
```

Inspect the anonymous atlases before reading label outputs. The recorded
initial notes document this analysis; rerunning is not a new blinded study.
The remaining commands reproduce its declared follow-ups and diagnostics:

```sh
uv run --offline --no-sync python -m lattmc.vision.semantic_followup_codexgen
uv run --offline --no-sync python -m lattmc.vision.semantic_controls_codexgen
uv run --offline --no-sync python -m lattmc.vision.semantic_report_codexgen
uv run --offline --no-sync python -m lattmc.vision.semantic_notebook_codexgen
uv run --offline --no-sync python -m lattmc.vision.semantic_verify_codexgen
```

The notebook uses a local Jupyter kernel. The verifier checks 404 outcomes,
240 legacy dataset comparisons, 1,428 numerical witnesses, input hashes,
and six executed notebook cells. Array comparisons retain full precision;
displayed decimals are rounded.

## Files and provenance

- `*_codexgen.json`: exact queries, sources, component counts, closures,
  numerical witnesses, and SHA-256 hashes of existing input artifacts.
- `*_masks_codexgen.npz`: complete patch and image membership masks.
- `initial_reading_codexgen.json`: interpretations before label reveal.
- `inspection_items_codexgen.json`: exact complete/rank-sampled image IDs.
- `labels_after_reading_codexgen.json`: later descriptive label comparison.
- `full_intents_codexgen.json`: native closed intent coordinates and values.
- `sensitivity_codexgen.json`: all 16 threshold perturbation outcomes.
- `figures/`: measured pixels, token boxes, and explicit membership margins.

The two repositories mirror the complete local folder. Compact numerical
records and the three final comparison figures are suitable for ordinary
Git; the other reproducible atlas/survey images need not be duplicated in
public Git history. No weights or new large activation arrays are created.
Original image provenance and redistribution context are inherited from
the parent experiment dataset records. The earlier large-cache release
upload is still pending; a source mirror is not a completed data deposit.
