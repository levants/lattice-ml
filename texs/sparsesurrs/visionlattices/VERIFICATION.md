# Verification record

Completed 2026-09-30 using the existing project uv environment.

## Manuscript

- Isolated pdfLaTeX/Biber build completed: 29 pages.
- Final LaTeX and Biber logs contain no warnings, unresolved references,
  unresolved citations, or overfull/underfull boxes.
- All pages rendered and visually reviewed. The final review enlarged the
  natural-image figure labels and checked their placement and contrast.
- Final PDF and BBL copied here and byte-compared with the isolated build.
- 17 TeX files, 85 unique labels, and 22 bibliography entries audited.
- Every section, subsection, formal result, displayed equation, table, and
  figure has a label. All theorem/proposition statements have names.
- Referenced labels and citation keys resolve; mathematical scripts are
  braced and inline mathematics uses dollar delimiters.
- TeX, bibliography, new Python source, documentation, environment metadata,
  and notebook source cells satisfy the 79-column limit. Binary files,
  generated lock/BBL files, serialized notebook outputs, and the unmodified
  upstream DINOv2 license are excluded.
  The previously supplied, unused class file was not reformatted.

## Mathematics and digit evidence

- 729 finite vector contexts; 41,472 adjunction checks.
- Ordinal-scaling closure equivalence, empty cases, closure identities,
  and join-to-intersection retrieval verified on that finite family.
- 729 spatial-query inclusion checks and the strict counterexample passed.
- Three CNN/SAE pairs trained on the declared digits split.
- All 150 retrieval queries recomputed, including calibration choices,
  stored scores, test metrics, and source examples.
- Splits, labels, and ten saved dataset/model/cache hashes verified.
- The migration preserved all 27 original cached arrays exactly.
- The digit notebook executes five code cells and reproduces sparse codes
  exactly from all three saved CNN/SAE pairs on all 1,797 images.

## Natural-image evidence

- Frozen ImageNet ResNet34 and a trained Top-16 SAE on 1,500 CIFAR-10 images.
- Full inference exactly reproduces the saved dense and sparse activations
  for all 1,500 images from released pixels and checkpoint weights.
- Original split/index identities, split disjointness, artifact hashes,
  reconstruction R2, and training-selected feature coordinates verified.
- Six feature selections, all test AP values, and top-ranked exemplars
  recomputed. No test-label mismatches were suppressed from the galleries.
- Eight saved queries reproduce their extents, source thresholds, meet/join
  identities, pooled counts, and same-site counts.
- Five additional figures regenerated from cached measurements.
- The natural-image notebook executes four code cells, checks cached
  results, and embeds the final feature galleries and query figures.
- Both notebooks pass nbformat validation and contain no error outputs.

## Imagenette, Imagewoof, and transformer evidence

- Official 320-pixel archive hashes and all 450 original JPEG hashes saved.
- Original JPEG bytes reproduce every released 224-pixel crop exactly.
- Imagenette has 200 training, 50 reserved calibration, and 100 test images;
  100 Imagewoof images are used only for transfer evaluation.
- ResNet34 and DINOv2 ViT-S/14 inference reproduces all 450 dense arrays
  exactly for each model. Saved SAE inference reproduces all sparse codes.
- Checkpoint and activation shard hashes, selected features, test AP, and
  split-specific reconstruction statistics independently recomputed.
- Actual responses to all eight optimized stimuli and initial images
  independently reproduced, with both initializations retained.
- All 36 synthetic stimuli per model have matched mean brightness and
  RMS contrast. Their sparse responses and all saved natural-image
  rotation responses were independently recomputed.
- Eight lattice queries recomputed, including source thresholds, meet and
  join vectors, extents, and same-site witnesses.
- The new notebook executes four cells and embeds six new evidence figures.
  All three notebooks validate, have executed code cells, and contain no
  error outputs; notebook source cells remain within 79 columns.
- Final 29-page PDF visually reviewed; a short isolated paragraph page was
  removed, and table floats were kept after their introduction.

These checks do not establish semantic interpretations, causal effects of
individual coordinates, mathematical novelty, or foundation-model
superiority. See `REVIEW.md` for scientific limitations and venue advice.

## Reproduction and release

Exact commands are in `README.md` and the repository-root
`vision_tokens/README.md`. Data, checkpoints, activations, queries, and
results now live in `vision_tokens/`, organized by experiment and purpose.
The former paper-local experiment cache has been removed after verification.
The source, notebook, evidence, and manuscript trees retain identical
relative paths in the Lattice-ML mirror.

`vision_tokens/provenance/verification_manifest_codexgen.json` records
SHA-256 hashes of the release files, excluding the manifest itself.
Experiment result files also record their data/model/cache hashes.
Previous manifests are retained explicitly as historical provenance.
The extension adds Transformers, Safetensors, and Pillow version records.

`vision_tokens/environment/requirements-tested.txt` records the versions
used for execution. The separate minimal uv environment has a resolved
lockfile; a fresh installation of that lock was not tested. Numerical
bitwise equality across different library versions or hardware is not
promised. The executed environment and saved evidence are distinguished.

The unrelated pre-existing presentation PDF modification was not changed.
