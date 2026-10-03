# Graded activation experiments and manuscript revision

Completed on 2026-09-24 with the repository's existing uv environment.

The subsequent appendix split changes layout and reference formatting only.
The subsequent held-out expansion is documented in CONFERENCE_EXPERIMENTS.md.
The combined PDF now has 48 pages and 188 unique labels, with references
before the appendices. See [BUILD.md](BUILD.md) for the verified combined,
main-only, and standalone appendix outputs. Numerical results below are
unchanged; the original 39-page build record is retained for provenance.

## What changed

The main paper now explains the vector lattice, its bounds, empty operations,
the source relation, the Galois derivations, and closure using a worked
two-coordinate example. It distinguishes continuous vector lattices, finite
observed-value lattices, and binary positive-support contexts. The three
original proposition proofs retain their arguments and order.

The main experiments use full token activation codes. They do not select only
the largest feature, replace amplitudes by minimum positive values, or reduce
the query to atoms. A support-only query is retained as an explicit classical
binary-context baseline. The paper also acknowledges that classical ordinal
scaling can represent graded thresholds; the contribution is their direct
vector-lattice organization, not exclusive representational power.

Two cases, New York City and Cat/dog, remain in the main empirical narrative.
The four additional cases, full threshold tables, Rio/de pairwise comparison,
closure counts, and cross-layer coincidence appear in Appendix A. The earlier
qualitative support-floor protocol and illustrations appear in Appendix B.
Their typeset excerpts were not regenerated and remain explicitly qualified.

## Executed experiment design

- Two pretrained surrogates: residual-pre SAEs and normalized-MLP transcoders.
- Layers 0, 8, and 11; 24,576 latent coordinates per model and layer.
- All 25,600 cached rows, with 128 positions per row, for every extent.
- Six fixed source occurrences, recorded with exact row and token positions.
- Full-code meets and joins at alpha 0.25, 0.5, 0.75, and 1.0.
- A positive-support baseline on each query's complete coordinate pattern.
- An additional Rio/de pairwise meet at each model and layer.
- 78 query families and 390 evaluated extents in total.

Both new notebooks executed all their cells successfully using CPU, four
threads, and offline loading. Each layer ran sequentially. No checkpoint,
corpus, or activation cache was downloaded or overwritten.

The six freshly recomputed source summaries replace their corresponding rows
only in the working matrix. This avoids a source failing its own exact query
because of small cross-version roundoff. The other 25,594 rows retain their
cached values. Maximum source-row differences are below 0.0004; two near-zero
coordinates change support. Manifests record these corrections and input
SHA-256 hashes. Comparisons use no epsilon or implicit threshold relaxation.

## Main outcomes

- 38 of 78 query families shrink between support-only and full amplitude.
- Five query families are zero and retrieve the entire corpus.
- All 36 exact full-code joins retrieve only their source row.
- Including meet families, 42 exact extents are singletons.
- Every checked closure preserves its extent; all nesting and tested
  meet/join identities pass.

The paper reports full-corpus counts, retained-support fractions, literal
reference counts, all four coincidence cells, and Jaccard overlap. It compares
retrieved row identities, not unaligned latent coordinates. Literal matching
is not semantic annotation, and source rows are included in the counts.
The observations do not support a blanket claim that later layers are more
semantic or that one surrogate family is generally superior.

## Artifacts

- [Executed SAE notebook][sae]
- [Executed transcoder notebook][tc]
- [Experiment
implementation](../../../src/lattmc/lattconferencetools/activation_experiments_codexgen.py)
- [Table
generator](../../../src/lattmc/lattconferencetools/build_activation_tables_codexgen.py)
- [Independent dense
checks](../../../src/lattmc/lattconferencetools/test_activation_experiments_codexgen.py)
- [Result manifests and arrays](experiments/graded_activation_results.zip)

The result archive contains the six JSON manifests, six NPZ files, and the
aggregate summary. NPZ entries include queries, complete row extents, lexical
reference sets, and the closed intents checked at alpha 0.5 and 1.0. The JSON
records contain statistics for every setting, source refresh details, software
versions, and hashes of the experiment code and cached inputs. The root ignore
rules exclude loose JSON/NPZ files under texs, so the ZIP preserves them as a
repository-deliverable artifact. Working copies remain in
`../../../data/activation_studies/legacy/lattconference/activation_results`.

The earlier minimum-activation companions now link to the graded notebooks
and contain cells for reading the new manifests. The original notebooks and
original Python implementation files are unchanged.

From the repository root, execute one model/layer with:

```sh
PYTHONPATH=src uv run --no-sync python \
  -m lattmc.lattconferencetools.run_codexgen \
  activation_experiments_codexgen sae 0
```

Replace sae by tc and 0 by 8 or 11 as needed; run sequentially. Regenerate
the tables after all six runs with:

```sh
PYTHONPATH=src uv run --no-sync python \
  -m lattmc.lattconferencetools.run_codexgen \
  build_activation_tables_codexgen
```

The existing uv.lock modifications present at the start of this revision
were preserved. No dependencies were changed for these experiments.

## Verification

Independent tests compare sparse dominance and closure with dense operations
on twelve small random matrices at all four levels, plus zero queries,
empty extents, support-only retrieval, and coincidence counts. They pass.
The earlier five FCA and notebook checks also pass: seven tests in total.

The manuscript contains 160 unique labels, 59 labeled equations, and 47
labeled tables. All section headings and mathematical environments have
labels, and all citation keys resolve. New source code and notebook code obey
the 79-column limit. All twelve code cells in the two new notebooks have
execution counts and saved outputs, with no error outputs. The six manifest
code hashes match the delivered implementation, and all thirteen archive
entries match the working result files.

The final isolated build produces 39 pages without LaTeX warnings, undefined
references, or overfull/underfull boxes. All 39 rendered pages were visually
reviewed, including the final appendix edits. The final PDF and bibliography
output are delivered as `lattconference.pdf` and `lattconference.bbl` beside
the
main source. The successful build command was:

```sh
PAR_TMPDIR=/private/tmp/lattconference-biber-cache latexmk -pdf \
  -outdir=/private/tmp/lattconference-graded-build \
  -interaction=nonstopmode -halt-on-error lattconference.tex
```

[sae]:
  ../../../notebooks/sae/sae_gpt_small_tokens_places_codexgen.ipynb
[tc]:
  ../../../notebooks/transcoders/tc_gpt_small_tokens_places_codexgen.ipynb
