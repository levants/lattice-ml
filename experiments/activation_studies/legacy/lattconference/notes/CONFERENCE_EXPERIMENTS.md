# Expanded cached-corpus experiments

The frozen primary design is in
[CONFERENCE_PROTOCOL.md](CONFERENCE_PROTOCOL.md).
The interpretation and remaining scientific work are in
[CONFERENCE_REVIEW.md](CONFERENCE_REVIEW.md).

## Reproduce

Use the existing project uv environment. From this manuscript directory:

```sh
PYTHONPATH=../../../src ../../../.venv/bin/python -m \
lattmc.lattconferencetools.run_codexgen \
  execute_conference_notebook_codexgen
```

The saved notebook is
`../../../notebooks/sae/conference_retrieval_codexgen.ipynb`. It reruns dataset
preparation,
all six model/layer pairs,
the explicitly post hoc ablation, and table generation. Its kernel uses
the project's `.venv/bin/python`; no packages or models are downloaded.
Jupyter needs a local kernel socket, so a restricted execution sandbox may
need scoped permission for this runner.

The token cache must be loaded with `map_location='cpu'`. It records an MPS
device; implicit restoration caused a native PyTorch segmentation fault
in the restricted environment. Explicit CPU loading fixed this new runner.
This is separate from the earlier TransformerLens/SAELens import mismatch.
The dependency repair recorded in NOTEBOOK_EXECUTION.md remains in place.

Inputs are the original `V0.npz`, `V8.npz`, and `V11.npz` under each of
`notebooks/sae/data/sae/gpt2` and
`notebooks/transcoders/data/transcoders/gpt2`, plus the existing transcoder
`owt_tokens/owt_tokens_torch.pt`. These are read-only inputs. Unlike the
earlier six token-code examples, this benchmark needs no fresh model pass
and makes no working-row corrections.

To run only one matrix comparison after preparation:

```sh
PYTHONPATH=../../../src ../../../.venv/bin/python -m \
lattmc.lattconferencetools.run_codexgen \
  conference_experiments_codexgen prepare
PYTHONPATH=../../../src ../../../.venv/bin/python -m \
lattmc.lattconferencetools.run_codexgen \
  conference_experiments_codexgen \
  run --kind sae --layer 8
```

Run the six combinations before regenerating all tables. The focused tests
are:

```sh
PYTHONPATH=python PYTHONPATH=../../../src ../../../.venv/bin/python -m \
unittest \
  lattconferencetools.test_conference_experiments_codexgen \
  lattconferencetools.test_activation_experiments_codexgen \
  lattconferencetools.review_checks_codexgen
```

## Artifact contents

The root repository ignores loose result JSON/NPZ and manuscript notebooks.
Tracked ZIP archives preserve them without changing that ignore policy:

- `experiments/conference_retrieval_results.zip`: design and result
  manifests, query vectors, coordinate orderings, TF-IDF vocabulary and IDF,
  scripts, protocol, and the fully executed notebook.
- `experiments/conference_scores_{model}_{layer}.zip`: exact test score
  arrays and fixed-budget ablation scores for one model/layer pair.
- `experiments/conference_scores_tfidf.zip`: the shared TF-IDF scores.

The archives are split to avoid a single file over common hosting limits.
Unpack the data entries into
`../../../data/activation_studies/legacy/lattconference/conference_results` for
analysis
without recomputing activations. Use the current tools in
`../../../src/lattmc/lattconferencetools/` and the notebook
in `notebooks/`. Scripts embedded in older archives retain their historical
layout and are provenance records; do not unpack them over current sources.
The archives do not include large input caches or publish copyrighted
corpus text. They are prepared local artifacts, not a public release.

`design.json` records package versions, token hash, duplicate-aware splits,
all eligible and selected phrases, and the source identifiers for every
repeat. Each model report records its matrix hash, design hash, and
experiment-source hash. The post hoc ablation records its primary-result
hash. `summary.json` holds bootstrap summaries and label-overlap diagnostics.

The summary program does not change the primary protocol or select tasks
after observing scores. Its fixed-budget table is explicitly marked post
hoc. No downloaded data, invented human judgments, or simulated outcomes
are included in the reported statistics.

## Final verification

The complete notebook executed all eight code cells without error outputs.
All 12 focused mathematical and retrieval tests passed. The separate
artifact audit recomputed all 6,720 AP values from saved test scores and
all 960 graded confusion tables, and checked split/source isolation and
protocol hashes. The authored TeX/Python/Markdown sources obey 79 columns;
the notebook's serialized maximum is 71 columns.

The full article has 48 pages and 188 unique labels; the split main and
appendix PDFs have 19 and 30 pages. The ICLR draft has five main-text pages
and 51 pages including statements, references, and the full supplement.
All 23 cross-document named destinations resolve. Rendered page surveys
and detailed checks of the new tables and proposition found no clipping or
overlap. The ICLR appendix retains underfull-box diagnostics; no output has
undefined references/citations, duplicate labels, or overfull boxes.

The arXiv source ZIP passed a separate three-pass pdfLaTeX build with its
supplied bibliography in an otherwise empty temporary directory. This is
local verification, not an arXiv server check. Result archives passed CRC
checks and each is below 50 MiB.
