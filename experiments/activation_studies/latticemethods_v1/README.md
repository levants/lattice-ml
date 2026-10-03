# Contextual lattice-operation study

This extension studies descriptions and their operations. Retrieval masks
are observable consequences, not semantic ground truth. The protocol was
fixed before the new quantitative computations on 2026-10-03. It reuses
previously examined corpora and is not a preregistration or new-corpus test.

## Executed analyses

- 40 distinct three-document source groups, reused across six checkpoint
  conditions: 240 source-group--checkpoint conditions in total.
- Full pair/triple meets and joins, constituent masks and additional E.
- 2,640 related selected-component joins and 2,640 different-label controls.
- 240 training-fitted transports across SmolLM2 blocks 3, 15, and 27.
- 396 selected-component GPT-2 joins, 36 full token pairs, 18 full triples.
- Raw token replay of fixed GPT-2, Pythia, and SmolLM2 examples, preserving
  S/D/R/N classifications and failures. The first category-matching joined
  document is a supplementary outcome-conditioned illustration; its query
  is unchanged and it does not supply an unbiased success estimate.
- Independent dense validation of 15,840 selected masks/confusion counts.
- Raw-token audit of all 25,600 AI-edited OpenWebText readings: 823 differ
  from raw decoding. Corrected prose is never used to place highlights.

No feature-mediated causal intervention or new independent semantic
annotation was executed. Dataset categories measure category refinement;
they do not establish conjunction of independent contextual properties.

## Reproduce with the existing project environment

From either checkout, use its existing uv environment and set:

```sh
export PYTHONPATH=.:src
study=data/activation_studies/latticemethods_v1
protocol=experiments/activation_studies/latticemethods_v1/PROTOCOL.md
uv run --no-sync python -m lattmc.contextstudy.methodstudy_codexgen \
  --root "$PWD" --out "$study" --protocol "$protocol"
uv run --no-sync python -m lattmc.contextstudy.methodaudit_codexgen \
  --root "$PWD" --out "$study"
uv run --no-sync python -m lattmc.contextstudy.test_operations_codexgen
```

The new notebook `notebooks/sae/lattice_methods_codexgen.ipynb` audits the
saved records and displays original-token examples. It does not silently
run model inference. Set LATTICE_REPOSITORY if launching outside a checkout.

GPT-2 replay is explicitly separate:

```sh
uv run --no-sync python -m lattmc.contextstudy.tokenstudy_codexgen \
  --root "$PWD" --out "$study" --protocol "$protocol" \
  --kind sae --layer 11
```

Repeat for sae/tc and blocks 0/8/11. Without --kind, this module audits the
corrected CSV. The original token tensor, matrices and local checkpoints
are required. Sources are refreshed only in memory under the recorded
numerical bound; original cached matrices are never overwritten.

The document gallery uses the extraction's original MPS batches:

```sh
uv run --no-sync python -m lattmc.contextstudy.methodgallery_codexgen \
  --root "$PWD" --out "$study"
```

Regeneration of tables accepts an explicit manuscript folder via --paper
and does not require copying manuscript sources into lattice-ml:

```sh
uv run --no-sync python -m lattmc.contextstudy.methodtables_codexgen \
  --out "$study" --paper /path/to/lattconference --galleries
```

## Record conventions

JSON contains source row IDs and original dataset split/item IDs,
checkpoint configurations, extraction hashes, selection rules, original
amplitudes, zero-based dictionary coordinates, all category confusion
counts and extent coincidence cells. `full_members` uses offsets into the
corresponding `design.test` array; selected `test_members` uses corpus row
IDs. Query values use the exact saved precision; displayed table values
are rounded only for readability. There is no comparison epsilon.

Transport query arrays and training members are recorded. Inclusion is
checked on the training universe, not assumed on held-out items. Empty
reference extents use the declared top, are explicitly flagged, and are
excluded from semantic transport summaries. Source-group bootstrap
intervals condition on these datasets, checkpoints and the fixed rank grid.

The upstream caches remain in families_v2 and contextdepth_v1; model
weights stay in the local Hugging Face cache. No model weights, text
collections or manuscript files are publicly uploaded by this revision.
Local synchronization is preparation for a release, not a public deposit.

## Local mirror and release boundary

The lattice-ml checkout shares the new ignored data directory through the
same local sibling-repository symlink arrangement as the earlier studies.
This link is not a downloadable dataset. Summary statistics and provenance
are copied with the code; large caches and third-party weights are not
committed or uploaded. Running the CSV audit from lattice-ml additionally
requires access to the original my_papers `owt_tokens/dataset_corrected.csv`.
The executed notebook reads the saved audit and does not require that CSV.

## Presentation and verified delivery

Titles identify the construction and its evidential role. Paired headings
identify each backbone, surrogate, block and activation site; captions
explain the selected source descriptions, witness rule and conditioning.
Exact releases, revisions and dictionary configurations remain in the
shared checkpoint references and `provenance.json`. This keeps short
headings readable without treating distinct dictionaries as aligned.
`TABLES.md` lists all fifteen new labels and final human-readable titles.

`build.json` records all five document modes. The canonical and combined
article has 116 pages; the main-only PDF has 31 and the appendix has 87.
The ICLR working draft has 122 pages including its extensive supplement;
this is not a claim of submission compliance. No unresolved references,
citations or overfull boxes remain. Ordinary underfull layout warnings
are recorded, not suppressed or misreported as a warning-free build.

The canonical delivered PDF and the clean extracted archive each match
the isolated build on all 116 pages rendered at 72 dpi. Numerical masks,
notebook output and layout have separate verification records. The
lattice-ml environment also passes the four unit tests and the independent
15,840-mask, 29-gallery-record audit. No public upload was performed.
