# Manuscript review

This file records the initial review of `lattcontexts.tex` on 2026-09-23.
The subsequent graded-activation revision, executed experiments, and current
manuscript verification are documented in
[GRADED_EXPERIMENTS.md](GRADED_EXPERIMENTS.md).
The initial verified PDF had 27 pages.
The three original proposition proofs retain their order and argument pattern:
downward closure and join closure, the antitone adjunction, and identification
of the row and column derivations. No empirical excerpts were regenerated.

## Main corrections

- Defined complete lattices, empty meets and joins, antitone maps, closure
  operators, extents, intents, and concept order before use.
- Used uppercase object-lattice elements and lowercase attribute-lattice
  elements throughout. Position sets use H, leaving K available for bonds.
- Named all three propositions and the functional-summary example. Added
  labels to every section, subsection, subsubsection, equation, and table.
  Internal aligned displays share their enclosing equation label.
- Distinguished canonical maps from embeddings: the maps can identify
  different queries and therefore need not be injective.
- Replaced the unsupported coprime-reduction and computational-complexity
  assertions with the explicit classical-context correspondence. Retained the
  graded-incidence example with its assumptions and join-preservation argument.
- Corrected SAE inputs to the residual stream before each block. The layer-0
  SAE reads token and positional embeddings. Transcoders read normalized MLP
  inputs after the block's attention computation.
- Corrected the SAE objective so the expectation includes both loss terms.
- Corrected the loaded-layer description: both notebooks select layers
  0, 4, 6, 8, 10, and 11, and the paper presents layers 0, 8, and 11.
- Corrected support floors to minima of positive sequence maxima. They are
  not minima over individual token activations. Explained why the relaxed
  query's corpus extent contains the exact-query extent even when replacing
  a coordinate by its floor increases that coordinate numerically.
- Distinguished exact sequence-level FG closure from the notebooks' v_FG
  diagnostic, which meets matching token codes in a retrieved joint extent.
- Corrected the layer-0 Rio/de comparison: the transcoder notebook uses the
  exact source meet, whereas the SAE notebook uses its support-floor version.
- Added source row/position indices and qualified qualitative comparisons.
  Removed the unsupported random-sampling description of the five examples.
- Used dollar-delimited inline mathematics, braced mathematical scripts, and
  source lines no longer than 79 columns.

## References

Added Ganter and Wille's foundational FCA text and a software citation for
the implementation. Corrected publication metadata or titles for the lattice
contexts article, Park et al., Li et al., Lan et al., Hirth and Hanika,
Dunefsky et al., and SAELens. Corrected the early author list of Bricken et al.
Removed three uncited records unrelated to this manuscript.

Primary metadata checked included:

- https://doi.org/10.1016/j.ins.2015.12.028
- https://link.springer.com/book/10.1007/978-3-642-59830-2
- https://proceedings.mlr.press/v235/park24c.html
- https://arxiv.org/abs/2410.19750
- https://arxiv.org/abs/2410.06981
- https://arxiv.org/abs/2209.13517
- https://arxiv.org/abs/2406.11944
- https://github.com/decoderesearch/SAELens

## Parallel implementation companions

The original Python files and notebooks were preserved. New files are beside
their originals, relative to the repository root:

- `src/lattmc/fca/fca_utils_codexgen.py`
- `src/lattmc/tc/transcoder_fca_codexgen.py`
- `src/lattmc/tc/transcoder_analyzers_codexgen.py`
- `notebooks/sae/sae_gpt_small_tokens_places_min_acts_codexgen.ipynb`
- `notebooks/transcoders/tc_gpt_small_tokens_places_min_acts_codexgen.ipynb`

The FCA companion corrects map_v for an empty extent: its intent is F(empty),
not the original query. Empty meets also respect an explicitly supplied
lattice top. The analyzer and vector-loader companions route the notebook
workflow through this corrected FCA class. The analyzer rejects support-floor
probes with no positive source coordinate or no positive corpus support.

The notebook companions fix stale third-token inputs in two-token meets,
stale third-token diagnostics in the animal example, a malformed query call
in the transcoder notebook, and a callable-versus-indexed corpus-text access.
They retain the existing SAE hooks, selected layers, query thresholds, and
manual exploratory queries. Notebook code and serialized JSON obey 79 columns.

Five focused checks in `review_checks_codexgen.py` pass using the repository's
existing uv-managed Python environment. They cover empty extents, closure
idempotence and extent invariance, explicit top bounds, the zero-source guard,
and all 12 pairwise diagnostic cells in the two companion notebooks. Notebook
schema validation and Python syntax checks also pass. The zero-source check
executes the actual extracted method without importing model-loading code.

Run these checks from this manuscript directory:

```sh
MPLCONFIGDIR=/private/tmp/lattcontexts-review/matplotlib \
  PYTHONPATH=../../../src ../../../.venv/bin/python -m \
lattmc.lattconferencetools.run_codexgen \
  review_checks_codexgen
```

The dependency blocker was subsequently resolved by restoring TransformerLens
3.9.0 and constraining it below version 4 in pyproject.toml and uv.lock.
SAELens and the full analyzer now import successfully. Six bounded verification
notebooks executed offline against the original caches: both surrogate types
at layers 0, 8, and 11. They verify 36 sampled source-row activation vectors,
all cached minima/maxima/support floors, six source-example queries per run,
and sequence-level closure identities. Two near-zero support coordinates
differ from the caches; no thresholds or cached values were changed.

See [the execution report](NOTEBOOK_EXECUTION.md) for versions, numerical
differences, saved outputs, and the repeatable execution command. The full
exploratory notebooks and full-extent token diagnostics were not executed
end to end, and the typeset table excerpts were not regenerated. Their complete
numerical provenance therefore remains unverified. The manuscript retains
this limitation and does not claim freshly reproduced table measurements.

## Initial manuscript verification (2026-09-23)

The final source audit found 121 unique labels, 48 labeled equations,
37 labeled tables, and 22 resolved citation keys. It found no duplicate or
missing labels,
missing citations, unbraced mathematical scripts, forbidden inline delimiters,
or source lines exceeding 79 columns. All section headings are labeled.
`git diff --check` passes.

The isolated build used:

```sh
PAR_TMPDIR=/private/tmp/lattcontexts-biber-cache \
  latexmk -pdf \
  -outdir=/private/tmp/lattcontexts-proofread-build \
  -interaction=nonstopmode -halt-on-error lattcontexts.tex
```

Latexmk completed successfully with Biber. The final log contains no warnings,
undefined references, or overfull/underfull boxes. All 27 rendered pages were
visually inspected, including the theory, colored tables, and bibliography.
The verified PDF and bibliography output were copied to this manuscript folder.
