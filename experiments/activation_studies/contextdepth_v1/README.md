# Context-level and matched-depth extension

The canonical data are in `data/activation_studies/contextdepth_v1`.
The existing project uv environment was used for every Python execution.
Neural inference ran offline after downloading the public SmolLM2 block-3
and block-27 SAE checkpoints. Upstream weights stay in the Hugging Face
cache, not in the code repository or result ZIP.

Run from repository root with `PYTHONPATH=src` and the existing environment:

```sh
uv run --no-sync python -m lattmc.contextstudy.depth_codexgen --layer 3
uv run --no-sync python -m lattmc.contextstudy.depth_codexgen --layer 27
uv run --no-sync python -m lattmc.contextstudy.depth_codexgen
uv run --no-sync python -m lattmc.contextstudy.gallery_codexgen pythia_topk
uv run --no-sync python -m lattmc.contextstudy.gallery_codexgen smol_topk
uv run --no-sync python -m lattmc.contextstudy.gallery_codexgen qwen_transcoder
uv run --no-sync python -m lattmc.contextstudy.gallery_codexgen \
  gemma_matryoshka
uv run --no-sync python -m lattmc.contextstudy.gallery_audit_codexgen
uv run --no-sync python -m lattmc.contextstudy.context_metrics_codexgen
uv run --no-sync python -m lattmc.contextstudy.depth_report_codexgen
```

The executed `notebooks/sae/context_depth_codexgen.ipynb` runs saved-array
checks and displays all 24 highlighted examples; it does not rerun neural
inference. Scores are independently checked for 540 query configurations
and six methods (3,240 metric vectors). Original block-15 AP reproduces
the earlier result; 360 configurations use newly extracted blocks.

The depth protocol predates the new extraction, but reuses known data and
methods. The context unions in `CONTEXT_LEVELS.md` are explicitly post hoc.
Intervals resample ten paired source-draw macro scores, conditional on
fixed documents and checkpoints. No multiplicity adjustment is applied.
Low-overlap AP retains 40/40 AG and 110/140 DB tasks with positives in the
fixed lower-half lexical-similarity subset; omitted DB tasks are not zero
AP successes. Empty/empty extent Jaccard equals one.

Main-text snippets select four rank-one cases; the full gallery keeps all
24 top-three rows, including mismatches and rejected rows. Highlights are
measured token maxima for the query bottleneck, not human semantic labels.
Original dataset IDs, source titles, query values, token IDs, and traces
are retained. The tables transliterate Latin diacritics and escape other
Unicode code points; the notebook retains original Unicode.

The main result is readout dependent: best-single and cosine AP improve
at later measured blocks, but graded AP is nonmonotonic. No monotonic
semantic abstraction or causal feature propagation is established.

Maintained packages now use `activationstudy`, `contextstudy`,
`lattconferencetools`, and `lattcontextstools`. Migration hashes are in
`../package_migration.json`; the later paper-helper move is recorded
in `../organization_v2/relocation.json`. Paper helpers now live under
`src/lattmc/lattconferencetools`. Upstream namespaces are unchanged.
The source mirror contains code/notebooks, never manuscript TeX.

The local release archive excludes full corpora and upstream weights.
It requires the earlier families_v2 artifact for the original splits,
source activations, tokens, and frozen checkpoint metadata. Local cache
symlinks are conveniences, not portable data dependencies. No upload is
performed by the packaging command.
