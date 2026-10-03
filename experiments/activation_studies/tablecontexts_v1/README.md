# Table identities and canonical document contexts

The presentation follows the experiment rather than a uniform word order:
caption titles identify the dataset, construction, or measured outcome;
panel headings identify backbone size, surrogate, zero-based block, and
activation site. Captions define the query, source selection, witnesses,
units, and abbreviated columns. Exact releases are collected in the shared
checkpoint identity table. Historical labels remain stable.

Residual-pre and residual-post are before and after a transformer block.
MLP-output SAEs reconstruct the feed-forward contribution, whereas a
transcoder encodes normalized MLP input and predicts MLP output. Matryoshka
names the nested Gemma dictionary; its activation mechanism is JumpReLU.
Independently trained coordinates and numerical scales are never aligned
by equal indices or common highlight colors.

The new DBpedia illustration selects rows 700--704: the first five cached
NaturalPlace training items, before any activation-based ranking. For each
of Pythia-70M block 3 (residual-post TopK SAE, k=20) and SmolLM2-135M block
15 (MLP-output TopK SAE, k=32), it takes the full meet of the five tokenwise
joins. All positive coordinates are retained and alpha is one. This is
an illustration of the original five-document common-feature construction,
not a replacement for the calibrated three-source benchmark.

The full intents have 119 and 139 positive coordinates. Both extents in
the 2,100-row context consist exactly of the five sources. Direct tests
verify source inclusion and intent closure. These are separate canonical
concepts, not a cross-model infomorphism. The first three source rows are
shown in the paper and all five in the notebook. No mismatches or rejected
outcomes were discarded: this source closure simply has no other members.
It provides no held-out generalization evidence.

Each of the ten replayed source summaries exactly matches the original
cached matrix, using the original MPS device and four-item batch boundaries.
The replay stores all positive query coordinates at every token position.
Padding and position 0 remain excluded, matching the original extraction.
All ten examples have distributed satisfaction. Yellow D describes this
predicate, not semantic correctness. Rose means any-coordinate witness;
bold would mean a whole-query token. Exact membership uses no tolerance.

Original historical colors remain dictionary-local, and the historical
OpenWebText common-feature table is not newly validated by this replay.
The four previously ambiguous historical excerpts remain unclassified.
TeX normalizes some Unicode; the notebook retains token IDs and decoded
Unicode text. Existing numerical benchmark results remain unchanged.

## Reproduction

Run from the my_papers root using its existing uv environment:

```sh
export PYTHONPATH=src
uv run --no-sync python -m \
  lattmc.contextstudy.commonexample_codexgen
uv run --no-sync python -m \
  lattmc.contextstudy.tableaudit_codexgen
uv run --no-sync python -m \
  lattmc.contextstudy.contexttables_codexgen
uv run --no-sync python -m \
  lattmc.contextstudy.galleries_codexgen
uv run --no-sync python -m \
  lattmc.latex.naming_codexgen
```

Only the first command performs targeted inference. It requires the pinned
local checkpoint and token caches. The notebook commoncontexts_codexgen
reads the saved records and independently audits the full construction.
Original notebooks and base FCA implementations are unchanged.

The reusable naming module is src/lattmc/latex/naming_codexgen.py. The text,
external-family, contextual-depth, graded, and conference table generators
call it, so regeneration retains titles and notes. Its operation is
idempotent. Run it last after any historical manual-table updates.

The mirror contains code, notebooks, and supporting records only, with a
relative link to the local cache. Set MY_PAPERS_REPOSITORY to the canonical
checkout when generating paper outputs from lattice-ml; alternatively set
LATTCONFERENCE_PAPER to an explicit TeX destination. No TeX is mirrored.

The local artifact supplements the existing repository dependencies and
prior experiment caches; it contains no pretrained weights. It has not
been uploaded. table_titles.json and TABLE_TITLES.md list every table label
and final title. build.json records final delivered PDFs and log checks.
