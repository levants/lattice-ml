# Context-first lattice-operation study

This is an executed exploratory extension of `latticemethods_v1`, not an
independent semantic benchmark. Read `READINGS.md` for the evidence and
`FOLLOWUPS.md` for adaptive decisions. `PROTOCOL.md` is the initial grid.
The updated paper contains a main contextual-reading subsection and a
supplementary reading analysis. Tables use the existing rose/yellow style.

## Executed scope

- 330 group/rule/condition constructions, with 660 meet/join queries and
  their constituents; 11 groups of 2--6 occurrences, five selection rules,
  GPT-2 SAE and transcoder dictionaries at blocks 0, 8, and 11.
- Independent exact audit of 1,686 query records, 942 distinct membership
  arrays, 619 closed extents, and 157 token-witness records.
- 93 GPT-2 query/item replays, preserving every inspected membership.
- 64 Pythia/SmolLM2 structural-probe rows, including sampled nonmembers.
- Seven bounded prefix contrasts, seven within-family alternatives, and
  seven suffix controls, with complete target codes saved.
- Nine new manuscript tables and a notebook of cached audits and examples.

Mathematical extent laws are distinct from exploratory textual readings.
No independent contextual gold labels, new corpus, latent intervention,
behavioral endpoint, or reconstruction-controlled causal pathway is claimed.

## Files and environment

Modules live in `src/lattmc/contextstudy/`; the companion notebook is
`notebooks/sae/contextual_reading_codexgen.ipynb`. Records and arrays live
in `data/activation_studies/contextreading_v1/`, outside the TeX tree.
The mirror uses the established local data symlink, not a public deposit.
Pinned checkpoints and cached data revisions are inherited from the
verified `latticemethods_v1` and `families_v2` extraction records. Result
JSON embeds the preceding GPT-2 record, source positions and hashes.
`provenance.json` adds the external extraction references and environment.

Use the existing project uv environment, without dependency upgrades:

```sh
export PYTHONPATH=.:src
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/lattice-reading-mpl"
study_root="$PWD"
study_out="$PWD/data/activation_studies/contextreading_v1"
study_protocol="$PWD/experiments/activation_studies/contextreading_v1"
uv run --no-sync python -m lattmc.contextstudy.semantic_audit_codexgen \
  --root "$study_root" --out "$study_out"
```

To recompute the initial grid, run the following for `sae` and `tc`, each
with block 0, 8, and 11. Prefer a separate output directory for independent
replication, preserving the delivered records.

```sh
uv run --no-sync python -m lattmc.contextstudy.exploration_codexgen \
  --root "$study_root" --out "$study_out" --kind tc --block 11 \
  --protocol "$study_protocol/PROTOCOL.md"
```

Then `readingreplay_codexgen --root ... --out ... --condition tc11`
replays the recorded qualitative selections. Other executed conditions
are tc0, tc8, sae8, and sae11. `prefixcontrast_codexgen --root ... --out ...`
runs the bounded GPT-2 input contrasts on CPU. `structureprobe_codexgen`
with the same root/output arguments uses the original four-row MPS batches
and requires Apple MPS plus the already cached Pythia/SmolLM2 weights.
It does not substitute another device or checkpoint silently.

`postreading_codexgen` verifies the original component-reading decodings
and reproduces the subsequent label comparison. The interpretation JSON
is a recorded analyst artifact, not an automatically inferred label set.
The component reading packet retains its selected row IDs and raw text;
its entries are checked against original token IDs. No additional inference
is needed to inspect these records.

Regenerate the nine tables only in the paper checkout or an isolated copy:

```sh
uv run --no-sync python -m lattmc.contextstudy.semantictables_codexgen \
  --out "$study_out" --paper texs/sparsesurrs/lattconference
```

The mirror synchronizes code, this protocol, audits and notebook, not TeX.
No data or weights were uploaded. A public reproducibility deposit remains
a separate release task; local symlinks are not downloadable artifacts.

## Numerical and textual conventions

Original amplitudes are compared exactly; rounded table values are display
only. Rank ties use increasing feature index. Full item descriptions are
tokenwise maxima; GPT-2 includes its legacy BOS, while the external probe
excludes BOS and padding. Different dictionaries have separate coordinates.
Only the four GPT-2 source rows are refreshed in memory. Replayed summaries
are compared with historical arrays and every selected membership is
checked. The untouched cached population is not claimed numerically
identical to all possible replays.

Yellow D means the item joins satisfy all requirements but no token does.
Rose marks any coordinate witness; bold rose marks whole-query witnesses.
Individual UTF-8 byte fragments can show replacement-code annotations in
TeX; they do not change the token IDs or activation tests. The notebook also
prints complete sequence decoding for readable Unicode. AI-improved CSV
text is never used for highlights, source positions, or new inference.

The bootstrap uncertainty in the preceding labeled study is unchanged.
No population interval is attached to these adaptive close readings:
source documents and duplicated/spliced chunks are not independent units.
