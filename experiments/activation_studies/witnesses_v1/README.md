# Complete token-witness inspection

This revision changes presentation and verifies predicates; it does not
change a retrieval protocol, model, source draw, threshold, or test score.

The source package is `src/lattmc/contextstudy`; `witnesses_codexgen.py`
derives all-coordinate masks from the existing contextual and cross-family
traces. It reuses FCA `join_all`, `le`, and `upper_mask` without modifying
those library implementations. `witnessaudit_codexgen.py` independently
checks masks, requirements, coordinate maxima, membership, and sparse/dense
caller agreement. `witnessrender_codexgen.py` shares the measured rendering
between notebooks and TeX; `galleries_codexgen.py` remains a CLI entry point.

Run from repository root using its existing uv environment:

```sh
export PYTHONPATH=src
uv run --no-sync python -m lattmc.contextstudy.witnesses_codexgen
uv run --no-sync python -m lattmc.contextstudy.witnessaudit_codexgen
uv run --no-sync python -m lattmc.contextstudy.galleries_codexgen
```

`legacyreplay_codexgen.py` is a separate, targeted CPU inference stage for
uniquely identifiable Cat/dog excerpts. Its fixed row inventory precedes
classification. It checks pooled codes against cached matrices and stores
checkpoint, token, query, and trace hashes. Four ambiguous rows are retained
without numerical claims. Original notebooks have no saved outputs to
resolve those identities; text matching is used only to locate candidate
rows, never to assign activation highlights.

The existing 72-row galleries use all saved positive query coordinates.
Twenty-two historical slots have new traces; four have unresolved identity.
Legacy and new protocols stay separate. GPT-2 contextual queries meet source
token codes; external galleries meet three pooled document vectors, select
coordinates, and calibrate alpha. External mean pooling is not a join and
has no corresponding displayed token gallery. The paper's correspondence
appendix records named-case inventories and missing counterparts.

S means at least one token satisfies the full query, D means collective
satisfaction with no whole-query token, R means rejected, and U means
unclassified historical identity. D is yellow, never a semantic judgment.
All matching token positions are rose; whole-query tokens are bold. The
main-text policy is 64 tokens around the preexisting diagnostic position;
appendix and notebook views retain every valid cached position. Special
and padding rules remain those of the original experiment. Notebook
Unicode text and IDs preserve details normalized in TeX.

All cached comparisons use the original alpha and exact float64 thresholds.
Numerical tolerance for replay consistency never relaxes membership.
The saved masks and numeric ledger are the evidence; a lone bottleneck
maximum is only an additional diagnostic. No missing counterpart is
silently created. Remaining historical tables are not newly verified.

Results are under `data/activation_studies/witnesses_v1`; the local release
requires the existing project source/dependencies and the documented prior
contextual and family artifacts. Model weights remain in upstream caches.
No data, checkpoints, or archives are uploaded by this workflow.
