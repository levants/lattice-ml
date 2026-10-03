# Manuscript/resource separation and shared text-table style

This revision removes every notebook/cache compatibility link from the
conference paper and moves its Python helpers to the repository package
`lattmc.lattconferencetools`. `relocation.json` records each original link
and exact source hashes. Existing serialized arrays are not recomputed by
relocation. The immutable lexical protocol retains its bytes at
`experiments/activation_studies/legacy/lattconference/CONFERENCE_PROTOCOL.md`.

All 24 cross-family and 48 GPT-2 replayed examples remain in the generated
paired text tables. The renderer is `lattmc.latex.texttables_codexgen`;
it imports the palette from the original notebook's tokenization utility
and uses its first pastel RGB (241, 208, 208). Historical multiple-feature
colors retain their within-table identities. Highlighting locates a
measured token or historical threshold match, not a semantic annotation.

Run from repository root with the existing uv environment and
`PYTHONPATH=src`. The main galleries command is
`python -m lattmc.contextstudy.galleries_codexgen`. The result protocols
and measured activations are unchanged by the style pass.

The code-only mirror receives modules and executed notebook companions;
it does not receive TeX or full model weights. Archives remain local until
a versioned GitHub Release or an appropriate data deposit is published.
