# Activation studies

Existing benchmark code lives in `src/lattmc/activationstudy`; new
contextual code uses `src/lattmc/contextstudy`. Notebooks live in
`notebooks/sae` and `notebooks/transcoders`. This folder contains protocols,
small provenance records, and execution instructions, not activation arrays.

- `external_v1`: original four-SAE, two-corpus evaluation.
- `families_v2`: Pythia, SmolLM2, Qwen transcoder, and Gemma Matryoshka
  extension. Its frozen protocol retains the same documents and baselines.
- `context_v1`: 48 contextual token-witness replays, prefix-order controls,
  complete highlighted cases, and a saved-array audit.
- `relocation_manifest.json`: checksums recorded when relocating the older
  caches and release archives. Historical source/result hashes are retained.

Canonical local caches are under `data/activation_studies`. Release ZIPs
belong under `artifacts/releases/activation_studies`; paper source bundles
belong under `artifacts/papers`. Templates remain within each paper because
those ZIPs are build dependencies, not experimental results.

Older paper-relative result paths and notebook cache paths are compatibility
symlinks. The local lattice-ml checkout shares these caches through relative
links; these links are conveniences, not portable release dependencies.
For another machine, unpack the versioned result/activation assets into the
repository root, preserving their `data/activation_studies/...` paths.

Upstream weights remain in the Hugging Face cache. Do not copy them or raw
text/token caches into Git history or the derived-data release. The large
archives are local release candidates; no remote publication is implied.

The contextual result archive contains the 48 selected token blocks and
their diagnostic traces for snippet verification, not the full raw corpus.
