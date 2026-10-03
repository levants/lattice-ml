# Reproducing the additional-family study

Run from the repository root with the existing uv environment. Set
`PYTHONPATH=src`; use `.venv/bin/python` or `uv run --no-sync python` so
reproduction does not upgrade dependencies implicitly. The extraction
manifests record the versions actually used for this run.

The notebook `notebooks/sae/saelens_families_codexgen.ipynb` executes an
independent cached-score audit. It records no claim of notebook GPU
inference. Extraction and retrieval were executed by the source modules.

```sh
export PYTHONPATH=src
FAMILY_DATA=data/activation_studies/families_v2
FAMILY_PLAN=experiments/activation_studies/families_v2/PROTOCOL.md
.venv/bin/python -m lattmc.activationstudy.families_audit_codexgen \
  --output "$FAMILY_DATA" --protocol "$FAMILY_PLAN"
.venv/bin/python -m unittest \
  lattmc.activationstudy.test_families_codexgen \
  lattmc.activationstudy.test_external_codexgen
```

The result archive contains scores and extraction manifests. The separate
activation archive contains pooled sparse codes and dense summaries.
Unpack both at the repository root. Cached-score auditing needs neither
weights nor raw texts. Extraction additionally requires pinned upstream
weights and token arrays, reconstructed from the original external_v1
source rows. The original protocol and dataset downloader remain intact.

`families_download_codexgen` restores and checks the files named in the
extraction manifests. `families_data_codexgen` takes `--previous`,
`--output`, `--protocol`, and `--downloads` (the supplied pinned manifest).
The previous directory must include reconstructed `texts.local.json`.
No document was removed by the extra tokenizer checks in this run.

```sh
.venv/bin/python -m lattmc.activationstudy.families_extract_codexgen \
  --output "$FAMILY_DATA" --family pythia_topk
```

Repeat extraction for `smol_topk`, `gemma_matryoshka`, and
`qwen_transcoder`, then run `families_evaluate_codexgen` with `--output`
and `--protocol` as above. The default accelerator is MPS; `--device cpu`
is supported. Fresh inference must use the recorded revisions and hooks;
changing numerical precision or library behavior requires a new run.
Existing manifests reject source changes instead of silently relabeling
old results as new computations.

The TopK adapter verifies SmolLM2 against native weights, including
centering and decoder orientation. Hook tests distinguish residual output,
MLP output, and transcoder input/target. The score audit independently
recomputes AP and confusion counts, checks source/test isolation and hashes,
and verifies all eight exact prefix-array identities.

The local lattice-ml checkout shares canonical data through a relative
symlink. This link is not a portable dependency: on another machine,
create a real `data/activation_studies/families_v2` from the release assets.
Code and notebooks are synchronized bytewise; TeX is not copied.

`candidate_status.json` distinguishes measured conditions from excluded
Llama/DeepSeek candidates. No raw text, tokens, or model weights are in the
prepared release archives. They remain local; no upload has occurred.
