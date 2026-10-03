# Offline notebook execution review

Completed on 2026-09-23 using the existing project uv environment.

For the subsequently executed full-vector activation experiments and the
updated manuscript, see [GRADED_EXPERIMENTS.md](GRADED_EXPERIMENTS.md).

## Import failure and working dependency stack

TransformerLens 4.0 removed HookedTransformer. SAELens 6.51.1 still
imports HookedRootModule from transformer_lens.HookedTransformer, and
the project also uses the removed model class and utils module.
The lockfile first changed TransformerLens 3.9.0 to 4.0.0 in commit
`3bfe55f` on 2026-09-21. This is a dependency compatibility failure.

Restored TransformerLens 3.9.0 and added its better-abc dependency.
Recorded `transformer-lens>=3.9.0,<4` in pyproject.toml; uv.lock pins
3.9.0. SAELens 6.51.1, Transformers 5.17.0, and PyTorch 2.14.0 remain
unchanged. SAELens, HookedTransformer, and the project analyzer import
successfully. `uv pip check` reports no dependency incompatibilities, and
`uv lock --check --offline` confirms that the lockfile is current.

The upstream alternative is TransformerBridge:

```python
from transformer_lens.model_bridge import TransformerBridge

model = TransformerBridge.boot_transformers("gpt2")
model.enable_compatibility_mode()
```

Compatibility mode applies the legacy weight processing. Migrating also
requires SAELens and project API changes, so this is not a tested drop-in
replacement for these notebooks. No third-party package was patched.

See [TransformerLens documentation][migration], under
`content/migrating_to_v4.html`, for the upstream migration guide.

[migration]: https://transformerlensorg.github.io/TransformerLens/

## Execution scope

Six verification notebooks executed successfully: both surrogate types
at layers 0, 8, and 11. Each runs the source notebook initialization and
source-prompt cells, using CPU with four threads and one layer at a time.
HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE were enabled. No datasets, token
files, pretrained models, or activation matrices were downloaded or
regenerated. The original Python files and original notebooks are intact.

Each run loads all 25,600 cached sequences of length 128 and the full
25,600 by 24,576 activation matrix for its layer. Checks include:

- Fresh sequence maxima for rows 3457, 5411, 117, 21481, 1924, and 4042.
- Exact cached minima, maxima, and positive floors across every row.
- Support-floor probes for all six source examples, using full-corpus
  retrieval and token inspection of at most two retrieved sequences.
- Inclusion of each source sequence and sequence-level closure checks.

These are bounded verification runs, not end-to-end executions of every
exploratory cell. In particular, full_tokens=True diagnostics and every
manual feature query were not rerun. The full notebooks request six dense
matrices (about 15 GB before models and temporary arrays), whereas the
machine had about 5.5 GB of available memory at the initial check.
The paper tables were not regenerated or independently reproduced.

## Numerical checks

All 36 sampled sequence vectors satisfy atol=0.001 and rtol=0.001.
The table reports the largest absolute error among the six sampled rows.
Support changes count coordinates crossing zero, before any tolerance.

| Model | Layer | Max absolute error | Support changes | NYC extent |
| --- | ---: | ---: | ---: | ---: |
| SAE | 0 | 0 | 0 | 67 |
| SAE | 8 | 0.00010299683 | 0 | 322 |
| SAE | 11 | 0.00026893616 | 1 | 122 |
| TC | 0 | 3.4332275e-05 | 0 | 47 |
| TC | 8 | 8.0436468e-05 | 0 | 78 |
| TC | 11 | 0.00038576126 | 1 | 226 |

Near-zero support differences:

- SAE, layer 11, row 1924, feature 11111:
  fresh 2.19643116e-05; cached 0.
- TC, layer 11, row 117, feature 8441:
  fresh 0; cached 2.354383469e-06.

The observed differences are small, but exact support-based operations
can depend on near-zero rounding. The cached matrices remain the retrieval
reference. No activation values or support thresholds were altered.

## Notebook repairs and reproducibility

The full _codexgen companions now enable offline cache access, check for
missing activation/token files before initialization, and import the
corrected FCA companion. The transcoder companion also restores the
missing A_ny[0] assignment before cross-layer comparisons and removes
a diagnostic dependency on a conditionally assigned c variable.

Executed notebook outputs are saved alongside the original notebooks:

- [SAE layer 0][sae-layer-0]
- [SAE layer 8][sae-layer-8]
- [SAE layer 11][sae-layer-11]
- [TC layer 0][tc-layer-0]
- [TC layer 8][tc-layer-8]
- [TC layer 11][tc-layer-11]

From the repository root, rerun one bounded check with:

```sh
PYTHONPATH=src uv run --no-sync python \
  -m lattmc.lattconferencetools.run_codexgen \
  execute_cached_notebooks_codexgen \
  sae 0
```

Replace sae with tc or layer 0 with 8 or 11. Run these sequentially.
The runner saves executed notebooks and JSON summaries in
`texs/sparsesurrs/lattconference/cached_notebook_checks` by default.
It explicitly marks full_notebook_execution as false.

The five earlier FCA/notebook regression checks also pass.

[sae-layer-0]:
  ../../../notebooks/sae/sae_layer0_cached_check_codexgen.ipynb
[sae-layer-8]:
  ../../../notebooks/sae/sae_layer8_cached_check_codexgen.ipynb
[sae-layer-11]:
  ../../../notebooks/sae/sae_layer11_cached_check_codexgen.ipynb
[tc-layer-0]:
  ../../../notebooks/transcoders/tc_layer0_cached_check_codexgen.ipynb
[tc-layer-8]:
  ../../../notebooks/transcoders/tc_layer8_cached_check_codexgen.ipynb
[tc-layer-11]:
  ../../../notebooks/transcoders/tc_layer11_cached_check_codexgen.ipynb
