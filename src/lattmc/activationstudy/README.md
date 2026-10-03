# External-corpus activation retrieval

This `_codexgen` companion package evaluates registered SAELens checkpoints
on disjoint AG News and DBpedia 14 document samples. It does not replace the
historical SAE or transcoder notebooks. The associated notebook is
`notebooks/sae/saelens_external_codexgen.ipynb`.

Use the existing `my_papers/.venv/bin/python` interpreter. Add the repository
`src/` directory to `PYTHONPATH` when the package is not installed. Commands
below use `python` to mean that interpreter. Set `OUT` to the experiment
cache and `PROTOCOL` to the supplied `external_protocol.md` file.

```sh
python -m lattmc.activationstudy.download_codexgen --output "$OUT"
python -m lattmc.activationstudy.data_codexgen \
  --output "$OUT" --protocol "$PROTOCOL"
python -m lattmc.activationstudy.extract_codexgen \
  --output "$OUT" --backbone gpt2 --device mps
python -m lattmc.activationstudy.extract_codexgen \
  --output "$OUT" --backbone gemma2 --device mps
python -m lattmc.activationstudy.evaluate_codexgen \
  --output "$OUT" --protocol "$PROTOCOL"
python -m lattmc.activationstudy.audit_codexgen \
  --output "$OUT" --protocol "$PROTOCOL"
python -m unittest lattmc.activationstudy.test_external_codexgen
```

The first command is the only dataset-network step. It checks pinned
parquet hashes. Data preparation and extraction use the existing offline
Hugging Face cache. The recorded backbone and SAE revisions and file hashes
are in each `*_extraction.json`; reconstruct that cache from those upstream
revisions before a fresh inference run. Gemma access is governed by its
upstream terms. Do not substitute a newly downloaded `main` revision and
call it an exact reproduction. The recorded run used the default cache at
`~/.cache/huggingface/hub`; custom cache layouts require adapting the cache
inventory helper and are not verified by this run.

Run backbones sequentially on a 24 GiB machine. The GPU run used Apple MPS;
CPU/CUDA are explicit alternatives supported by the device argument, but
were not benchmarked here. Numerical identity across devices is not assumed.
The GPT-2 path retains release-compatible TransformerLens preprocessing.
The Gemma path uses a Hugging Face forward hook and `SAE.encode`, avoiding
any requirement that the backbone support `HookedTransformer`.

Do not load the MLP SAE solely from its local configuration: registered
loading folds the scale factor 0.9123762783479742 into its weights. Its
inputs are MLP outputs; it is not a transcoder. GPT-2 residual inputs are
before block 8, whereas Gemma residual inputs are after block 8.

Stages preserve source and artifact hashes. The independent audit
recomputes AP from distinct score groups and verifies confusion counts,
source-label isolation, all derived array hashes, and aggregate means.
The full run has 2,880 configurations, 37,440 method-score vectors, and
23,712,000 test predictions. Repeated predictions are not independent
observations. Bootstrap intervals condition on the fixed document sample.

`report_codexgen --output "$OUT" --tables <destination>` regenerates the
paper's tables. This is the only component that writes TeX; generated TeX
is not copied into the code repository. All thirteen methods and both
pooling rules are retained, including unfavorable comparisons.

The notebook executes the cached-score audit and presents the completed
run. Fresh extraction commands are documented rather than automatically
rerunning costly inference when opening the notebook. Its displayed audit
is an executed computation, not an assertion that GPU inference was rerun
inside Jupyter.

## Artifact policy

The prepared small archive contains code, the executed notebook, protocol,
source-row IDs, predictions, diagnostics, and result summaries. A separate
activation archive contains derived pooled feature arrays and dense states.
Neither archive contains raw text, token sequences, or third-party weights.
The dataset source manifests provide pinned upstream reconstruction paths.
These are local release candidates, not already uploaded GitHub/Zenodo
records. Keep large archives in Releases or a versioned research-data
repository, not ordinary Git history. Preserve upstream model/data terms;
no new license for third-party inputs is asserted here.

## Additional families

The `families_*_codexgen.py` companions implement the fixed Pythia,
SmolLM2, Qwen transcoder, and Gemma Matryoshka extension. Read
`experiments/activation_studies/families_v2/README.md` from the repository
root. The canonical cache is `data/activation_studies/families_v2`;
release ZIPs belong under `artifacts/releases/activation_studies`.

The original external_v1 source files and numerical results remain intact.
Its canonical cache is now `data/activation_studies/external_v1`; older
paper-relative and notebook cache paths are compatibility symlinks.
