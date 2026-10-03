# Additional sparse-surrogate families

The fixed protocol lives at repository-relative
`experiments/activation_studies/families_v2/PROTOCOL.md`. Reusable modules
are `src/lattmc/activationstudy/families_*_codexgen.py`; the companion is
`notebooks/sae/saelens_families_codexgen.ipynb`.

Data and derived arrays belong to `data/activation_studies/families_v2`.
The paper includes `families-experiments.tex`, an appendix description,
and generated `tables/families-*.tex` files. No model weights or raw token
arrays belong in this paper directory. Release candidates belong under
`artifacts/releases/activation_studies/families_v2`.

The four new checkpoints are Pythia-70M TopK, SmolLM2-135M MLP-output TopK,
Qwen3-0.6B ReLU transcoding, and Gemma-2-2B Matryoshka. Prefix views of the
last dictionary are not independent checkpoints. Native Hugging Face
hooks capture the specified input and target; the SmolLM2 adapter checks
centering, TopK encoding, and decoder orientation against native weights.

This extension reuses the same 3,300 documents as external_v1. Its choices
were fixed before new checkpoint outcomes, after the older results were
known. It does not constitute a new-corpus replication. Every baseline,
pooling rule, source size, and prefix view is retained in the appendix.

Candidate exclusions are recorded separately in `candidate_status.json`:
Llama's backbone access request failed with `GatedRepoError`; the selected
DeepSeek-R1-Distill-Llama-8B weights exceed local disk capacity. Neither
candidate contributes measured results. The latter is a Llama-based
DeepSeek distillation, not a native DeepSeek architecture.
