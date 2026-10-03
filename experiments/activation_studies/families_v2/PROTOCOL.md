# Additional surrogate-family evaluation

Frozen on 2026-10-01 before computing retrieval outcomes for these new
checkpoints. This is a local prospective extension, not a preregistration.
The earlier external_v1 outcomes were already known. We reuse its document
sample and retrieval rule, so this is a checkpoint extension, not an
independent dataset replication or a new test set.

## Included conditions

- Pythia-70M-deduped: SAEBench TopK release
  `sae_bench_pythia70m_sweep_topk_ctx128_0730`,
  `blocks.3.hook_resid_post__trainer_0`; 4,096 features, k=20.
- SmolLM2-135M: `EleutherAI/sae-SmolLM2-135M-64x`, `layers.15.mlp`.
  Use the pretrained backbone, not the similarly named random-backbone SAE.
  Preserve native TopK encoding with decoder-bias centering. Its target is
  the MLP output; override incorrect generic residual-hook metadata.
- Qwen3-0.6B: registered
  `mwhanna-qwen3-0.6b-transcoders-lowl0`, `layer_14`.
  Encode the actual normalized MLP input and reconstruct the MLP output.
  Retain the released ReLU encoder, rather than calling it JumpReLU.
- Gemma-2-2B: registered `gemma-2-2b-res-matryoshka-dc`,
  `blocks.8.hook_resid_post`, width 32,768. Encode with the released
  JumpReLU thresholds; the release was trained with BatchTopK/Matryoshka.
  Compare the full code with its first 512 and 2,048 coordinates. These
  are nested views of one trained dictionary, not independently trained
  checkpoints. Decode prefixes with the matching decoder rows and bias.

These six conditions were selected for documented compatibility and local
resources, not retrieval scores. Record pinned revisions and all file
hashes. Missing access or incompatible releases are reported as exclusions,
not replaced after inspecting task results. Llama and DeepSeek candidates
are catalogued separately; running a smaller model does not establish
results for those families. Do not quantize a backbone merely to fit a
checkpoint experiment whose SAE was trained on a different numerical model.

## Data and inference

Reuse the 3,300 source rows and official splits recorded by external_v1.
Apply the same normalized-text and word-five-gram duplicate checks, adding
exact truncated-token checks under Pythia, SmolLM2, and Qwen tokenizers.
If any duplicate component is found, retain test over calibration over
training, remove conflicting-label groups, and use the same surviving rows
for every new condition. Do not refill or select classes by retrieval score.

Use a prefix plus at most 127 content tokens, with padding and the prefix
excluded from pooled codes. Use the tokenizer's BOS token when defined;
otherwise use EOS as an explicit document-start delimiter. This difference
is recorded, not described as an identical BOS convention across models.
Use float32 small backbones and SAE encoders. Gemma retains bfloat16 backbone
inference with float32 encoding, matching the initial external study.

Record test-token active-feature counts and uncentered normalized squared
reconstruction error. For transcoders, the denominator and residual use the
MLP output target, not the encoder input. Save mean-pooled dense encoder
inputs for the dense baseline, labeling this distinction explicitly.

## Retrieval and reporting

Retain both max and mean pooling, three and ten positive source examples,
ten source draws per class, and all thirteen earlier methods. Positive and
negative sources, calibration labels, coordinate ranks, budget choices,
threshold grid, regularization grid, tie rules, and TF-IDF fitting follow
external_v1 unchanged. Do not tune the rule to rescue previous failures.

Use the same AP, tie-aware P@10, calibrated confusion/coincidence counts,
source-draw bootstrap intervals, three half-test subsets, and low-lexical-
overlap diagnostic. Intervals condition on the fixed rows and classes;
repeated predictions are not independent observations. Report both source
sizes and pooling operators, including zero queries and empty full extents.
Do not claim a causal effect of architecture from heterogeneous releases.

Keep raw text, tokens, and upstream weights outside Git history. Data live
under `data/activation_studies/families_v2`; release archives belong under
`artifacts/releases/activation_studies/families_v2`. The previous run and
its protocol remain unchanged.
