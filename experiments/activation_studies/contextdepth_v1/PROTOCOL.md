# Context and depth extension, version 1

This protocol fixes the new depth choices before their activation/retrieval
outcomes. It reuses the known families_v2 data and baselines, and is not an
independent corpus replication or a preregistration.

## Context-level target and depth comparison

Use all 3,300 existing AG News and DBpedia 14 documents, official split
identities, and ten existing three-positive source draws for every class.
Dataset categories are coarse contextual labels, not labels of individual
latent features. Measure transfer to different held-out documents rather
than exact occurrence of a chosen phrase.

Compare SmolLM2-135M MLP-output TopK SAEs at zero-based blocks 3, 15, and 27.
Use the same EleutherAI/sae-SmolLM2-135M-64x release and pinned revision
57ea2cb986e2545844cdd4a5bb2eb39523243494. All dictionaries have 36,864
coordinates with k=32. Replay the native decoder-centered TopK equation.
Reuse the original block-15 arrays and extract blocks 3 and 27 with the
same float32 backbone, tokenizer, prefix/padding policy, and MPS device.
Maximum pooling excludes prefix and padding positions. Keep reconstruction
error and actual token sparsity as diagnostics.

Use the existing graded budget/threshold selection, independently selected
support, calibration-selected single feature, SAE cosine, dense cosine,
and TF-IDF cosine. No new tuning rule is introduced. Reuse the exact
training source IDs; calibration selects parameters; test labels are used
only for evaluation. Report macro AP, calibrated macro precision/recall,
extent size, and AP on the lower half of test TF-IDF source similarity.
This low-overlap subset is fixed per task across layers; it still contains
lexical information. Retain prevalence and compare with TF-IDF in that
same subset.

Bootstrap the ten repetition-level macro scores in paired fashion, 2,000
resamples with the existing benchmark seed. Report later-minus-block-3 AP
intervals, not unpaired intervals or independent-document confidence.
Calibrated extent Jaccard and change in coverage distinguish a broader
neighborhood from an improved contextual retrieval result. Do not infer
monotonic semantic abstraction from layer number or extent size.

## Cross-family highlighted examples

Use four existing checkpoints: Gemma Matryoshka (full dictionary), Pythia
TopK, SmolLM2 TopK, and Qwen ReLU transcoder. Fix DBpedia NaturalPlace
(label 7) and Animal (label 9), source repeat 0, three positives, max pooling.
For each condition select the top three graded-scoring test rows, ties by
row index, irrespective of their labels. Retain all 24 examples and report
label, score, threshold, and whether the row is in the calibrated extent.
This is an illustrative gallery, not a new performance sample.

Replay the original four-row inference batches containing each candidate.
Select the query coordinate attaining the weakest max-activation/query
ratio (ties by coordinate ID), then highlight its first maximal valid token.
Store all selected-coordinate traces and the full cached block token IDs.
Recheck the original selected coordinates, score and membership against the
saved evaluation. Record the original dataset split/row ID and source names.
No minimal-positive-activation reduction or atom selection is used.

The existing all-category outcomes and low-overlap statistics are reported
alongside these examples; gallery selection does not replace those metrics.
