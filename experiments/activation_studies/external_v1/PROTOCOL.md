# External activation study protocol

Frozen on 2026-09-30 before computing benchmark retrieval outcomes.
This is a local prospective plan, not an external preregistration.

- Data: pinned public `fancyzhx/ag_news` and `fancyzhx/dbpedia_14` releases.
  Use all four and fourteen supplied classes respectively. Sample at most
  150/50/100 AG News rows per class for training/calibration/test and
  75/25/50 DBpedia rows per class. Training and calibration come from the
  official training split; test comes from the official test split.
- Select rows by seeded permutation, seed 20260930 plus dataset index.
  Remove normalized-text duplicates and connected components of near
  duplicates with word-five-gram Jaccard at least 0.8. Preserve test over
  calibration over training; preserve at most one row per component. Remove
  components with conflicting labels. Also deduplicate identical truncated
  token sequences for either backbone. Report every removal without refill.
- Use a BOS token and at most 127 subsequent tokens. Ignore BOS and padding
  in every pooled summary. Dataset labels are coarse topic/entity labels,
  not feature annotations or evidence of causal interpretability.
- Checkpoints: SAELens `gpt2-small-res-jb`, block 8 residual pre;
  `gpt2-small-mlp-tm`, block 8 MLP out; and Gemma Scope 2B residual post,
  layer 8, width 16k, average L0 identifiers 37 and 301. Load through the
  SAELens registry, including its normalization overrides. Use the cached
  GPT-2 and Gemma-2-2B backbones. Record revisions, hashes and precision.
- Pool each SAE code by both maximum and mean over the same valid tokens.
  Retain raw mean-pooled dense activations for a representation baseline.
  Record valid-token activation sparsity and uncentered reconstruction
  squared error divided by input squared norm on test tokens.
- For each class and ten repetitions, sample ten positive and ten negative
  training examples, using seed 20260930 + 1000 * class + repetition.
  Compare three- and ten-positive queries using nested source subsets.
  Every method has access to the same examples and calibration labels.
  Linear probes use the corresponding three or ten negative examples;
  positive-centroid and lattice methods ignore these negative exemplars.
- Graded query: coordinatewise minimum of the positive source summaries.
  Rank positive coordinates by query/training maximum times
  log((training size + 1)/(training support count + 1)). Break ties by index.
  Select budgets 1, 4, 16, 64 or all using calibration average precision,
  preferring smaller budgets on ties. Select alpha by calibration F1 from
  1, .75, .5, .25, .1, .05, preferring larger alpha on ties.
- Compare independently selected support conjunctions, matched-coordinate
  support, a fixed top-ranked single coordinate, the full graded query,
  SAE-code cosine to the positive centroid, and a logistic probe on
  row-L2-normalized SAE codes. Probe C is selected from .01, .1, 1, 10 by
  calibration AP. Include dense-activation cosine/probes and unigram/bigram
  TF-IDF cosine/probes with the same examples. Fit vocabulary and unlabeled
  coordinate statistics on training rows only. TF-IDF uses min_df=2,
  sublinear term frequency, and at most 30,000 features.
- Before outcome computation, added on 2026-10-01: also select the best
  single coordinate among the first 64 training-ranked positive coordinates
  by calibration AP. This distinguishes conjunction selection from a
  single-feature baseline with comparable calibration access.
- Report AP, tie-aware P@10, and calibrated confusion counts. For smooth
  baselines, choose a classification threshold among calibration-score
  deciles; thresholds never use test labels. Include zero-query and empty
  full-extent rates and selected coordinate budgets.
- Main aggregate: macro average over every supplied class and then ten
  source repetitions. Paired 95% bootstrap intervals resample the ten
  repetition-level macro scores (2,000 resamples, seed 20260930). They
  describe source-selection uncertainty conditional on these fixed classes,
  documents and checkpoints, not population or training-run uncertainty.
- Sensitivity: report AP on three seeded half-size stratified test subsets
  without refitting. Also report AP in the lower half of each query's test
  TF-IDF similarities, retaining only strata with positives and negatives.
  Report stratum prevalence and number of valid tasks; this is an overlap
  diagnostic, not removal of all lexical cues.
- Keep all conditions, including negative results. Do not optimize this
  protocol after inspecting test outcomes. Record execution problems and
  deviations explicitly. Do not claim independence from backbone or SAE
  pretraining: the historical corpora may overlap those training sources.
- Store source row identifiers, hashes, scores, feature arrays and software
  metadata. Keep raw text/tokens local; distribute reconstruction manifests
  and upstream links rather than republishing benchmark text or weights.
