# Frozen cached-corpus retrieval protocol

Written before computing the new model comparison, 2026-09-24.
This is a local prospective analysis plan, not an external preregistration.
The six earlier illustrative queries are not used to select these tasks.

- Corpus: the existing 25,600 GPT-2 token blocks and six original cached
  max-pooled activation matrices. No cache correction, download, or training.
- Split exact token-block SHA-256 groups deterministically with seed 20260924
  into 60% training, 20% calibration, 20% test, using hash intervals.
  Exact duplicates cannot cross splits. Document identities are unavailable;
  this does not establish document-level or pretraining independence.
- Decode with the cached GPT-2 tokenizer. Candidate labels are adjacent ASCII
  alphabetic words of length at least three, separated only by whitespace,
  with neither word in scikit-learn's English stop-word list. Lowercase first.
  Count each phrase once per row. Retain corpus frequencies 50--400, with at
  least 15/5/5 positives in training/calibration/test. This eligibility rule
  uses label counts, not model scores, and defines the evaluated population.
- Sample 32 eligible phrases uniformly without replacement from the sorted
  list, seed 20260924. If fewer exist, use all and report that fact.
- For each phrase, sample three positive training rows without replacement
  for each of five repetitions. Seeds are shared by every model and method.
  Independently sample three uniform training rows for a random-source
  control, without requiring them to be negative.
- The graded query is the coordinatewise minimum of the three cached
  sequence summaries. This differs from the earlier token-level examples.
  Rank positive coordinates by
  (query / training maximum) * log((training size + 1) / (support count + 1)).
  Break ties by coordinate index. Compare budgets 1, 4, 16, 64, and all.
- Graded score: minimum candidate/query ratio on selected coordinates.
  Zero query gives the constant score 1. Rank scores without clipping.
  A threshold alpha returns exactly the lattice extent of alpha times the
  selected query. Choose budget by calibration average precision (AP), with
  smaller budgets preferred on ties; choose alpha from .05, .1, .25, .5,
  .75, 1 by calibration F1, preferring larger alpha on ties.
- Binary support baseline: intersection of positive-coordinate supports,
  independently selecting the same budget grid by calibration AP. A second
  binary baseline uses the graded method's selected coordinates to isolate
  magnitude information without changing the coordinate set.
- Additional baselines: full-coordinate graded query; cosine similarity to
  the arithmetic centroid of the three cached activation vectors; TF-IDF
  cosine to the three-text centroid. Fit TF-IDF on training text only, using
  unigrams and bigrams, min_df=2, max_features=100000, and sublinear term
  frequency.
  Apply the graded method to random-source queries as a negative control.
- Primary outcome: test AP, with scikit-learn's threshold-group treatment
  of ties. Secondary: expected precision at 10 under uniform tie-breaking,
  and F1 for calibrated graded thresholds. Save TP/FP/FN/TN for coincidence.
  Literal phrase lookup is an oracle by label definition, not a competitor.
- Average five repetitions within each phrase. Report macro means and 95%
  percentile intervals from 10,000 phrase-level bootstrap samples, seed
  20260925. Paired differences use the same bootstrap phrase indices.
  These intervals describe task variation within this selected lexical
  population, not semantic validity or uncertainty across trained models.
- Report every SAE/transcoder layer 0, 8, 11. Do not select a winning layer
  on test data. Do not retune this protocol after seeing test results.
- Save split IDs, eligible/selected phrases, source IDs, all per-run metrics,
  selected parameters, hashes, versions, and exact test score arrays.
  Report limitations and negative findings without suppressing failed tasks.
