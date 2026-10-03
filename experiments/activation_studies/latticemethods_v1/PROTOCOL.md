# Lattice-operation study, fixed before new outcome computation

Date: 2026-10-03. Seed: 20261003. Existing corpora/checkpoints are reused;
this is an additional analysis, not a new independent dataset validation.

## Labeled document study

Use AG News labels 0--3 and DBpedia labels 7--10, five nonoverlapping
three-document training groups per label, sampled with the fixed seed.
Retain the existing train/calibration/test identities. No new threshold
calibration is used. Use maximum-pooled nonnegative codes only.
Conditions: SmolLM2-135M MLP-output TopK at blocks 3, 15, 27; Pythia-70M
TopK block 3 residual-post; Qwen3-0.6B ReLU transcoder block 14; full
Gemma-2-2B Matryoshka checkpoint at block 8 residual-post. Do not equate
coordinates between dictionaries. Exclude BOS/padding as in extraction.

Evaluate all three constituent full vectors, pair/full-three meets and
joins, and pair meet extension E. For the first two sources, evaluate all
nine rank pairs in {1,2,3} squared plus ranks (1,2)/(1,2) and (2,3)/(2,3).
Rank positive stored amplitudes descending, break ties by coordinate ID;
skip insufficient-rank cases explicitly. Never rescale a selected amplitude.
Ranks are within-checkpoint quantities, not comparable feature importance
across dictionaries. Save selected coordinates and original amplitudes.

Compare each selected join to both constituents. Save all four joint
membership cells, both retention ratios, same-coordinate flags, and
independent dataset-label precision/recall/confusion counts. Labels test
category refinement, not conjunction of independently annotated properties.
Include different-label controls: keep the first source fixed and match
the second on log L0, log norm, and log mean training feature frequency,
using standardized training covariates, Euclidean distance, and row-ID ties.
Matching is condition-specific and does not make source meaning random.

Hold out exact normalized duplicates, repeated DBpedia titles, and test
items with character 3--5-gram TF-IDF cosine >= 0.90 to any training item.
Fit the duplicate detector on training text only. Record exclusions and
remaining source groups. Additional source-name exclusion uses all words
of length >=4 from the three DBpedia titles; report this as a lexical
control, not perfect named-entity recognition. Broad unions {7,8}, {9,10}
are exploratory and not an abstraction hierarchy. AG News has no entity
annotations; do not report invented entity coverage.

Use 2000 paired bootstrap draws over the five source groups within each
class, followed by macro averaging classes. Intervals are conditional on
these corpora, checkpoints, source selection, and the frozen grid. Do not
pool repeated predictions as independent items. Undefined precision stays
undefined; report empty extents explicitly.

For SmolLM2, transport the rank-1/rank-1 joined description and the full
pair meet between blocks 3->15, 15->27, and 3->27. Fit T=F_k G_l on the
training reference corpus; verify inclusion on that corpus. Evaluate
unchanged transported thresholds on held-out items and retain losses of
held-out inclusion. Empty reference extents use the declared top and are
flagged separately; do not interpret them semantically. Record zero T.

## Contextual-token study

Use the historical GPT-2 sources: New/York/City at 3457@(1,2,3),
Rio/de/Janeiro at 5411@(15,16,17), Cat/dog at 4042@(8,82), and sports
at 1924@(15,39,94). Test six pairs: New/York, Rio/Janeiro, Rio/de,
New/de, York/de, Cat/dog; plus full three-source NYC, Rio, and sports.
Use SAE residual-pre and transcoder normalized MLP-input codes at blocks
0,8,11. Recompute source traces offline using cached weights. Compare
pooled source traces to cached summaries; retain numerical differences and
refresh only source rows in memory if within the prior documented bound.
Original caches are never overwritten.

Apply the same selected-component grid. Report complete-corpus extents,
E and retention, not semantic accuracy from literal word matches. For
New/de rank2/rank3 at blocks 8 and 11, replay the first non-source member
of each constituent, meet, and join, ordered by cache row ID. Also replay
one first non-source join nonmember. Keep mismatches and replay membership
changes. Highlights must follow actual stored token inequalities.

Audit dataset_corrected.csv row IDs against cached tokens. AI-edited text
is a separate reading aid, never activation-aligned evidence or semantic
ground truth. Highlight original decoded tokens only.

## Interpretation

No new causal intervention is specified: current caches cannot establish
feature-mediated causal pathways. Existing prefix/suffix controls remain
context-sensitivity evidence. No monotonic abstraction, cross-layer feature
alignment, infomorphism, or lattice homomorphism is assumed. Save failures.
