# Contextual witness study, version 1

This exploratory extension reuses the original OpenWebText cache and full
activation queries. It is not a new corpus or a semantic-accuracy benchmark.
The row selection rule was fixed before neural perturbation outcomes were
computed. Decoded candidate snippets were inspected during protocol design;
this is not a blinded or preregistered semantic evaluation.

## Selection and membership

- GPT-2-small, SAE residual-pre and transcoder normalized MLP input,
  zero-based blocks 8 and 11, float32 CPU, four threads.
- All four saved nonzero full-code meet queries: NYC, Rio, animals, sports.
  Original source rows/positions and unscaled amplitudes are retained.
- Exclude source rows and case-insensitive whole-word matches to
  `new|york|city`, `rio|janeiro`, `cats?|dogs?`, or
  `cleveland|clippers|cavaliers`, respectively. The function word `de`
  is not excluded. This is a lexical exclusion, not a semantic label.
- For each of 16 conditions, sample three eligible rows without replacement
  using a freshly initialized NumPy default_rng(20261001); sort for display.
  The repeated seed is explicit; the 16 samples are not independent trials.
- Replay every selected row. Require cached/fresh max summaries to agree
  within atol=rtol=0.001, and separately report exact query membership.
- Report every sampled row, including irrelevant and ambiguous contexts.
  Any shorter main-text gallery is explicitly selected for explanation.

## Witnesses and contextual dependence

For every positive query coordinate, retain its complete 128-position
activation trace and first argmax position. A max-pooled query may have
many different token witnesses, and does not imply simultaneous activity.

Choose one diagnostic coordinate before perturbation: the query coordinate
with the fewest cached rows meeting its amplitude, ties by coordinate ID.
Choose its first maximal token in each selected row. Hold this target token
and its absolute position fixed. For three seeds (20261001 + 10*row + repeat,
repeat=0,1,2), permute all earlier ordinary token IDs, keeping special-token
slots fixed. Preserve the full token multiset. No word replacement or
retokenization occurs. These shuffled prefixes may be ungrammatical.

Permute only the later ordinary tokens once, seed=20261001+row, as a causal
mask control. Record all permutations, changed-token counts, full target
codes, witness traces, feature amplitudes, and saved-query amplitudes.
Count changes only on nonidentity prefix permutations. A material amplitude
change exceeds 0.001 + 0.001*abs(original). Count downward crossings of the
original query threshold. These counts concern the chosen witness, not
necessarily membership of the perturbed whole sequence: other positions
can still satisfy a max-pooled requirement. No population confidence
interval is claimed for this small, dependent exploratory sample.

Changes identify prefix-order sensitivity at the measured activation site.
They do not isolate an MLP-only pathway, prove semantic equivalence, or
establish which circuit causes an output. Suffix controls should agree
within atol=rtol=0.001. Source snapshots and hashes are retained; cached
arrays, weights, and original notebooks are never rewritten.
