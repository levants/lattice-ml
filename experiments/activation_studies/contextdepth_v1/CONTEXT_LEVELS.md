# Context granularity diagnostic

This is a post hoc descriptive analysis of existing families_v2 scores.
The label unions were chosen after inspecting some existing ranked examples;
this is not a prospective semantic benchmark. The same definitions are fixed
before scoring the new SmolLM2 depth comparison.

Use DBpedia queries with NaturalPlace sources (label 7) or Animal sources
(label 9), all ten existing three-source draws, maximum pooling and the
original graded scoring rule, without retuning on broader labels.

- Geographic contexts: NaturalPlace plus Village (labels 7 and 8).
- Biological contexts: Animal plus Plant (labels 9 and 10).

For each query, report strict-class AP, broad-context AP, and sibling-class
transfer AP. For sibling transfer, remove the source class from the test
universe and retrieve Village using NaturalPlace sources, or Plant using
Animal sources. This tests transfer across the declared class boundary,
not literal source-word matching. Record positive prevalence: 50/700 for
strict, 100/700 for broad, and 50/650 for sibling. Also report AP divided by
prevalence, since broadening the label set mechanically changes the baseline.

These explicit label unions approximate broad contextual relatedness.
They do not annotate feature semantics or make every category mismatch an
irrelevant passage. Conversely, they do not establish semantic similarity
for individual examples, grammatical contextual dependence, or causal flow.
Use the same ten-source-draw conditional bootstrap as the other comparisons.
