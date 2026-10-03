# Publication assessment and completed empirical revision

Assessment date: 2026-09-24. Primary target: an ML main-track submission.

## Assessment

The revised work is a more defensible methods-and-experiments preprint.
I would not yet describe it as a strong NeurIPS, ICML, or ICLR main-track
submission. Formatting and proofreading cannot resolve the remaining
scientific gap: the main construction specializes established FCA/pattern
structures, while the new evidence concerns lexical retrieval in one cached
GPT-2 corpus. There is no independent semantic or causal validation.

The original six examples were useful illustrations but insufficient
evidence of retrieval quality or generalization. The revision now includes
a fixed-protocol held-out comparison rather than relying on those examples.
The strongest supportable empirical contribution is selective graded
activation retrieval and the failure of high-dimensional conjunctions.
Claiming a new Galois connection, new expressive power over classical FCA,
or generally interpretable concepts would overstate the contribution.

An arXiv version can present this narrower contribution after author review
and resolution of the artifact availability statement. It should describe
the results as a controlled diagnostic, preserve the limitations, and avoid
implying that historical excerpts have all been reproduced. Posting a
preprint does not establish peer-review readiness or acceptance prospects.

## What was completed

- Added explicit pattern-structure positioning and the observed-threshold
  equivalence proposition, preserving the preceding proofs.
- Clarified that the bounded activation box is a lattice of vectors, not a
  vector space or a Riesz space; its continuous nontrivial factors are
  atomless.
- Executed 32 automatically sampled phrase tasks, five source draws, and
  six model/layer pairs: 960 paired configurations. Queries use three
  positive training rows and the original cached activation magnitudes.
- Separated training/calibration/test rows, grouping exact duplicate token
  blocks. Fitted TF-IDF and coordinate statistics on training data only.
- Compared graded retrieval, independently calibrated binary support,
  matched-coordinate support, full-coordinate queries, activation cosine,
  TF-IDF, and random-source controls.
- Added phrase-bootstrap intervals, paired differences, tie-aware P@10,
  calibrated confusion counts, all task counts, and budget diagnostics.
- Added a clearly labeled post hoc fixed-budget ablation after inspecting
  the primary comparison. It does not change the frozen primary protocol.
- Executed the complete companion notebook with the project uv Python.
  Kept original notebooks and activation/token caches unchanged.
- Prepared an anonymous, shortened ICLR 2027 working draft using the
  official style, alongside the full paper and split appendix versions.
  This draft is a format-ready starting point, not submission approval.

## Results and their limits

| Model/layer | Graded AP | Support AP | Cosine AP | TF-IDF AP |
| --- | ---: | ---: | ---: | ---: |
| SAE 0 | 38.94% | 25.75% | 28.87% | 53.95% |
| SAE 8 | 61.10% | 34.87% | 17.72% | 53.95% |
| SAE 11 | 57.77% | 29.36% | 19.02% | 53.95% |
| TC 0 | 64.51% | 40.08% | 41.05% | 53.95% |
| TC 8 | 55.17% | 29.18% | 32.36% | 53.95% |
| TC 11 | 49.83% | 19.15% | 26.19% | 53.95% |

The graded-support paired AP improvement is 13.19--30.67 percentage points.
All six phrase-bootstrap intervals exclude zero, but these are descriptive
comparisons over correlated tasks, not a multiple-testing-adjusted claim.
Matched-coordinate comparisons also favor magnitudes. The TF-IDF result
is mixed; only transcoder layer 0 has a positive paired interval excluding
zero, and SAE layer 0 has a negative interval excluding zero.

Full-coordinate extents are empty in 956/960 calibrated test evaluations.
The selected budgets are one coordinate in 571 cases, four in 363, and
sixteen in 26. A fixed single-coordinate query is already competitive;
calibration improves AP over it by 4.56--14.72 points. This supports choosing
small conjunctions, not a blanket claim that combining more features helps.

The corpus contains 25,598 distinct token blocks. The 60/20/20 hash split
produces 15,429/5,187/4,984 rows. Exact duplicates cannot cross splits,
but source-document identities and near-duplicate controls are absent.
Six phrase pairs have test-label Jaccard overlap at least 0.5. The phrase
bootstrap does not remove these correlations. Test-label frequency enters
the eligibility rule, so the population is frequent phrases in this corpus.

## Next experiments required for a stronger main-track submission

1. **Independent semantic evaluation.** Define a compact set of semantic
   retrieval tasks with blinded annotation or an established labeled corpus.
   Use at least two annotators, report agreement and adjudication, and keep
   lexical-overlap strata separate. Do not relabel the current literal
   matches as semantic ground truth. This requires actual annotations;
   none have been fabricated in this revision.
2. **External replication.** Freeze the current selection rule and evaluate
   on a second document-separated corpus and another model/surrogate family.
   Record model-training overlap when known and evaluate near-duplicate
   sensitivity. The available GPT-2 cache cannot supply this evidence.
3. **Matched supervision and combination baselines.** Compare to a simple
   linear probe using the same labeled examples/calibration budget, and to
   a calibrated single-feature selector. The existing single-feature
   ablation uses the first training-ranked coordinate; it does not search
   all coordinates with the same calibration supervision as a probe.
4. **A concrete interpretability use case.** If retaining mechanistic claims,
   test whether lattice-derived groups improve a specific analyst task or
   survive causal intervention. Otherwise keep the paper framed as retrieval
   and representation analysis, where causal validation is not the claim.
5. **Release and author validation.** Resolve the artifact URL, choose the
   release license, document access to large caches, and independently
   review the added proof, labels, code, and AI-use statement. No public
   repository, license, deposit, or submission was created by this run.

These steps address contribution and evidence, not just presentation.
There is no sound way to promise acceptance or assign a meaningful numeric
acceptance probability from the manuscript alone.

## Venue and timing

- ICLR/NeurIPS/ICML: plausible topical fit for representation analysis,
  with the evidence and novelty gaps above. The anonymous ICLR draft is
  the current concrete format target; do not submit it unchanged elsewhere.
- ICCV/ECCV: the present language-only study has a weak topical fit. A
  meaningful vision experiment would be needed before targeting those venues.
- A formal-concept-analysis or neuro-symbolic venue is a closer fit to the
  mathematical framing. A journal such as TMLR is another possible route
  after the empirical contribution is strengthened; it is not an automatic
  fallback acceptance. IEEE and ACM name publishers, not unique standards.

The [ICLR 2027 instructions][iclr] allow nine main-text pages and unlimited
references/appendices, with appendices after references. They require an
AI-use statement. The [call for papers][iclr-cfp] permits arXiv preprints.
Its abstract deadline was September 18, 2026, and its paper deadline is
September 25, 2026, Anywhere on Earth. Do not assume that a new submission
can be started after the abstract deadline, or rush an unfinished empirical
case to meet the paper deadline.

[ICML 2026][icml] specifies eight main-text pages and an anonymous combined
PDF. [NeurIPS 2026][neurips] has its own official style and submission
requirements. These are the verified available cycles, not a guarantee of
unchanged requirements for later calls. [TMLR][tmlr] permits preprints, but
requires its own editorial and formatting checks. Select one venue at a
time; this work does not authorize any submission.

[iclr]: https://iclr.cc/Conferences/2027/AuthorGuidelines
[iclr-cfp]: https://iclr.cc/Conferences/2027/CallForPapers
[icml]: https://icml.cc/Conferences/2026/AuthorInstructions
[neurips]: https://neurips.cc/Conferences/2026/CallForPapers
[tmlr]: https://www.jmlr.org/tmlr/author-guide.html
