# Scientific review and submission assessment

## Assessment

The manuscript is a coherent exploratory methods paper with proved
statements and an executed small vision pilot. It is suitable for author
review toward an arXiv methods preprint. It is not currently a competitive
submission to a prestigious mathematics journal or a main-track vision or
machine-learning conference. Compilation and reproducibility establish
technical integrity, not novelty or acceptance prospects.

Before an arXiv release, check author metadata and settle the public
citation for the companion. Its current bibliography entry explicitly says
unpublished companion manuscript, without inventing an arXiv identifier.
The reproducibility artifacts and this manuscript are mirrored to the
public Lattice-ML repository. No arXiv upload or venue submission is made.

## Content review

The argument retains the source paper's relation-induced Galois ideal,
antitone adjunction, row/column derivation proof, canonical concept maps,
and ordinal-scaling equivalence. All prerequisites are now in this paper.
Language-model experiments and their numeric claims were not transplanted.

The spatial extension distinguishes separate-site witnesses from a common
site, proves the single-vector criterion, and recovers exact closure through
downsets. These are elementary consequences of existing order theory.
The paper explicitly cites pattern structures and does not present the
adjunction or downset construction as a new general mathematical theory.
An exhaustive novelty review of all spatial pattern-structure literature
has not been completed; no priority claim is justified.

The three trained CNN/SAE pairs and 150 queries are real executed evidence.
Graded retrieval improves on the specified presence baseline but loses to
both cosine baselines. The presence baseline uses amplitude-informed
coordinate selection; this limitation is disclosed. No statistical
significance claim is made from overlapping query draws or shared splits.
The paired training seeds change CNN and SAE together.

The natural-image extension adds six measured feature galleries, class
response profiles, and cat/dog and ship/truck lattice comparisons. A frozen
ResNet34 supplies the dense features on 1,500 CIFAR-10 images. The
surrogate reconstruction R2 is 0.576 and only 165/512 features are active
on training data; this is illustrative evidence with limited fidelity.

The greatest remaining empirical weaknesses are the small digit benchmark,
deterministic natural-image subset, absence of ViTs and transcoders,
many inactive SAE coordinates,
no spatial semantic annotations, no feature-specific causal intervention,
and no full evaluation of downset mining. Reconstruction replacement is a
faithfulness diagnostic, not validation of individual feature meanings.
The experiment does not compare against the original spatial classifier
as a retrieval baseline, and max pooling discards useful digit geometry.

## Priorities for a stronger submission

1. Evaluate fixed CLIP and DINO-family backbones with matching public SAEs,
   and matched CLIP transcoders if the paper claims a surrogate comparison.
2. Add independent natural-image datasets and spatially annotated queries.
   Split by original image before generating patches or crops.
3. Match reconstruction quality and sparsity; investigate inactive features.
   Repeat backbone and surrogate seeds independently.
4. Compare coordinate selection methods, pure support-only selection,
   ordinal FCA, dense/sparse cosine, and region-aware retrieval. Report
   whether downset descriptions add useful evidence beyond site filtering.
5. Add controlled edits with random and norm-matched controls if causal
   interpretation is claimed. Use hierarchical uncertainty over appropriate
   units instead of treating all query draws as independent samples.
6. Strengthen the mathematical contribution with an efficient algorithm,
   approximation guarantees, or a substantial new structural result if
   targeting a theory journal. The current elementary results are not enough
   to support a prestigious pure-mathematics submission.

## Venues, ranked by fit

These are editorial judgments based on official scope pages checked on
2026-09-30. They are not predictions of acceptance or claims of open calls.

- **CONCEPTS (ICFCA/CLA/ICCS community)** is the closest audience for the
  present FCA application and spatial-description analysis. A focused
  application or methods contribution is more realistic than a broad
  frontier-vision claim. The 2026 edition already took place on
  August 31--September 4; consider the next announced edition.
  [Official conference](https://concepts2026.org/).
- **Annals of Mathematics and Artificial Intelligence** is a plausible
  journal after strengthening novelty and validation. Its stated scope
  includes algebraic and algorithmic methods for machine learning and
  computer vision.
  [Scope](https://link.springer.com/journal/10472/aims-and-scope).
- **Order** is conditional on substantial new order theory. Its scope fits
  ordered structures and computing, but the current reused FCA machinery
  and elementary spatial lemmas are insufficient as a theory contribution.
  [Scope](https://link.springer.com/journal/11083/aims-and-scope).
- **CVPR or ICLR main tracks** are longer-term targets only after large-model
  experiments, stronger comparative evidence, and a clear contribution over
  PatchSAE, SAEV, and Prisma. An appropriate interpretability workshop is
  a more plausible intermediate venue; check its next actual call.
  [CVPR scope](https://cvpr.thecvf.com/Conferences/2026/CallForPapers) and
  [ICLR scope](https://iclr.cc/Conferences/2026/CallForPapers).

See `VERIFICATION.md` for the final artifact checks and `LIBRARIES.md` for
researched extension options. A longer paper alone will not resolve the
novelty and empirical limitations identified above.
