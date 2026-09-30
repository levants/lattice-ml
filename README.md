# lattice-ml

This repository implements a lattice‐theoretic formal concept framework for
analyzing transcoder features in transformer‐based language models. Formal
Concept Analysis (FCA) provides a mathematical basis—using complete lattices
and Galois connections—for identifying and organizing interpretable “concepts”
as extents (sets of objects) and intents (sets of attributes). Transcoders
approximate densely activating MLP sublayers with sparsely activating ones,
enabling fine‐grained circuit analysis of hidden features. Sparse autoencoders
are used to learn efficient, sparse representations of neural activations,
highlighting the most salient features for interpretability. Transcoders build
on this by transforming dense activations into sparse feature patterns, a
strategy that has been applied not only to LLMs but also to vision models: in
language models, transcoders extract interpretable feature circuits from
transformer layers, while in vision, sparse autoencoders and transcoders enable
sparse circuit analysis of convolutional neural networks. By combining FCA with
transcoder outputs, we derive formal contexts where objects correspond to model
inputs or neuron activations and attributes correspond to transcoder features.
This yields a concept lattice that reveals how features co‐occur and propagate
across layers. The lattice facilitates operations such as join (least common
superconcept) and meet (greatest common subconcept), uncovering hierarchies
among feature sets and their neuron activations. Ultimately, this approach
provides interpretable insights into which transcoder features drive specific
behaviors or circuits inside GPT‐2‐style models.

## Computer vision feature analysis

The vision lattice paper and its reproducibility artifacts use the same
folder and Python package structure as the research source repository.

- [Paper and compiled PDF](texs/sparsesurrs/visionlattices/README.md).
- [Data, weights, activations, and queries](vision_tokens/README.md).
- [Python sources](src/lattmc/vision/).
- [Executed notebooks](notebooks/vision/).
- [Scientific review](texs/sparsesurrs/visionlattices/REVIEW.md).

The evidence includes three digit CNN/SAE runs and a frozen ResNet34 with a
Top-16 SAE on a 1,500-image CIFAR-10 sample. Measured galleries show animal
and vehicle features, spatial maps, and pooled versus same-site retrieval.
The paper cites Distill's investigative approach; no Distill figures are
copied and no causal circuit interpretation is claimed.

Start with `vision_tokens/README.md` for reproduction commands and the
isolated uv environment. Artifact hashes and migration provenance are in
`vision_tokens/provenance/`; upstream attribution is in
`vision_tokens/THIRD_PARTY.md`.

### Imagenette, Imagewoof, and a vision transformer

The extension adds 350 Imagenette photographs and 100 Imagewoof transfer
photographs, with original JPEGs, crops, and exact model activations.
Frozen ResNet34 and DINOv2 ViT-S/14 have separate Top-32 SAEs. Six new figures
compare dataset exemplars, eight optimized inputs, synthetic shape probes,
and rotation responses; numerical lattice-query extents are also provided.
See [the reproduction guide](vision_tokens/IMAGENETTE.md) and the executed
notebook `notebooks/vision/imagenette_features_codexgen.ipynb`.

The study follows Distill's multi-evidence approach to feature hypotheses.
It does not claim to recover the same curve detectors or causal circuits.

## Sparse vision features across datasets

The native Overcomplete and pretrained SAE/transcoder study is documented in
[the experiment README](vision_tokens/overcomplete/README.md). It includes
36 local fits, two additional pretrained surrogates, patch-level lattice
queries, cross-dataset galleries, and annotated controls. The executed
[notebook](notebooks/vision/overcomplete_features_codexgen.ipynb) reloads
measured evidence. Large weights and caches are supplied through
[Releases](https://github.com/levants/lattice-ml/releases), with pinned
upstream weights on Hugging Face.
