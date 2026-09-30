# Vision sparse-surrogate resources

Reviewed on 2026-09-30 against project repositories and model collections.
These are extension candidates. The executed studies use the local
PyTorch digit CNN/SAE and a pretrained torchvision ResNet34 with a
locally trained SAE, plus a Hugging Face Transformers DINOv2 ViT-S/14
with its own SAE. ViT-Prisma, SAEV, and Overcomplete remain researched
options; those three packages were not used for the experiments.

## Recommended first choice: ViT-Prisma

[Repository](https://github.com/Prisma-Multimodal/ViT-Prisma) and
[paper](https://arxiv.org/abs/2504.19475).

Its documented scope includes vision/video activation access, SAE training,
and pretrained CLIP/DINO SAEs plus CLIP transcoders. This is the closest
match to the companion manuscript's SAE/transcoder comparison. The README
links a registry of hooks and pretrained dictionaries. Follow the current
loading notebook rather than copying its abbreviated README snippets;
those snippets include inconsistent Hugging Face keyword names.

A concrete checkpoint family is the repository's linked CLIP ViT-B/32
layer-11 residual-post vanilla SAE (expansion 64, L1 coefficient 1e-5).
Find its exact model card through the repository's pretrained-weight table.
The model card was checked; it was not downloaded or executed here.

## Alternative: SAEV

[Repository](https://github.com/Imageomics/saev),
[research paper](https://arxiv.org/abs/2502.06755), and
[checkpoint collection](https://huggingface.co/collections/osunlp/saev).

This is a focused vision SAE workflow with inference examples. The
collection includes `osunlp/SAE_CLIP_24K_ViT-B-16_IN1K` and
`osunlp/SAE_DINOv2_24K_ViT-B-14_IN1K`. Their existence in the collection
was verified; compatibility with this environment has not been tested.

## Alternative: Overcomplete

[Repository](https://github.com/KempnerInstitute/overcomplete).

A PyTorch toolbox for vision dictionary learning, multiple SAE variants,
visualizations, and evaluation. Prefer this when training and comparing
surrogate variants is central. It is not a drop-in guarantee that any
arbitrary pretrained SAE is compatible with a given vision checkpoint.

## Activation-cache contract

The code's input is an array `(images, spatial_sites, latent_coordinates)`
with finite nonnegative values. Maintain the following metadata alongside
external caches:

- Exact backbone and surrogate identifiers and immutable revisions.
- Image identifiers, dataset version, preprocessing, and split indices.
- Layer/hook, input versus output state, and any centering or rescaling.
- Grid coordinates and explicit treatment of CLS and register tokens.
- Surrogate architecture, sparsity, decoder conventions, and code dtype.
- Reconstruction and downstream replacement diagnostics.

Pass valid arrays to `pooled_codes` and `spatial_extents` in
`src/lattmc/vision/contexts_codexgen.py`. Variable-size site sets need
separate per-image evaluation or documented padding. Zero padding preserves
nonnegative max pooling and nonzero-query existence, but do not use it to
infer real spatial witnesses for the zero query. Preserve at least one
actual site per image.

Do not clip negative codes silently: either use a nonnegative surrogate or
explicitly define a different ordered description space. Do not align
features across models by coordinate index. Do not infer a transformer's
causal receptive field from the patch grid.


## Executed transformer adapter: Hugging Face Transformers

The extension uses `facebook/dinov2-small` at immutable revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`, loaded locally with eager
attention. It takes final layer-normalized patch tokens and excludes CLS.
The original checkpoint, configuration, upstream license, and hashes are
released under `vision_tokens/imagenette_dinov2/checkpoints/`.
See `vision_tokens/IMAGENETTE.md` for the exact input and SAE conventions.

[Model card](https://huggingface.co/facebook/dinov2-small) and
[upstream model repository](https://github.com/facebookresearch/dinov2).
