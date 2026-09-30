---
language: en
tags:
- clip
- vision
- transformers
- interpretability
- sparse autoencoder
- sae
- mechanistic interpretability
license: apache-2.0
library_name: torch
pipeline_tag: feature-extraction
metrics:
- type: explained_variance 
  value: 98.5
  pretty_name: Explained Variance %
  range:
    min: 0
    max: 100
- type: l0
  value: 1408.150
  pretty_name: L0 
---

# CLIP-B-32 Sparse Autoencoder x64 vanilla - L1:1e-05

![Explained Variance](https://img.shields.io/badge/Explained%20Variance-98.5%25-blue)
![Sparsity](https://img.shields.io/badge/Active%20Features-140815.0%-green)

### Training Details

- Base Model: CLIP-ViT-B-32 (LAION DataComp.XL-s13B-b90K)
- Layer: 11
- Component: hook_resid_post

### Model Architecture

- Input Dimension: 768
- SAE Dimension: 49,152
- Expansion Factor: x64 (vanilla architecture)
- Activation Function: ReLU
- Initialization: encoder_transpose_decoder
- Context Size: 50 tokens

### Performance Metrics

- L1 Coefficient: 1e-05
- L0 Sparsity: 1408.1500
- Explained Variance: 0.9848 (98.48%)

### Training Configuration

- Learning Rate: 0.0004
- LR Scheduler: Cosine Annealing with Warmup (200 steps)
- Epochs: 10
- Gradient Clipping: 1.0
- Device: NVIDIA Quadro RTX 8000

**Experiment Tracking:**
- Weights & Biases Run ID: j4bop08g
- Full experiment details: https://wandb.ai/perceptual-alignment/clip/runs/j4bop08g/overview
- Git Commit: e22dd02726b74a054a779a4805b96059d83244aa

## Citation

```bibtex
@misc{2024josephsparseautoencoders,
    title={Sparse Autoencoders for CLIP-ViT-B-32},
    author={Joseph, Sonia},
    year={2024},
    publisher={Prisma-Multimodal},
    url={https://huggingface.co/Prisma-Multimodal},
    note={Layer 11, hook_resid_post, Run ID: j4bop08g}
}
