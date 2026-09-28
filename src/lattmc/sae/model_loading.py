"""Loading HookedTransformer models for SAE and transcoders 
for Lattice-theoretic Formal Concept Analysis (FCA)."""

import logging

from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)

model = HookedTransformer.from_pretrained("gpt2-small")

logits, activations = model.run_with_cache("Hello World")

logger.info("Logits shape:", logits.shape)
logger.info("Activations keys:", activations.keys())
