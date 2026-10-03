"""Check activation-site semantics independently of downloaded weights."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from .families_adapters_codexgen import attach


class Block(torch.nn.Module):
    def __init__(self: Block) -> None:
        """Initialize Block and its required state."""
        super().__init__()
        self.mlp = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.mlp.weight.copy_(2 * torch.eye(2))

    def forward(self: Block, x: torch.Tensor) -> torch.Tensor:
        """Apply the toy model used to verify activation-hook locations."""
        return x + self.mlp(x + 3)


class Model(torch.nn.Module):
    def __init__(self: Model) -> None:
        """Initialize Model and its required state."""
        super().__init__()
        self.model = SimpleNamespace(layers=[Block()])

    def forward(
        self: Model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> torch.Tensor:
        """Apply the toy model used to verify activation-hook locations."""
        return self.model.layers[0](input_ids.float())


class HookTests(unittest.TestCase):
    def test_transcoder_input_and_target_are_distinct(self: HookTests) -> None:
        """Verify transcoder input and target are distinct."""
        run = attach(Model(), dict(layer=0, hook='mlp_input_output'))
        ids = torch.tensor([[[1, 2]]])
        x, y = run(ids, torch.ones(1, 1, dtype=torch.bool))
        torch.testing.assert_close(x, ids.float() + 3)
        torch.testing.assert_close(y, 2 * (ids.float() + 3))

    def test_residual_output_includes_skip(self: HookTests) -> None:
        """Verify residual output includes skip."""
        run = attach(Model(), dict(layer=0, hook='residual_post'))
        ids = torch.tensor([[[1, 2]]])
        x, y = run(ids, torch.ones(1, 1, dtype=torch.bool))
        torch.testing.assert_close(x, 3 * ids.float() + 6)
        torch.testing.assert_close(x, y)

    def test_mlp_sae_excludes_skip(self: HookTests) -> None:
        """Verify mlp sae excludes skip."""
        run = attach(Model(), dict(layer=0, hook='mlp_output'))
        ids = torch.tensor([[[1, 2]]])
        x, y = run(ids, torch.ones(1, 1, dtype=torch.bool))
        torch.testing.assert_close(x, 2 * (ids.float() + 3))
        torch.testing.assert_close(x, y)


if __name__ == '__main__':
    unittest.main()
