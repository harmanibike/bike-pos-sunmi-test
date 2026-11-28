from __future__ import annotations

import unittest
import sys
from pathlib import Path

# Match the import layout used by the existing model test suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:  # The lightweight corpus CI job does not install PyTorch.
    torch = None

from tools.model.transformer_config import TransformerConfig
from tools.model.train_transformer import build_model_class


@unittest.skipIf(torch is None, "PyTorch is installed by the dedicated runtime smoke workflow")
class TransformerRuntimeSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = TransformerConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            intermediate_size=32,
            context_length=8,
            tie_embeddings=True,
            use_bias=True,
        )
        self.config.validate()
        model_class = build_model_class(torch)
        self.model = model_class(self.config)

    def test_forward_shape_and_finite_logits(self):
        tokens = torch.randint(0, self.config.vocab_size, (2, 8))
        logits = self.model(tokens)
        self.assertEqual((2, 8, self.config.vocab_size), tuple(logits.shape))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_future_tokens_do_not_change_previous_logits(self):
        self.model.eval()
        original = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        changed = original.clone()
        changed[0, 6] = 19
        with torch.no_grad():
            original_logits = self.model(original)
            changed_logits = self.model(changed)
        torch.testing.assert_close(
            original_logits[:, :6, :],
            changed_logits[:, :6, :],
            rtol=0,
            atol=1e-6,
        )

    def test_backward_and_optimizer_update_are_finite(self):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        tokens = torch.randint(0, self.config.vocab_size, (2, 8))
        targets = torch.randint(0, self.config.vocab_size, (2, 8))
        before = self.model.token_embedding.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        logits = self.model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
        )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [
            parameter.grad for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
        self.assertTrue(any(gradient.abs().sum().item() > 0 for gradient in gradients))
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.token_embedding.weight.detach()))


if __name__ == "__main__":
    unittest.main()
