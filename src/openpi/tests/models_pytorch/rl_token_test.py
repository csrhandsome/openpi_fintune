"""Masked PyTorch readout reconstruction and gradients."""

import pytest
import torch

from openpi.models_pytorch.rl_token import RLTokenConfig
from openpi.models_pytorch.rl_token import RLTokenReadout


def test_masked_reconstruction_ignores_invalid_target():
    torch.manual_seed(3)
    readout = RLTokenReadout(RLTokenConfig(input_dim=8, embed_dim=8, num_heads=2, max_prefix_len=5, use_mask=True))
    prefix = torch.randn(2, 4, 8, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    loss, tokens = readout.reconstruction_loss(prefix, mask)
    changed = prefix.detach().clone()
    changed[~mask] = 1e6
    changed_loss, _ = readout.reconstruction_loss(changed, mask)
    torch.testing.assert_close(loss, changed_loss)
    loss.backward()
    assert prefix.grad is None
    assert readout.encoder_queries.grad is not None
    assert tokens.shape == (2, 1, 8)
    with pytest.raises(ValueError, match="no valid"):
        readout.encode(prefix, torch.zeros_like(mask))
