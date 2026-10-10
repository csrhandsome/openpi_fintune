"""Numerical parity with the Flax blocks used by openpi-RLT."""

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from openpi.models.utils.fsq_tokenizer import CrossAttentionLayer
from openpi.models.utils.fsq_tokenizer import sinusoidal_pe_init
from openpi.models_pytorch.rl_token import RLTokenConfig
from openpi.models_pytorch.rl_token import RLTokenReadout

# GPU float32 matmul defaults to TF32, which moves this comparison by about 1e-3.
jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class ReferenceSide(nn.Module):
    cfg: RLTokenConfig
    decode: bool = False

    @nn.compact
    def __call__(self, memory, *, mask=None):
        cfg = self.cfg
        length = cfg.max_prefix_len if self.decode else cfg.num_rl_tokens
        if not self.decode and cfg.input_dim != cfg.embed_dim:
            memory = nn.Dense(cfg.embed_dim, name="input_proj")(memory)
        query = self.param("q_embed", sinusoidal_pe_init, (length, cfg.embed_dim))
        query = jnp.broadcast_to(query, (memory.shape[0], *query.shape))
        memory = memory + self.param("y_pos_enc", sinusoidal_pe_init, memory.shape[-2:])
        attention_mask = None if mask is None else mask[:, None, None, :]
        for _ in range(cfg.num_layers):
            query = CrossAttentionLayer(num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio)(
                query, memory, train=False, mask_cross=attention_mask
            )
        if self.decode and cfg.input_dim != cfg.embed_dim:
            query = nn.Dense(cfg.input_dim, name="output_proj")(query)
        return query


class ReferenceReadout(nn.Module):
    cfg: RLTokenConfig

    def setup(self):
        self.encoder = ReferenceSide(self.cfg)
        self.decoder = ReferenceSide(self.cfg, decode=True)

    def __call__(self, prefix, mask=None):
        tokens = self.encoder(prefix, mask=mask)
        reconstruction = self.decoder(tokens)
        error = (reconstruction - jax.lax.stop_gradient(prefix)) ** 2
        loss = error.mean() if mask is None else (error * mask[..., None]).sum() / (mask.sum() * prefix.shape[-1])
        return loss, tokens, reconstruction


def copy_reference_weights(readout, params):
    def copy(destination, source):
        destination.copy_(torch.from_numpy(np.array(source)))

    def dense(destination, source):
        copy(destination.weight, source["kernel"].T)
        copy(destination.bias, source["bias"])

    with torch.no_grad():
        for label in ("encoder", "decoder"):
            side = params[label]
            copy(getattr(readout, label + "_queries"), side["q_embed"])
            copy(getattr(readout, label + "_positions"), side["y_pos_enc"])
            projection = "input_proj" if label == "encoder" else "output_proj"
            if projection in side:
                dense(getattr(readout, projection), side[projection])
            for index, block in enumerate(getattr(readout, label)):
                source = side[f"CrossAttentionLayer_{index}"]
                for i, norm in enumerate((block.self_norm, block.q_norm, block.mlp_norm)):
                    copy(norm.weight, source[f"LayerNorm_{i}"]["scale"])
                    copy(norm.bias, source[f"LayerNorm_{i}"]["bias"])
                for i, attention in enumerate((block.self_attn, block.attn)):
                    weights = source[f"MultiHeadDotProductAttention_{i}"]
                    copy(
                        attention.in_proj_weight,
                        jnp.concatenate(
                            [
                                weights[name]["kernel"].reshape(readout.config.embed_dim, -1).T
                                for name in ("query", "key", "value")
                            ]
                        ),
                    )
                    copy(
                        attention.in_proj_bias,
                        jnp.concatenate([weights[name]["bias"].reshape(-1) for name in ("query", "key", "value")]),
                    )
                    copy(attention.out_proj.weight, weights["out"]["kernel"].reshape(-1, readout.config.embed_dim).T)
                    copy(attention.out_proj.bias, weights["out"]["bias"])
                dense(block.mlp[0], source["Dense_0"])
                dense(block.mlp[2].proj, source["GeGLU_0"]["Dense_0"])
                dense(block.mlp[3], source["Dense_1"])


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("input_dim", [6, 8])
def test_readout_outputs_loss_and_gradients(input_dim, use_mask):
    cfg = RLTokenConfig(
        input_dim=input_dim,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        num_rl_tokens=2,
        mlp_ratio=2,
        max_prefix_len=4,
        use_mask=use_mask,
    )
    prefix = np.random.default_rng(0).normal(size=(2, 4, input_dim)).astype(np.float32)
    mask = np.array([[True, True, False, False], [True, True, True, False]]) if use_mask else None
    reference = ReferenceReadout(cfg)
    params = reference.init(jax.random.key(1), jnp.array(prefix), mask)["params"]
    readout = RLTokenReadout(cfg)
    copy_reference_weights(readout, params)
    expected = reference.apply({"params": params}, prefix, mask)
    torch_prefix = torch.tensor(prefix, requires_grad=True)
    torch_mask = None if mask is None else torch.tensor(mask)
    loss, tokens = readout.reconstruction_loss(torch_prefix, torch_mask)
    reconstruction = readout.decode(tokens, 4)
    for actual, desired in zip((loss, tokens, reconstruction), expected, strict=True):
        torch.testing.assert_close(actual, torch.from_numpy(np.array(desired)), rtol=2e-5, atol=2e-5)
    gradients = jax.grad(lambda p: reference.apply({"params": p}, prefix, mask)[0])(params)
    loss.backward()
    assert torch_prefix.grad is None
    torch.testing.assert_close(
        readout.encoder_queries.grad, torch.from_numpy(np.array(gradients["encoder"]["q_embed"])), rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(
        readout.decoder_queries.grad, torch.from_numpy(np.array(gradients["decoder"]["q_embed"])), rtol=2e-4, atol=2e-5
    )
