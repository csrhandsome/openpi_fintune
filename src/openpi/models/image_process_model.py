"""CBAM modules for force-image feature extraction."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


@at.typecheck
class ChannelAttention(nn.Module):
    """Channel attention in CBAM."""

    reduction_ratio: int = 16
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        channels = x.shape[-1]
        hidden_dim = max(channels // self.reduction_ratio, 1)

        mlp_hidden = nn.Dense(hidden_dim, dtype=self.dtype, name="mlp_hidden")
        mlp_out = nn.Dense(channels, dtype=self.dtype, name="mlp_out")

        avg_pool = jnp.mean(x, axis=(1, 2))
        max_pool = jnp.max(x, axis=(1, 2))

        avg_attn = mlp_out(nn.relu(mlp_hidden(avg_pool)))
        max_attn = mlp_out(nn.relu(mlp_hidden(max_pool)))
        attn = nn.sigmoid(avg_attn + max_attn)[:, None, None, :]
        return x * attn


@at.typecheck
class SpatialAttention(nn.Module):
    """Spatial attention in CBAM."""

    kernel_size: int = 7
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        avg_pool = jnp.mean(x, axis=-1, keepdims=True)
        max_pool = jnp.max(x, axis=-1, keepdims=True)
        pooled = jnp.concatenate([avg_pool, max_pool], axis=-1)
        attn = nn.Conv(
            features=1,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="SAME",
            dtype=self.dtype,
            name="conv",
        )(pooled)
        return x * nn.sigmoid(attn)


@at.typecheck
class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    reduction_ratio: int = 16
    spatial_kernel_size: int = 7
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = ChannelAttention(reduction_ratio=self.reduction_ratio, dtype=self.dtype, name="channel")(x)
        x = SpatialAttention(kernel_size=self.spatial_kernel_size, dtype=self.dtype, name="spatial")(x)
        return x


@at.typecheck
class TemporalGRUEncoder(nn.Module):
    """Temporal GRU over per-frame force features."""

    hidden_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"TemporalGRUEncoder expects [B, T, C], got shape {x.shape}")

        batch_size = x.shape[0]
        init_carry = jnp.zeros((batch_size, self.hidden_dim), dtype=self.dtype)
        gru_cell = nn.GRUCell(features=self.hidden_dim, dtype=self.dtype, name="gru_cell")

        def _step(carry: jnp.ndarray, x_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            next_carry, out = gru_cell(carry, x_t)
            return next_carry, out

        final_carry, _ = jax.lax.scan(_step, init_carry, jnp.swapaxes(x, 0, 1))
        return final_carry
