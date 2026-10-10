"""PyTorch port of openpi-RLT's cross-attention encoder and decoder."""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
from torch.nn import functional


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    input_dim: int
    embed_dim: int = 512
    num_rl_tokens: int = 1
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    max_prefix_len: int = 1024
    dropout_rate: float = 0.0
    use_mask: bool = False

    def __post_init__(self):
        if (
            min(
                self.input_dim, self.embed_dim, self.num_rl_tokens, self.num_layers, self.max_prefix_len, self.num_heads
            )
            <= 0
        ):
            raise ValueError("RL-token dimensions and layer count must be positive")
        if self.embed_dim % self.num_heads or self.embed_dim % 2:
            raise ValueError("embed_dim must be even and divisible by num_heads")
        if not 0 <= self.dropout_rate < 1 or self.mlp_ratio <= 0:
            raise ValueError("Invalid dropout or MLP ratio")


def _positions(length: int, width: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000) / width))
    phase = position * frequency
    return torch.cat([phase.sin(), phase.cos()], dim=-1)


class GeGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, 2 * width)

    def forward(self, x):
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * functional.gelu(gate, approximate="tanh")


class CrossBlock(nn.Module):
    def __init__(self, cfg: RLTokenConfig):
        super().__init__()
        self.self_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout_rate, batch_first=True)
        self.q_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout_rate, batch_first=True)
        self.mlp_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, hidden),
            nn.Dropout(cfg.dropout_rate),
            GeGLU(hidden),
            nn.Linear(hidden, cfg.embed_dim),
        )

    def forward(self, query, memory, valid=None):
        normalized = self.self_norm(query)
        query = query + self.self_attn(normalized, normalized, normalized, need_weights=False)[0]
        query = (
            query
            + self.attn(
                self.q_norm(query),
                memory,
                memory,
                key_padding_mask=None if valid is None else ~valid,
                need_weights=False,
            )[0]
        )
        return query + self.mlp(self.mlp_norm(query))


class RLTokenReadout(nn.Module):
    def __init__(self, config: RLTokenConfig):
        super().__init__()
        self.config = config
        self.input_proj = (
            nn.Linear(config.input_dim, config.embed_dim) if config.input_dim != config.embed_dim else nn.Identity()
        )
        self.encoder_queries = nn.Parameter(_positions(config.num_rl_tokens, config.embed_dim))
        self.encoder_positions = nn.Parameter(_positions(config.max_prefix_len, config.embed_dim))
        self.decoder_queries = nn.Parameter(_positions(config.max_prefix_len, config.embed_dim))
        self.decoder_positions = nn.Parameter(_positions(config.num_rl_tokens, config.embed_dim))
        self.encoder = nn.ModuleList(CrossBlock(config) for _ in range(config.num_layers))
        self.decoder = nn.ModuleList(CrossBlock(config) for _ in range(config.num_layers))
        self.output_proj = (
            nn.Linear(config.embed_dim, config.input_dim) if config.input_dim != config.embed_dim else nn.Identity()
        )
        # Flax Dense/DenseGeneral use LeCun truncated-normal kernels and zero
        # biases. Initialize each fused Q/K/V matrix with its own fan-in.
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    self._lecun_normal(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.MultiheadAttention):
                    for weight in module.in_proj_weight.chunk(3, dim=0):
                        self._lecun_normal(weight)
                    nn.init.zeros_(module.in_proj_bias)

    @staticmethod
    def _lecun_normal(weight):
        std = weight.shape[1] ** -0.5 / 0.8796256610342398
        nn.init.trunc_normal_(weight, std=std, a=-2 * std, b=2 * std)

    def _check(self, prefix, mask):
        if prefix.ndim != 3 or prefix.shape[-1] != self.config.input_dim:
            raise ValueError(f"prefix must have shape [B,S,{self.config.input_dim}]")
        if mask is not None and (mask.shape != prefix.shape[:2] or mask.dtype != torch.bool):
            raise ValueError("mask must be boolean [B,S]")
        if prefix.shape[1] > self.config.max_prefix_len:
            raise ValueError("prefix exceeds max_prefix_len")
        if self.config.use_mask and (mask is None or not mask.any(dim=1).all()):
            raise ValueError("An example has no valid image tokens or mask is missing")

    def encode(self, prefix, mask=None):
        self._check(prefix, mask)
        memory = self.input_proj(prefix.float()) + self.encoder_positions[: prefix.shape[1]][None]
        query = self.encoder_queries[None].expand(prefix.shape[0], -1, -1)
        for layer in self.encoder:
            query = layer(query, memory, mask if self.config.use_mask else None)
        return query

    def decode(self, tokens, sequence_length):
        if sequence_length > self.config.max_prefix_len:
            raise ValueError("prefix exceeds max_prefix_len")
        query = self.decoder_queries[:sequence_length][None].expand(tokens.shape[0], -1, -1)
        memory = tokens + self.decoder_positions[None]
        for layer in self.decoder:
            query = layer(query, memory)
        return self.output_proj(query)

    def reconstruction_loss(self, prefix, mask=None):
        target = prefix.detach().float()
        tokens = self.encode(target, mask)
        error = (self.decode(tokens, prefix.shape[1]) - target).square()
        loss = (
            (error * mask[..., None]).sum() / (mask.sum() * prefix.shape[-1]) if self.config.use_mask else error.mean()
        )
        return loss, tokens
