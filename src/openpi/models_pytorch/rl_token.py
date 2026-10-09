"""PyTorch RL-token readout trained separately from a frozen Pi0 backbone."""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    input_dim: int
    embed_dim: int = 512
    num_rl_tokens: int = 1
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    max_prefix_len: int = 1024

    def __post_init__(self):
        if min(self.input_dim, self.embed_dim, self.num_rl_tokens, self.num_layers, self.max_prefix_len) <= 0:
            raise ValueError("RL-token dimensions and layer count must be positive")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")


def _positions(length: int, width: int, *, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000) / width))
    phase = position * frequency
    out = torch.zeros(length, width, device=device)
    out[:, 0::2] = phase.sin()
    out[:, 1::2] = phase.cos()[:, : out[:, 1::2].shape[1]]
    return out


class CrossBlock(nn.Module):
    def __init__(self, cfg: RLTokenConfig):
        super().__init__()
        self.self_norm = nn.LayerNorm(cfg.embed_dim)
        self.self_attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, batch_first=True)
        self.q_norm = nn.LayerNorm(cfg.embed_dim)
        self.kv_norm = nn.LayerNorm(cfg.embed_dim)
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(cfg.embed_dim)
        hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(cfg.embed_dim, hidden), nn.GELU(), nn.Linear(hidden, cfg.embed_dim))

    def forward(self, query: torch.Tensor, memory: torch.Tensor, valid: torch.Tensor | None = None):
        normalized = self.self_norm(query)
        self_update, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        query = query + self_update
        update, _ = self.attn(
            self.q_norm(query),
            self.kv_norm(memory),
            self.kv_norm(memory),
            key_padding_mask=None if valid is None else ~valid,
            need_weights=False,
        )
        query = query + update
        return query + self.mlp(self.mlp_norm(query))


class RLTokenReadout(nn.Module):
    """Cross-attention compressor and masked prefix reconstruction head."""

    def __init__(self, config: RLTokenConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.embed_dim)
        self.encoder_queries = nn.Parameter(torch.randn(config.num_rl_tokens, config.embed_dim) * 0.02)
        self.encoder = nn.ModuleList(CrossBlock(config) for _ in range(config.num_layers))
        self.decoder = nn.ModuleList(CrossBlock(config) for _ in range(config.num_layers))
        self.output_proj = nn.Linear(config.embed_dim, config.input_dim)

    def _check(self, prefix: torch.Tensor, mask: torch.Tensor):
        if prefix.ndim != 3 or prefix.shape[-1] != self.config.input_dim:
            raise ValueError(f"prefix must have shape [B,S,{self.config.input_dim}]")
        if mask.shape != prefix.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean [B,S]")
        if prefix.shape[1] > self.config.max_prefix_len or not mask.any(dim=1).all():
            raise ValueError("prefix is too long or an example has no valid image tokens")

    def encode(self, prefix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check(prefix, mask)
        memory = self.input_proj(prefix.float())
        memory = memory + _positions(prefix.shape[1], self.config.embed_dim, device=prefix.device)[None]
        query = self.encoder_queries[None].expand(prefix.shape[0], -1, -1)
        for layer in self.encoder:
            query = layer(query, memory, mask)
        return query

    def decode(self, tokens: torch.Tensor, sequence_length: int) -> torch.Tensor:
        if sequence_length > self.config.max_prefix_len:
            raise ValueError("prefix exceeds max_prefix_len")
        query = _positions(sequence_length, self.config.embed_dim, device=tokens.device)[None].expand(
            tokens.shape[0], -1, -1
        )
        memory = tokens + _positions(tokens.shape[1], self.config.embed_dim, device=tokens.device)[None]
        for layer in self.decoder:
            query = layer(query, memory)
        return self.output_proj(query)

    def reconstruction_loss(self, prefix: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The VLA is frozen even if a caller accidentally leaves its parameters trainable.
        target = prefix.detach().float()
        tokens = self.encode(target, mask)
        reconstructed = self.decode(tokens, prefix.shape[1])
        loss = ((reconstructed - target).square() * mask[..., None]).sum() / (mask.sum() * prefix.shape[-1])
        return loss, tokens
