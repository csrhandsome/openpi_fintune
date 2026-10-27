"""Image JEPA with rectangular targets and an exponential moving average teacher.

Inputs are RGB tensors normalized by the dataset. Mask indices are sampled on
CPU, shared by a minibatch, and moved to the image device before use. Sharing
indices keeps all sequences dense without padding or exposing masked pixels.
"""

import copy
from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class JEPAMasks:
    context: torch.Tensor
    targets: tuple[torch.Tensor, ...]


def sample_masks(
    grid_size: int,
    *,
    num_local_targets: int = 2,
    num_global_targets: int = 2,
    local_scale: tuple[float, float] = (0.05, 0.10),
    global_scale: tuple[float, float] = (0.15, 0.20),
    max_target_overlap: float = 0.25,
    generator: torch.Generator | None = None,
) -> JEPAMasks:
    """Sample rectangles, retaining at least one context patch.

    Pairwise intersection divided by the smaller rectangle's area is bounded
    by ``max_target_overlap``. Random search is followed by an exhaustive
    fallback for the next rectangle. Unsatisfiable settings fail explicitly;
    neither overlap nor context exclusion is silently relaxed.
    """
    if grid_size < 2:
        raise ValueError("JEPA requires a patch grid of at least 2 by 2")
    if num_local_targets < 0 or num_global_targets < 0 or num_local_targets + num_global_targets == 0:
        raise ValueError("At least one target rectangle is required")
    if not 0 <= max_target_overlap <= 1:
        raise ValueError("max_target_overlap must be between 0 and 1")
    total = grid_size**2
    target_sets: list[set[int]] = []
    occupied: set[int] = set()
    scales = [local_scale] * num_local_targets + [global_scale] * num_global_targets
    for scale in scales:
        if len(scale) != 2 or not 0 < scale[0] <= scale[1] <= 1:
            raise ValueError(f"Invalid target area scale: {scale}")
        min_area = max(1, round(total * scale[0]))
        max_area = max(min_area, round(total * scale[1]))
        shapes = [
            (height, width)
            for height in range(1, grid_size + 1)
            for width in range(1, grid_size + 1)
            if min_area <= height * width <= max_area and 0.5 <= height / width <= 2.0
        ]
        if not shapes:
            raise ValueError(f"No rectangular patch mask fits scale {scale} on a {grid_size} by {grid_size} grid")

        def acceptable(candidate: set[int]) -> bool:
            return len(occupied | candidate) < total and all(
                len(previous & candidate) / min(len(previous), len(candidate)) <= max_target_overlap
                for previous in target_sets
            )

        def rectangle(height: int, width: int, top: int, left: int) -> set[int]:
            return {row * grid_size + col for row in range(top, top + height) for col in range(left, left + width)}

        chosen = None
        for _ in range(128):
            height, width = shapes[int(torch.randint(len(shapes), (), generator=generator))]
            top = int(torch.randint(grid_size - height + 1, (), generator=generator))
            left = int(torch.randint(grid_size - width + 1, (), generator=generator))
            candidate = rectangle(height, width, top, left)
            if acceptable(candidate):
                chosen = candidate
                break
        if chosen is None:
            for index in torch.randperm(len(shapes), generator=generator).tolist():
                height, width = shapes[index]
                for top in range(grid_size - height + 1):
                    for left in range(grid_size - width + 1):
                        candidate = rectangle(height, width, top, left)
                        if acceptable(candidate):
                            chosen = candidate
                            break
                    if chosen is not None:
                        break
                if chosen is not None:
                    break
        if chosen is None:
            raise ValueError(
                "Could not place JEPA target rectangles while preserving context and overlap bounds; "
                "reduce target count/area or increase max_target_overlap"
            )
        occupied.update(chosen)
        target_sets.append(chosen)
    return JEPAMasks(
        context=torch.tensor(sorted(set(range(total)) - occupied), dtype=torch.long),
        targets=tuple(torch.tensor(sorted(target), dtype=torch.long) for target in target_sets),
    )


def _position_embedding(grid_size: int, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("Embedding dimensions must be divisible by four for 2D sinusoidal positions")
    axis = torch.arange(grid_size, dtype=torch.float32)
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    frequency = 1.0 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)))
    positions = [coordinate.flatten()[:, None] * frequency for coordinate in (row, col)]
    return torch.cat([value for position in positions for value in (position.sin(), position.cos())], dim=-1)[None]


def _transformer(dim: int, depth: int, heads: int, mlp_ratio: float) -> nn.TransformerEncoder:
    if depth < 1 or dim % heads:
        raise ValueError("Transformer depth must be positive and dimension divisible by number of heads")
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=int(dim * mlp_ratio),
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    transformer = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False)
    # TransformerEncoder clones the first layer; initialize each clone independently.
    for module in transformer.modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
    return transformer


class ImageEncoder(nn.Module):
    """Vision transformer exposing pooled embeddings and spatial patch tokens.

    ``forward(images)`` returns ``[B, D]`` mean-pooled embeddings.
    ``forward_features(images)`` returns ``[B, H/P * W/P, D]`` patch tokens.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        self.config = {
            "image_size": image_size,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
        }
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.register_buffer("pos_embed", _position_embedding(self.grid_size, embed_dim))
        self.blocks = _transformer(embed_dim, depth, num_heads, mlp_ratio)

    def forward_features(self, images: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected RGB images [B, 3, {self.image_size}, {self.image_size}], got {images.shape}")
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype)
        if indices is not None:
            tokens = tokens.index_select(1, indices)
        return self.blocks(tokens)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images).mean(dim=1)


class ImageJEPA(nn.Module):
    """Predict normalized teacher features using only visible image patches."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        predictor_depth: int = 4,
        predictor_dim: int = 192,
        mlp_ratio: float = 4.0,
        num_local_targets: int = 2,
        num_global_targets: int = 2,
        local_scale: tuple[float, float] = (0.05, 0.10),
        global_scale: tuple[float, float] = (0.15, 0.20),
        max_target_overlap: float = 0.25,
    ):
        super().__init__()
        self.context_encoder = ImageEncoder(image_size, patch_size, embed_dim, depth, num_heads, mlp_ratio)
        self.target_encoder = copy.deepcopy(self.context_encoder).requires_grad_(requires_grad=False).eval()
        self.context_projection = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.register_buffer("predictor_pos_embed", _position_embedding(self.context_encoder.grid_size, predictor_dim))
        self.predictor = _transformer(predictor_dim, predictor_depth, num_heads, mlp_ratio)
        self.prediction_head = nn.Linear(predictor_dim, embed_dim)
        self.mask_config = {
            "num_local_targets": num_local_targets,
            "num_global_targets": num_global_targets,
            "local_scale": local_scale,
            "global_scale": global_scale,
            "max_target_overlap": max_target_overlap,
        }
        self.model_config = {
            **self.context_encoder.config,
            "predictor_depth": predictor_depth,
            "predictor_dim": predictor_dim,
            **self.mask_config,
        }

    def train(self, mode: bool = True):  # noqa: FBT001, FBT002
        super().train(mode)
        self.target_encoder.eval()
        return self

    def sample_masks(self, generator: torch.Generator | None = None) -> JEPAMasks:
        return sample_masks(self.context_encoder.grid_size, generator=generator, **self.mask_config)

    def forward(self, images: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        masks = self.sample_masks(generator)
        context_indices = masks.context.to(images.device)
        target_indices = torch.unique(torch.cat(masks.targets)).to(images.device)
        with torch.no_grad():
            targets = self.target_encoder.forward_features(images).index_select(1, target_indices)
            targets = functional.layer_norm(targets.float(), (targets.shape[-1],))
        context = self.context_projection(self.context_encoder.forward_features(images, context_indices))
        position = self.predictor_pos_embed.to(dtype=context.dtype)
        context = context + position.index_select(1, context_indices)
        target_tokens = self.mask_token.to(dtype=context.dtype) + position.index_select(1, target_indices)
        target_tokens = target_tokens.expand(images.shape[0], -1, -1)
        predictions = self.predictor(torch.cat((context, target_tokens), dim=1))[:, context.shape[1] :]
        predictions = self.prediction_head(predictions)
        return {
            "loss": functional.mse_loss(predictions.float(), targets),
            "target_std": targets.flatten(0, 1).std(dim=0, unbiased=False).mean().detach(),
            "prediction_std": predictions.detach().float().flatten(0, 1).std(dim=0, unbiased=False).mean(),
            "context_tokens": images.new_tensor(context_indices.numel()),
            "target_tokens": images.new_tensor(target_indices.numel()),
        }

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        if not math.isfinite(momentum) or not 0 <= momentum <= 1:
            raise ValueError("Teacher momentum must be between 0 and 1")
        for teacher, student in zip(self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True):
            teacher.lerp_(student, 1.0 - momentum)
        for teacher, student in zip(self.target_encoder.buffers(), self.context_encoder.buffers(), strict=True):
            teacher.copy_(student)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, *, pool: bool = True, use_teacher: bool = True) -> torch.Tensor:
        encoder = self.target_encoder if use_teacher else self.context_encoder
        tokens = encoder.forward_features(images)
        return tokens.mean(dim=1) if pool else tokens


def load_encoder(checkpoint: str | Path, device: str | torch.device = "cpu") -> ImageEncoder:
    """Load the EMA encoder from a training checkpoint, ready for inference.

    Checkpoints contain ``model_config`` and ``model`` (the full JEPA state
    dictionary). Only encoder weights are loaded; optimizer state is unused.
    Apply the training RGB resize and normalization before calling the encoder.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_config = payload["model_config"]
    encoder_keys = ("image_size", "patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio")
    config = {key: model_config[key] for key in encoder_keys if key in model_config}
    encoder = ImageEncoder(**config)
    state = payload["model"]
    prefix = "target_encoder."
    encoder.load_state_dict({key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)})
    return encoder.to(device).eval().requires_grad_(requires_grad=False)
