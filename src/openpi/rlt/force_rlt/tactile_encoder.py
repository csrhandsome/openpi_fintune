"""Frozen, weight-sharing ResNet-18 for the two in-finger RGB cameras."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

PREPROCESS_VERSION = "finger-rgb-roi-imagenet-v1"


class SharedFingerEncoder(nn.Module):
    """Encode each finger independently with one frozen torchvision ResNet-18.

    The caller must provide locally stored ImageNet-1K ResNet-18 state_dict
    weights. No implicit download or random encoder is allowed in the experiment path.
    """

    feature_dim = 512

    def __init__(self, weights_path: str | Path, *, roi: tuple[int, int, int, int], device: str = "cpu"):
        super().__init__()
        from torchvision.models import resnet18

        self.weights_path = Path(weights_path)
        self.roi = roi  # x0, y0, x1, y1 in raw RGB pixels
        network = resnet18(weights=None)
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        network.load_state_dict(state, strict=True)
        network.fc = nn.Identity()
        self.network = network.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.to(device)
        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = self.encode_pairs(left_rgb[None], right_rgb[None])
        return features[0, 0], features[0, 1]

    @torch.no_grad()
    def encode_pairs(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        """Encode an entire history in one GPU batch, returning [T,2,512]."""
        from torch.nn import functional

        if left_rgb.shape != right_rgb.shape or left_rgb.ndim != 4 or left_rgb.shape[-1] != 3:
            raise ValueError("Both finger histories must be matching THWC RGB images")
        if left_rgb.dtype != np.uint8 or right_rgb.dtype != np.uint8:
            raise ValueError("Finger images must be uint8 RGB in [0,255]")
        x0, y0, x1, y1 = self.roi
        if not (0 <= x0 < x1 <= left_rgb.shape[2] and 0 <= y0 < y1 <= left_rgb.shape[1]):
            raise ValueError("ROI must be within both finger images")
        pair = np.stack([left_rgb[:, y0:y1, x0:x1], right_rgb[:, y0:y1, x0:x1]], axis=1)
        batch = torch.from_numpy(pair.copy()).flatten(0, 1).permute(0, 3, 1, 2)
        batch = batch.float().to(self.mean.device) / 255.0
        batch = functional.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
        embedding = self.network((batch - self.mean) / self.std).float().cpu().numpy()
        return embedding.reshape(len(left_rgb), 2, self.feature_dim)
