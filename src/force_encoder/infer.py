"""Export embeddings for local RGB tactile images without training dependencies."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from force_encoder.data import preprocess_image
from force_encoder.model import load_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Output .npy file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens", action="store_true", help="Return patch tokens instead of pooled embeddings")
    args = parser.parse_args()
    encoder = load_encoder(args.checkpoint, device=args.device)
    results = []
    with torch.inference_mode():
        for path in args.images:
            with Image.open(path) as image:
                tensor = preprocess_image(image.convert("RGB"), encoder.image_size).unsqueeze(0).to(args.device)
            output = encoder.forward_features(tensor) if args.tokens else encoder(tensor)
            results.append(output.float().cpu().numpy())
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, np.concatenate(results))
    print(f"Saved {len(results)} embeddings to {destination}")


if __name__ == "__main__":
    main()
