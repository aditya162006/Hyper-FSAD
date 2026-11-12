import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .backbone import DinoV3Encoder, DEFAULT_MODELS, load_encoder
from .core import Features, HyperFSAD


def image_paths(directory):
    paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise ValueError(f"No images found in {directory}")
    return paths


def read_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Hyper-FSAD v2 inference for one category")
    parser.add_argument("--support", required=True, help="Directory containing only normal images")
    parser.add_argument("--query", required=True, help="Query image directory")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--backbone", choices=("dinov2", "dinov3"), default="dinov3")
    parser.add_argument("--model", default=None, help="Hugging Face model ID or local checkpoint directory")
    parser.add_argument("--shots", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    supports, queries = image_paths(args.support), image_paths(args.query)
    if len(supports) < args.shots:
        parser.error("Not enough normal support images")
    if set(p.resolve() for p in supports) & set(p.resolve() for p in queries):
        parser.error("Support and query files overlap; use separate directories")
    indices = np.random.default_rng(args.seed).choice(len(supports), args.shots, replace=False)
    chosen = [supports[i] for i in indices]
    args.model = args.model or DEFAULT_MODELS[args.backbone]
    encoder = load_encoder(args.backbone, args.model, device=args.device)
    features = [encoder([read_image(p)]) for p in chosen]
    support = Features(torch.cat([f.patches for f in features], dim=1), torch.cat([f.cls for f in features], dim=1), features[0].grid)
    detector = HyperFSAD(chunk_size=args.chunk_size).fit(support)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "support.txt").write_text("\n".join(str(p.resolve()) for p in chosen) + "\n")
    with (output / "scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "score", "spatial_score", "global_score", "map"])
        for start in range(0, len(queries), args.batch_size):
            paths = queries[start:start + args.batch_size]
            images = [read_image(p) for p in paths]
            prediction = detector.predict(encoder(images))
            for i, (path, image) in enumerate(zip(paths, images)):
                relative = path.relative_to(Path(args.query))
                destination = output / "maps" / relative.parent / (relative.name + ".npy")
                destination.parent.mkdir(parents=True, exist_ok=True)
                resized = torch.nn.functional.interpolate(prediction.patch_map[i:i+1, None], size=(image.height, image.width), mode="bilinear", align_corners=False)[0, 0]
                np.save(destination, resized.cpu().numpy())
                writer.writerow([str(path), float(prediction.image_score[i]), float(prediction.spatial_score[i]), float(prediction.global_score[i]), str(destination)])
    print(f"Saved scores and anomaly maps to {output.resolve()}")


if __name__ == "__main__":
    main()
