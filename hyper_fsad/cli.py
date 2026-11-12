import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .backbone import DinoV3Encoder, DEFAULT_MODELS, load_encoder
from .core import Features, HyperFSAD


def image_paths(directory):
    # Search subfolders too, and sort paths so a seed gives a repeatable selection.
    matching_paths = sorted(
        image_path
        for image_path in Path(directory).rglob("*")
        if image_path.suffix.lower()
        in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not matching_paths:
        raise ValueError(f"No images found in {directory}")
    return matching_paths


def read_image(path):
    # Convert grayscale or RGBA inputs to the three channels expected by DINO.
    with Image.open(path) as image:
        return image.convert("RGB")


def main():
    # These options let us run inference without editing the Python code.
    parser = argparse.ArgumentParser(
        description="Hyper-FSAD v2 inference for one category"
    )
    parser.add_argument(
        "--support", required=True, help="Directory containing only normal images"
    )
    parser.add_argument("--query", required=True, help="Query image directory")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--backbone", choices=("dinov2", "dinov3"), default="dinov3")
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face model ID or local checkpoint directory",
    )
    parser.add_argument("--shots", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    support_paths, query_paths = image_paths(args.support), image_paths(args.query)
    if len(support_paths) < args.shots:
        parser.error("Not enough normal support images")
    # A query must not also be one of the normal reference images.
    if set(image_path.resolve() for image_path in support_paths) & set(
        image_path.resolve() for image_path in query_paths
    ):
        parser.error("Support and query files overlap; use separate directories")
    # Choose the normal references without picking the same image twice.
    selected_indices = np.random.default_rng(args.seed).choice(
        len(support_paths), args.shots, replace=False
    )
    selected_support_paths = [
        support_paths[image_index] for image_index in selected_indices
    ]
    args.model = args.model or DEFAULT_MODELS[args.backbone]
    encoder = load_encoder(args.backbone, args.model, device=args.device)
    support_features = [
        encoder([read_image(image_path)]) for image_path in selected_support_paths
    ]
    # Join the reference features along the image dimension, keeping layers separate.
    support_bank = Features(
        torch.cat(
            [image_features.patches for image_features in support_features], dim=1
        ),
        torch.cat([image_features.cls for image_features in support_features], dim=1),
        support_features[0].grid,
    )
    detector = HyperFSAD(chunk_size=args.chunk_size).fit(support_bank)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Record the chosen references so we can repeat this run later.
    (output_dir / "support.txt").write_text(
        "\n".join(str(image_path.resolve()) for image_path in selected_support_paths)
        + "\n"
    )
    with (output_dir / "scores.csv").open("w", newline="") as scores_file:
        csv_writer = csv.writer(scores_file)
        csv_writer.writerow(["image", "score", "spatial_score", "global_score", "map"])
        # Process a few query images at a time to keep memory use manageable.
        for batch_start in range(0, len(query_paths), args.batch_size):
            batch_paths = query_paths[batch_start : batch_start + args.batch_size]
            images = [read_image(image_path) for image_path in batch_paths]
            prediction = detector.predict(encoder(images))
            for image_index, (query_path, image) in enumerate(zip(batch_paths, images)):
                # Keep the input subfolders so files with the same name do not overwrite each other.
                relative_path = query_path.relative_to(Path(args.query))
                map_path = (
                    output_dir
                    / "maps"
                    / relative_path.parent
                    / (relative_path.name + ".npy")
                )
                map_path.parent.mkdir(parents=True, exist_ok=True)
                # Save the map at the original image size, not the smaller patch-grid size.
                resized_map = torch.nn.functional.interpolate(
                    prediction.patch_map[image_index : image_index + 1, None],
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                np.save(map_path, resized_map.cpu().numpy())
                csv_writer.writerow(
                    [
                        str(query_path),
                        float(prediction.image_score[image_index]),
                        float(prediction.spatial_score[image_index]),
                        float(prediction.global_score[image_index]),
                        str(map_path),
                    ]
                )
    print(f"Saved scores and anomaly maps to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
