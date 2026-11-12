"""Evaluate Hyper-FSAD on MVTec AD."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
import torch

from .backbone import DinoV3Encoder, DEFAULT_MODELS, load_encoder
from .cli import read_image
from .core import Features, HyperFSAD


@dataclass(frozen=True)
class Sample:
    # Keep an image path, its optional answer-mask path, and its image-level label
    # together. anomalous=True means the dataset says the image contains a defect.
    image: Path
    mask: Path | None
    anomalous: bool


def load_category(root: Path, category: str):
    if Path(category).name != category or category in {".", ".."}:
        raise ValueError("Category must be a directory name")
    category_dir = root / category
    # Only normal training images can be references. Test images stay separate.
    support_paths = sorted((category_dir / "train" / "good").glob("*.png"))
    test_samples = []
    for image_path in sorted((category_dir / "test").glob("*/*.png")):
        defect_type = image_path.parent.name
        # Normal images have no mask file. Defect masks use the original name plus _mask.
        mask_path = (
            None
            if defect_type == "good"
            else category_dir
            / "ground_truth"
            / defect_type
            / (image_path.stem + "_mask.png")
        )
        if mask_path is not None and not mask_path.is_file():
            raise ValueError(f"Missing ground-truth mask: {mask_path}")
        test_samples.append(Sample(image_path, mask_path, defect_type != "good"))
    if not support_paths or not test_samples:
        raise ValueError(f"Missing training or test PNG images in {category_dir}")
    if len({sample.anomalous for sample in test_samples}) != 2:
        raise ValueError(
            "AUROC evaluation requires both normal and anomalous test images"
        )
    if {support_path.resolve() for support_path in support_paths} & {
        sample.image.resolve() for sample in test_samples
    }:
        raise ValueError("Support and test images overlap")
    return support_paths, test_samples


def binary_metrics(labels, scores):
    # labels contain the known answers (0 normal, 1 anomalous); scores are predictions.
    # reshape(-1) turns either image lists or pixel arrays into a single long list.
    labels, scores = np.asarray(labels).reshape(-1), np.asarray(scores).reshape(-1)
    if (
        labels.shape != scores.shape
        or labels.size == 0
        or not np.isfinite(scores).all()
    ):
        raise ValueError(
            "Labels and finite scores must be nonempty and have equal size"
        )
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Metrics require binary labels with both classes present")
    # Check all score thresholds to report the best test-set F1. This does not train the model.
    precision, recall, _ = precision_recall_curve(labels, scores)
    precision_plus_recall = precision + recall
    # Avoid dividing by zero when both precision and recall are zero.
    f1_scores = np.divide(
        2 * precision * recall,
        precision_plus_recall,
        out=np.zeros_like(precision),
        where=precision_plus_recall > 0,
    )
    # AUROC checks ranking, AP summarizes precision versus recall, and maximum F1
    # is the best precision/recall balance found on this test set. None is ordinary accuracy.
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "f1_max": float(f1_scores.max()),
    }


def evaluate_category(
    root, category, encoder, shots=1, seed=0, batch_size=8, chunk_size=256
):
    if shots < 1 or batch_size < 1:
        raise ValueError("shots and batch_size must be positive")
    # This function handles one category and one random reference selection.
    # An evaluation run has labels; ordinary inference in cli.py does not need them.
    root = Path(root)
    support_paths, test_samples = load_category(root, category)
    if len(support_paths) < shots:
        raise ValueError("Insufficient normal support images")
    # A fixed seed lets another run select the same normal references.
    selected_support_paths = [
        support_paths[image_index]
        for image_index in np.random.default_rng(seed).choice(
            len(support_paths), shots, replace=False
        )
    ]
    support_features = [
        encoder([read_image(support_path)]) for support_path in selected_support_paths
    ]
    # Combine the reference images into the memory bank used for this category.
    support_bank = Features(
        torch.cat([image_features.patches for image_features in support_features], 1),
        torch.cat([image_features.cls for image_features in support_features], 1),
        support_features[0].grid,
    )
    detector = HyperFSAD(chunk_size=chunk_size).fit(support_bank)
    image_records, pixel_labels, pixel_scores = [], [], []
    for batch_start in range(0, len(test_samples), batch_size):
        sample_batch = test_samples[batch_start : batch_start + batch_size]
        images = [read_image(sample.image) for sample in sample_batch]
        # The mask is not passed to the encoder or detector. It must not influence scores.
        prediction = detector.predict(encoder(images))
        for image_index, (sample, image) in enumerate(zip(sample_batch, images)):
            # Every pixel in a normal image gets label 0; defect masks mark anomalous pixels.
            if sample.mask is None:
                ground_truth_mask = np.zeros((image.height, image.width), dtype=bool)
            else:
                with Image.open(sample.mask) as mask_image:
                    # Nonzero mask pixels mark defects. Convert them to True/False labels.
                    ground_truth_mask = np.asarray(mask_image.convert("L")) > 0
                if ground_truth_mask.shape != (image.height, image.width):
                    raise ValueError(
                        f"Mask dimensions differ from image: {sample.mask}"
                    )
            # Compare the predicted map and mask at the original image resolution.
            resized_anomaly_map = (
                torch.nn.functional.interpolate(
                    prediction.patch_map[image_index : image_index + 1, None],
                    size=ground_truth_mask.shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                .cpu()
                .numpy()
            )
            # Flatten each map so the metrics can evaluate all test pixels together.
            pixel_labels.append(ground_truth_mask.reshape(-1))
            pixel_scores.append(resized_anomaly_map.reshape(-1))
            image_records.append(
                {
                    "image": str(sample.image.relative_to(root)),
                    "label": int(sample.anomalous),
                    "score": float(prediction.image_score[image_index]),
                    "spatial_score": float(prediction.spatial_score[image_index]),
                    "global_score": float(prediction.global_score[image_index]),
                }
            )
        print(
            f"{category} seed={seed}: {min(batch_start + batch_size, len(test_samples))}/{len(test_samples)}",
            flush=True,
        )
    return {
        "category": category,
        "shots": shots,
        "seed": seed,
        "support_images": [
            str(support_path.relative_to(root))
            for support_path in selected_support_paths
        ],
        "test_image_count": len(image_records),
        "image_metrics": binary_metrics(
            [record["label"] for record in image_records],
            [record["score"] for record in image_records],
        ),
        "pixel_metrics": binary_metrics(
            np.concatenate(pixel_labels), np.concatenate(pixel_scores)
        ),
        "predictions": image_records,
    }


def summarize(runs):
    """Combine repeated experiments without favoring larger categories.

    Each seed can select a different normal reference. The mean summarizes those
    runs and the standard deviation shows how much they differ. We then average
    category means, rather than pooling all images into one overall score.
    """
    # First measure how the results vary with the normal reference selection.
    category_results = {}
    for category in sorted({run["category"] for run in runs}):
        category_runs = [run for run in runs if run["category"] == category]
        category_results[category] = {}
        for metric_group in ("image_metrics", "pixel_metrics"):
            category_results[category][metric_group] = {}
            for metric in category_runs[0][metric_group]:
                seed_scores = [run[metric_group][metric] for run in category_runs]
                category_results[category][metric_group][metric] = {
                    "mean": float(np.mean(seed_scores)),
                    "std": float(np.std(seed_scores)),
                }
    # Give every category one vote, even when their test sets have different sizes.
    macro_metrics = {
        metric_group: {
            metric: float(
                np.mean(
                    [
                        category_summary[metric_group][metric]["mean"]
                        for category_summary in category_results.values()
                    ]
                )
            )
            for metric in ("auroc", "average_precision", "f1_max")
        }
        for metric_group in ("image_metrics", "pixel_metrics")
    }
    return {
        "per_category": category_results,
        "macro_mean": macro_metrics,
        "std_convention": "population (ddof=0)",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="MVTec root containing category directories",
    )
    parser.add_argument("--categories", nargs="+", default=["bottle"])
    parser.add_argument("--shots", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--backbone", choices=("dinov2", "dinov3"), default="dinov3")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=Path("results/mvtec-1shot"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.chunk_size < 1 or min(args.seeds) < 0:
        parser.error("Batch/chunk sizes must be positive and seeds nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.categories)) != len(
        args.categories
    ):
        parser.error("Seeds and categories must be unique")
    # Check the dataset layout before spending time loading the model.
    for category in args.categories:
        support_paths, _ = load_category(args.data, category)
        if len(support_paths) < args.shots:
            parser.error(f"Insufficient supports for {category}")
    args.model = args.model or DEFAULT_MODELS[args.backbone]
    encoder = load_encoder(args.backbone, args.model, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    # Run each category with each requested seed and keep the individual results.
    evaluation_runs = []
    for category in args.categories:
        for seed in args.seeds:
            run_result = evaluate_category(
                args.data,
                category,
                encoder,
                args.shots,
                seed,
                args.batch_size,
                args.chunk_size,
            )
            # Save each finished run immediately, before starting the next seed.
            run_path = args.output / f"{category}-{args.shots}shot-seed{seed}.json"
            run_path.write_text(
                json.dumps(run_result, indent=2, allow_nan=False) + "\n"
            )
            evaluation_runs.append(run_result)
    # Save the model revision and settings alongside the numbers for reproducibility.
    report = {
        "paper": "arXiv:2605.10628v2",
        "model": args.model,
        "model_revision": getattr(encoder.model.config, "_commit_hash", None),
        "config": {
            option_name: (
                str(option_value) if isinstance(option_value, Path) else option_value
            )
            for option_name, option_value in vars(args).items()
        },
        "protocol": {
            "backbone": args.backbone,
            "input_size": encoder.image_size,
            "layers": list(encoder.layers),
            "fusion_weight": 0.5,
            "pixel_resolution": "native",
            "map_interpolation": "bilinear, align_corners=False",
            "f1_max": "oracle test-set metric, not a deployment threshold",
            "pro": "not implemented",
        },
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        **summarize(evaluation_runs),
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["macro_mean"], indent=2))


if __name__ == "__main__":
    main()
