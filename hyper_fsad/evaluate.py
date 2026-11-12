"""Evaluate Hyper-FSAD on MVTec AD."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch

from .backbone import DinoV3Encoder, DEFAULT_MODELS, load_encoder
from .cli import read_image
from .core import Features, HyperFSAD


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path | None
    anomalous: bool


def load_category(root: Path, category: str):
    if Path(category).name != category or category in {".", ".."}:
        raise ValueError("Category must be a directory name")
    folder = root / category
    supports = sorted((folder / "train" / "good").glob("*.png"))
    samples = []
    for image in sorted((folder / "test").glob("*/*.png")):
        defect = image.parent.name
        mask = None if defect == "good" else folder / "ground_truth" / defect / (image.stem + "_mask.png")
        if mask is not None and not mask.is_file():
            raise ValueError(f"Missing ground-truth mask: {mask}")
        samples.append(Sample(image, mask, defect != "good"))
    if not supports or not samples:
        raise ValueError(f"Missing training or test PNG images in {folder}")
    if len({s.anomalous for s in samples}) != 2:
        raise ValueError("AUROC evaluation requires both normal and anomalous test images")
    if {p.resolve() for p in supports} & {s.image.resolve() for s in samples}:
        raise ValueError("Support and test images overlap")
    return supports, samples


def binary_metrics(labels, scores):
    labels, scores = np.asarray(labels).reshape(-1), np.asarray(scores).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Labels and finite scores must be nonempty and have equal size")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Metrics require binary labels with both classes present")
    precision, recall, _ = precision_recall_curve(labels, scores)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(precision), where=denominator > 0)
    return {"auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "f1_max": float(f1.max())}


def evaluate_category(root, category, encoder, shots=1, seed=0, batch_size=8, chunk_size=256):
    if shots < 1 or batch_size < 1:
        raise ValueError("shots and batch_size must be positive")
    root = Path(root)
    supports, samples = load_category(root, category)
    if len(supports) < shots:
        raise ValueError("Insufficient normal support images")
    chosen = [supports[i] for i in np.random.default_rng(seed).choice(len(supports), shots, replace=False)]
    support_features = [encoder([read_image(p)]) for p in chosen]
    support = Features(torch.cat([f.patches for f in support_features], 1),
                       torch.cat([f.cls for f in support_features], 1), support_features[0].grid)
    detector = HyperFSAD(chunk_size=chunk_size).fit(support)
    records, pixel_labels, pixel_scores = [], [], []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [read_image(s.image) for s in batch]
        prediction = detector.predict(encoder(images))
        for i, (sample, image) in enumerate(zip(batch, images)):
            if sample.mask is None:
                mask = np.zeros((image.height, image.width), dtype=bool)
            else:
                with Image.open(sample.mask) as source:
                    mask = np.asarray(source.convert("L")) > 0
                if mask.shape != (image.height, image.width):
                    raise ValueError(f"Mask dimensions differ from image: {sample.mask}")
            native_map = torch.nn.functional.interpolate(prediction.patch_map[i:i+1, None],
                size=mask.shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            pixel_labels.append(mask.reshape(-1))
            pixel_scores.append(native_map.reshape(-1))
            records.append({"image": str(sample.image.relative_to(root)), "label": int(sample.anomalous),
                            "score": float(prediction.image_score[i]),
                            "spatial_score": float(prediction.spatial_score[i]),
                            "global_score": float(prediction.global_score[i])})
        print(f"{category} seed={seed}: {min(start + batch_size, len(samples))}/{len(samples)}", flush=True)
    return {"category": category, "shots": shots, "seed": seed,
            "support_images": [str(p.relative_to(root)) for p in chosen],
            "test_image_count": len(records),
            "image_metrics": binary_metrics([r["label"] for r in records], [r["score"] for r in records]),
            "pixel_metrics": binary_metrics(np.concatenate(pixel_labels), np.concatenate(pixel_scores)),
            "predictions": records}


def summarize(runs):
    """Average over seeds first, then give each category equal weight."""
    categories = {}
    for category in sorted({r["category"] for r in runs}):
        selected = [r for r in runs if r["category"] == category]
        categories[category] = {}
        for branch in ("image_metrics", "pixel_metrics"):
            categories[category][branch] = {}
            for metric in selected[0][branch]:
                values = [r[branch][metric] for r in selected]
                categories[category][branch][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    macro = {branch: {metric: float(np.mean([c[branch][metric]["mean"] for c in categories.values()]))
                      for metric in ("auroc", "average_precision", "f1_max")}
             for branch in ("image_metrics", "pixel_metrics")}
    return {"per_category": categories, "macro_mean": macro, "std_convention": "population (ddof=0)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="MVTec root containing category directories")
    parser.add_argument("--categories", nargs="+", default=["bottle"])
    parser.add_argument("--shots", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--backbone", choices=("dinov2", "dinov3"), default="dinov3")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/mvtec-1shot"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.chunk_size < 1 or min(args.seeds) < 0:
        parser.error("Batch/chunk sizes must be positive and seeds nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.categories)) != len(args.categories):
        parser.error("Seeds and categories must be unique")
    for category in args.categories:
        supports, _ = load_category(args.data, category)
        if len(supports) < args.shots:
            parser.error(f"Insufficient supports for {category}")
    args.model = args.model or DEFAULT_MODELS[args.backbone]
    encoder = load_encoder(args.backbone, args.model, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for category in args.categories:
        for seed in args.seeds:
            result = evaluate_category(args.data, category, encoder, args.shots, seed, args.batch_size, args.chunk_size)
            path = args.output / f"{category}-{args.shots}shot-seed{seed}.json"
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            runs.append(result)
    report = {"paper": "arXiv:2605.10628v2", "model": args.model,
              "model_revision": getattr(encoder.model.config, "_commit_hash", None),
              "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "protocol": {"backbone": args.backbone, "input_size": encoder.image_size, "layers": list(encoder.layers), "fusion_weight": 0.5,
                           "pixel_resolution": "native", "map_interpolation": "bilinear, align_corners=False",
                           "f1_max": "oracle test-set metric, not a deployment threshold", "pro": "not implemented"},
              "versions": {"torch": torch.__version__, "numpy": np.__version__},
              **summarize(runs)}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["macro_mean"], indent=2))


if __name__ == "__main__":
    main()
