import numpy as np
import pytest
from PIL import Image
import torch

from hyper_fsad import Features
from hyper_fsad.evaluate import binary_metrics, evaluate_category, load_category, summarize


def make_dataset(root):
    for name, color in [("train/good/000.png", "red"), ("train/good/001.png", "red"),
                        ("test/good/000.png", "red"), ("test/crack/000.png", "blue")]:
        path = root / "bottle" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (6, 4), color).save(path)
    path = root / "bottle/ground_truth/crack/000_mask.png"
    path.parent.mkdir(parents=True)
    Image.new("L", (6, 4), 255).save(path)


def color_encoder(images):
    vectors = torch.tensor([[1., 0.] if image.getpixel((0, 0))[0] else [0., 1.] for image in images])
    return Features(vectors[None, :, None], vectors[None], (1, 1))


def test_metrics_perfect_tied_and_reversed():
    perfect = binary_metrics([0, 0, 1, 1], [0., .1, .8, 1.])
    assert all(v == 1 for v in perfect.values())
    tied = binary_metrics([0, 1], [.5, .5])
    assert tied["auroc"] == .5
    assert tied["average_precision"] == .5
    assert tied["f1_max"] == pytest.approx(2 / 3)
    assert binary_metrics([0, 1], [1., 0.])["auroc"] == 0
    with pytest.raises(ValueError):
        binary_metrics([1, 1], [0., 1.])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [np.nan, 1.])


def test_complete_category_evaluation_and_determinism(tmp_path):
    make_dataset(tmp_path)
    run = evaluate_category(tmp_path, "bottle", color_encoder, seed=42, batch_size=2)
    second = evaluate_category(tmp_path, "bottle", color_encoder, seed=42, batch_size=1)
    assert run == second
    assert all(value == 1 for value in run["image_metrics"].values())
    assert all(value == 1 for value in run["pixel_metrics"].values())
    assert run["test_image_count"] == 2
    assert all("train/good/" in p for p in run["support_images"])
    assert all("test/" in p["image"] for p in run["predictions"])
    summary = summarize([run, second])
    assert summary["macro_mean"]["image_metrics"]["auroc"] == 1
    assert summary["per_category"]["bottle"]["image_metrics"]["auroc"]["std"] == 0


def test_missing_and_wrong_size_masks_fail(tmp_path):
    make_dataset(tmp_path)
    path = tmp_path / "bottle/ground_truth/crack/000_mask.png"
    Image.new("L", (2, 2), 255).save(path)
    with pytest.raises(ValueError, match="dimensions"):
        evaluate_category(tmp_path, "bottle", color_encoder)
    path.unlink()
    with pytest.raises(ValueError, match="Missing ground-truth"):
        load_category(tmp_path, "bottle")


def test_summary_macro_averages_categories_not_samples():
    runs = [{"category": c, "image_metrics": {"auroc": x, "average_precision": x, "f1_max": x},
             "pixel_metrics": {"auroc": x, "average_precision": x, "f1_max": x}}
            for c, x in [("a", 0.), ("a", 1.), ("b", 1.)]]
    assert summarize(runs)["macro_mean"]["image_metrics"]["auroc"] == .75
