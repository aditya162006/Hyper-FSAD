import csv
import sys

import numpy as np
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast

from hyper_fsad import cli
from hyper_fsad.backbone import DinoV3Encoder


def test_folder_inference_with_local_test_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    config = DINOv3ViTConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=12,
                           num_attention_heads=4, patch_size=16, num_register_tokens=2)
    DINOv3ViTModel(config).save_pretrained(checkpoint)
    DINOv3ViTImageProcessorFast().save_pretrained(checkpoint)
    for folder, color in [("support", "red"), ("query", "blue")]:
        (tmp_path / folder).mkdir()
        Image.new("RGB", (40, 24), color).save(tmp_path / folder / "image.png")
    original_loader = DinoV3Encoder.from_pretrained
    monkeypatch.setattr(DinoV3Encoder, "from_pretrained", lambda name, **kw: original_loader(name, image_size=32, **kw))
    monkeypatch.setattr(sys, "argv", ["hyper-fsad", "--support", str(tmp_path / "support"),
                                    "--query", str(tmp_path / "query"), "--model", str(checkpoint),
                                    "--output", str(tmp_path / "outputs"), "--device", "cpu"])
    cli.main()
    with (tmp_path / "outputs/scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert 0 <= float(rows[0]["score"]) <= 2
    anomaly_map = np.load(rows[0]["map"])
    assert anomaly_map.shape == (24, 40)
    assert np.isfinite(anomaly_map).all()
    assert (tmp_path / "outputs/support.txt").read_text().strip().endswith("image.png")
