import torch
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast

from hyper_fsad.backbone import DinoV3Encoder
from hyper_fsad import HyperFSAD


def test_dinov2_layers_and_patch_grid():
    from transformers import Dinov2Config, Dinov2Model, BitImageProcessor
    from hyper_fsad.backbone import DinoV2Encoder

    config = Dinov2Config(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                         image_size=28, patch_size=14)
    model = Dinov2Model(config)
    encoder = DinoV2Encoder(model, BitImageProcessor(), layers=(1, 2, 3, 4), image_size=28)
    features = encoder([Image.new("RGB", (30, 45), "red")])
    assert features.patches.shape == (4, 1, 4, 32)
    assert features.cls.shape == (4, 1, 32)
    assert features.grid == (2, 2)
    assert not features.patches.requires_grad
    assert all(not p.requires_grad for p in model.parameters())


def test_real_dinov3_architecture_without_checkpoint_download():
    config = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                           num_attention_heads=4, patch_size=16, num_register_tokens=2)
    model = DINOv3ViTModel(config)
    encoder = DinoV3Encoder(model, DINOv3ViTImageProcessorFast(), layers=(1, 2, 3, 4), image_size=32)
    images = [Image.new("RGB", (45, 30), "red"), Image.new("RGB", (20, 60), "blue")]
    features = encoder(images)
    assert features.patches.shape == (4, 2, 4, 32)
    assert features.cls.shape == (4, 2, 32)
    assert not features.patches.requires_grad
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    result = HyperFSAD().fit(features).predict(features)
    assert torch.isfinite(result.image_score).all()
