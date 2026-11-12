"""Feature extraction with frozen DINOv2 and DINOv3 models."""

from collections.abc import Sequence

import torch
from PIL import Image

from .core import Features


class DinoV3Encoder:
    model_type = "dinov3_vit"

    def __init__(self, model, processor, layers=(3, 6, 9, 12), image_size=448, device="cpu"):
        if getattr(model.config, "model_type", None) != self.model_type:
            raise ValueError(f"This adapter requires model_type={self.model_type}")
        if not layers or len(set(layers)) != len(layers) or any(l < 1 or l > model.config.num_hidden_layers for l in layers):
            raise ValueError("Layers must be unique one-based transformer block indices")
        self.patch_size = model.config.patch_size
        if image_size < self.patch_size or image_size % self.patch_size:
            raise ValueError("image_size must be a positive multiple of patch_size")
        self.model = model.to(device).eval().requires_grad_(False)
        self.processor = processor
        self.layers = tuple(layers)
        self.image_size = image_size
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", **kwargs):
        from transformers import AutoImageProcessor, AutoModel

        return cls(AutoModel.from_pretrained(model_name, trust_remote_code=False),
                   AutoImageProcessor.from_pretrained(model_name, trust_remote_code=False), **kwargs)

    @torch.inference_mode()
    def __call__(self, images: Sequence[Image.Image]) -> Features:
        if not images:
            raise ValueError("At least one image is required")
        # Keep the whole image so defects near the edges are not cropped out.
        pixels = self.processor(images=[image.convert("RGB") for image in images],
                                do_resize=True, size={"height": self.image_size, "width": self.image_size},
                                do_center_crop=False, return_tensors="pt")["pixel_values"].to(self.device)
        states = self.model(pixel_values=pixels, output_hidden_states=True).hidden_states
        side = self.image_size // self.patch_size
        count = side * side
        prefix = 1 + getattr(self.model.config, "num_register_tokens", 0)
        selected = [states[layer] for layer in self.layers]
        if any(state.shape[1] != prefix + count for state in selected):
            raise RuntimeError("Unexpected visual transformer token layout")
        return Features(torch.stack([state[:, prefix:] for state in selected]),
                        torch.stack([state[:, 0] for state in selected]), (side, side))


class DinoV2Encoder(DinoV3Encoder):
    """DINOv2 ViT-B/14 without register tokens, at 224x224 by default."""
    model_type = "dinov2"

    def __init__(self, model, processor, layers=(3, 6, 9, 12), image_size=224, device="cpu"):
        super().__init__(model, processor, layers, image_size, device)

    @classmethod
    def from_pretrained(cls, model_name="facebook/dinov2-base", **kwargs):
        return super().from_pretrained(model_name, **kwargs)


DEFAULT_MODELS = {"dinov2": "facebook/dinov2-base", "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m"}


def load_encoder(backbone="dinov3", model=None, **kwargs):
    encoder_type = {"dinov2": DinoV2Encoder, "dinov3": DinoV3Encoder}[backbone]
    return encoder_type.from_pretrained(model or DEFAULT_MODELS[backbone], **kwargs)
