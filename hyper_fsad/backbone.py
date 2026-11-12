"""Feature extraction with frozen DINOv2 and DINOv3 models."""

from collections.abc import Sequence

import torch
from PIL import Image

from .core import Features


class DinoV3Encoder:
    model_type = "dinov3_vit"

    def __init__(
        self, model, processor, layers=(3, 6, 9, 12), image_size=448, device="cpu"
    ):
        if getattr(model.config, "model_type", None) != self.model_type:
            raise ValueError(f"This adapter requires model_type={self.model_type}")
        if (
            not layers
            or len(set(layers)) != len(layers)
            or any(
                layer_index < 1 or layer_index > model.config.num_hidden_layers
                for layer_index in layers
            )
        ):
            raise ValueError(
                "Layers must be unique one-based transformer block indices"
            )
        self.patch_size = model.config.patch_size
        if image_size < self.patch_size or image_size % self.patch_size:
            raise ValueError("image_size must be a positive multiple of patch_size")
        # Keep the pretrained weights fixed and turn off training behavior such as dropout.
        self.model = model.to(device).eval().requires_grad_(False)
        self.processor = processor
        self.layers = tuple(layers)
        self.image_size = image_size
        self.device = device

    @classmethod
    def from_pretrained(
        cls, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", **kwargs
    ):
        from transformers import AutoImageProcessor, AutoModel

        # Load both the model weights and the matching image preprocessing settings.
        return cls(
            AutoModel.from_pretrained(model_name, trust_remote_code=False),
            AutoImageProcessor.from_pretrained(model_name, trust_remote_code=False),
            **kwargs,
        )

    @torch.inference_mode()
    def __call__(self, images: Sequence[Image.Image]) -> Features:
        if not images:
            raise ValueError("At least one image is required")
        # Keep the whole image so defects near the edges are not cropped out.
        pixel_values = self.processor(
            images=[image.convert("RGB") for image in images],
            do_resize=True,
            size={"height": self.image_size, "width": self.image_size},
            do_center_crop=False,
            return_tensors="pt",
        )["pixel_values"].to(self.device)
        # Ask for intermediate outputs because we need features from several layers.
        layer_outputs = self.model(
            pixel_values=pixel_values, output_hidden_states=True
        ).hidden_states
        # For example, 448 pixels with 16-pixel patches gives a 28 by 28 patch grid.
        grid_side = self.image_size // self.patch_size
        num_patches = grid_side * grid_side
        # CLS is first. DINOv3 also has register tokens, which are not image patches.
        num_special_tokens = 1 + getattr(self.model.config, "num_register_tokens", 0)
        # Output 0 is the embedding stage; indices 3, 6, 9 and 12 are transformer blocks.
        selected_layer_outputs = [layer_outputs[layer] for layer in self.layers]
        if any(
            layer_tokens.shape[1] != num_special_tokens + num_patches
            for layer_tokens in selected_layer_outputs
        ):
            raise RuntimeError("Unexpected visual transformer token layout")
        # Keep patch tokens and CLS tokens separately, with layers as the first dimension.
        return Features(
            torch.stack(
                [
                    layer_tokens[:, num_special_tokens:]
                    for layer_tokens in selected_layer_outputs
                ]
            ),
            torch.stack(
                [layer_tokens[:, 0] for layer_tokens in selected_layer_outputs]
            ),
            (grid_side, grid_side),
        )


class DinoV2Encoder(DinoV3Encoder):
    """DINOv2 ViT-B/14 without register tokens, at 224x224 by default."""

    model_type = "dinov2"

    def __init__(
        self, model, processor, layers=(3, 6, 9, 12), image_size=224, device="cpu"
    ):
        super().__init__(model, processor, layers, image_size, device)

    @classmethod
    def from_pretrained(cls, model_name="facebook/dinov2-base", **kwargs):
        return super().from_pretrained(model_name, **kwargs)


DEFAULT_MODELS = {
    "dinov2": "facebook/dinov2-base",
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}


def load_encoder(backbone="dinov3", model=None, **kwargs):
    # Pick the adapter and default checkpoint for the requested DINO version.
    encoder_type = {"dinov2": DinoV2Encoder, "dinov3": DinoV3Encoder}[backbone]
    return encoder_type.from_pretrained(model or DEFAULT_MODELS[backbone], **kwargs)
