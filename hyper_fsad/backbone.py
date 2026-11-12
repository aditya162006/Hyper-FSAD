"""Feature extraction with frozen DINOv2 and DINOv3 models."""

from collections.abc import Sequence

import torch
from PIL import Image

from .core import Features


class DinoV3Encoder:
    """Convert pictures into features using a pretrained DINOv3 network.

    The model supplies visual descriptions, not anomaly scores. core.py compares
    those descriptions with normal references. We read several internal layers
    because each stage can represent different visual information.
    """

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
        # A patch is a small square of the input image. Require an exact number
        # of patches across each side so the output can form a complete grid.
        self.patch_size = model.config.patch_size
        if image_size < self.patch_size or image_size % self.patch_size:
            raise ValueError("image_size must be a positive multiple of patch_size")
        # .to(device) chooses CPU or GPU. .eval() turns off training behavior.
        # requires_grad_(False) freezes the weights; we do not learn new ones here.
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

    # inference_mode avoids storing gradient information that is only needed for training.
    @torch.inference_mode()
    def __call__(self, images: Sequence[Image.Image]) -> Features:
        if not images:
            raise ValueError("At least one image is required")
        # RGB gives three color channels. The processor resizes images and applies
        # the pixel scaling/normalization expected by this pretrained model.
        # return_tensors="pt" asks for PyTorch arrays. Keep the whole image instead
        # of cropping, since a defect might be near an edge.
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
        # A token is a list of feature values. The first one (CLS) summarizes the image.
        # Register tokens are extra internal model tokens with no image-grid location,
        # so skip them before collecting the actual patch tokens.
        num_special_tokens = 1 + getattr(self.model.config, "num_register_tokens", 0)
        # Output 0 is the embedding stage; indices 3, 6, 9 and 12 are transformer blocks.
        selected_layer_outputs = [layer_outputs[layer] for layer in self.layers]
        if any(
            layer_tokens.shape[1] != num_special_tokens + num_patches
            for layer_tokens in selected_layer_outputs
        ):
            raise RuntimeError("Unexpected visual transformer token layout")
        # [:, num_special_tokens:] keeps all images but skips CLS/register positions.
        # [:, 0] takes just CLS. stack adds a new axis for the selected layers,
        # giving patch shape [layers, images, patches, feature values].
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

    # DINOv2 uses the same extraction steps. This subclass changes the model type,
    # default input size and checkpoint instead of repeating all of that code.
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
