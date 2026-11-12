"""Hyper-FSAD scoring, following equations 4, 5, 8 and 9 of the paper."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


def sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    """Sparsemax from Appendix A.5 of the paper."""
    if not logits.is_floating_point() or logits.numel() == 0:
        raise ValueError("sparsemax requires nonempty floating-point logits")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    # Use float32 for a more stable threshold with half-precision inputs.
    z = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    z = z - z.amax(dim=dim, keepdim=True)
    ordered = z.sort(dim=dim, descending=True).values
    cumulative = ordered.cumsum(dim) - 1
    shape = [1] * z.ndim
    shape[dim] = z.shape[dim]
    ranks = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype).view(shape)
    count = (ranks * ordered > cumulative).sum(dim=dim, keepdim=True)
    threshold = cumulative.gather(dim, count - 1) / count
    return (z - threshold).clamp_min(0)


@dataclass(frozen=True)
class Features:
    patches: Tensor  # [layers, batch, patches, channels]
    cls: Tensor  # [layers, batch, channels]
    grid: tuple[int, int]

    def normalized(self) -> "Features":
        p, c = self.patches, self.cls
        if p.ndim != 4 or c.ndim != 3 or c.shape != p.shape[:2] + p.shape[3:]:
            raise ValueError("Expected patches [L,B,N,D] and CLS [L,B,D]")
        if any(n == 0 for n in p.shape):
            raise ValueError("Feature dimensions must be nonempty")
        if len(self.grid) != 2 or min(self.grid) < 1 or math.prod(self.grid) != p.shape[2]:
            raise ValueError("Grid must match patch count")
        if p.device != c.device or p.dtype != c.dtype or not p.is_floating_point():
            raise ValueError("Patch and CLS features must share floating dtype and device")
        if not torch.isfinite(p).all() or not torch.isfinite(c).all():
            raise ValueError("Features must be finite")
        if (p.norm(dim=-1) == 0).any() or (c.norm(dim=-1) == 0).any():
            raise ValueError("Input tokens must have nonzero norms")
        dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
        return Features(F.normalize(p.to(dtype), dim=-1), F.normalize(c.to(dtype), dim=-1), self.grid)


@dataclass(frozen=True)
class Prediction:
    patch_map: Tensor  # [batch, grid height, grid width]
    image_score: Tensor  # [batch]
    spatial_score: Tensor
    global_score: Tensor

    def resize_maps(self, size: tuple[int, int]) -> Tensor:
        """Resize the maps. Image scores still come from the patch grid."""
        return F.interpolate(self.patch_map[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]


class HyperFSAD:
    """Store normal features and score new images without training."""

    def __init__(self, fusion_weight: float = 0.5, chunk_size: int = 256):
        if not 0 <= fusion_weight <= 1:
            raise ValueError("fusion_weight must lie in [0,1]")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.fusion_weight = fusion_weight
        self.chunk_size = chunk_size
        self._memory: Features | None = None

    @torch.inference_mode()
    def fit(self, support: Features) -> "HyperFSAD":
        s = support.normalized()
        self._memory = Features(s.patches.detach().clone(), s.cls.detach().clone(), s.grid)
        return self

    @torch.inference_mode()
    def predict(self, query: Features) -> Prediction:
        if self._memory is None:
            raise RuntimeError("Call fit with normal support features first")
        q, s = query.normalized(), self._memory
        if (q.patches.shape[0], q.patches.shape[-1]) != (s.patches.shape[0], s.patches.shape[-1]):
            raise ValueError("Query and support must share layer count and feature dimension")
        if q.patches.device != s.patches.device or q.patches.dtype != s.patches.dtype:
            raise ValueError("Query and support must share device and precision")
        layers, batch, patches, channels = q.patches.shape
        local = torch.zeros((batch, patches), device=q.patches.device, dtype=q.patches.dtype)
        global_score = torch.zeros(batch, device=local.device, dtype=local.dtype)
        for layer in range(layers):
            memory = s.patches[layer].reshape(-1, channels)
            flat_query = q.patches[layer].reshape(-1, channels)
            distances = []
            # Split queries to save memory, but match each one against all support patches.
            for chunk in flat_query.split(self.chunk_size):
                weights = sparsemax(chunk @ memory.T)
                reconstruction = weights @ memory
                # PyTorch handles a zero reconstruction with epsilon, giving distance 1.
                distances.append((1 - F.cosine_similarity(chunk, reconstruction, dim=-1)).clamp(0, 2))
            local += torch.cat(distances).reshape(batch, patches) / layers
            nearest = (q.cls[layer] @ s.cls[layer].T).amax(dim=-1)
            global_score += (1 - nearest).clamp(0, 2) / layers
        patch_map = local.reshape(batch, *q.grid)
        spatial = local.amax(dim=-1)
        image = self.fusion_weight * spatial + (1 - self.fusion_weight) * global_score
        return Prediction(patch_map, image, spatial, global_score)
