"""Hyper-FSAD scoring, following equations 4, 5, 8 and 9 of the paper."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


def sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    """Turn similarity scores into weights for the normal reference patches.

    Each row usually contains the matches for one query patch. A large weight
    means a reference patch contributes more to the reconstruction. Some weights
    become exactly zero, so poor matches do not contribute at all. The weights
    in a row add up to one. ``dim`` selects the axis containing the matches.
    """
    if not logits.is_floating_point() or logits.numel() == 0:
        raise ValueError("sparsemax requires nonempty floating-point logits")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    # Use float32 for a more stable threshold with half-precision inputs.
    shifted_scores = (
        logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    )
    # Subtracting the largest score keeps the numbers small without changing the weights.
    shifted_scores = shifted_scores - shifted_scores.amax(dim=dim, keepdim=True)
    # Sort the matches so we can find which ones are above the sparsemax cutoff.
    sorted_scores = shifted_scores.sort(dim=dim, descending=True).values
    # cumsum gives running totals: [a, b, c] becomes [a, a+b, a+b+c].
    # Subtracting 1 is part of the formula that makes the final weights sum to 1.
    cumulative_scores = sorted_scores.cumsum(dim) - 1
    # Make the ranks [1, 2, 3, ...] line up with the matching axis.
    # The other dimensions have size 1 so PyTorch can reuse these ranks for every row.
    rank_shape = [1] * shifted_scores.ndim
    rank_shape[dim] = shifted_scores.shape[dim]
    support_ranks = torch.arange(
        1,
        shifted_scores.shape[dim] + 1,
        device=shifted_scores.device,
        dtype=shifted_scores.dtype,
    ).view(rank_shape)
    # Count the matches that will get a nonzero weight. The rest will be ignored.
    active_count = (support_ranks * sorted_scores > cumulative_scores).sum(
        dim=dim, keepdim=True
    )
    # gather picks the running total at the last active match in each row.
    # Indices start at 0 in Python, so match number k is at position k - 1.
    threshold = cumulative_scores.gather(dim, active_count - 1) / active_count
    # Scores below the cutoff become zero; the remaining weights add up to one.
    return (shifted_scores - threshold).clamp_min(0)


@dataclass(frozen=True)
class Features:
    """Keep the encoder's numeric image descriptions together.

    A tensor is an array with several dimensions. Patch features use dimensions
    [layer, image, patch, feature value]. CLS features leave out the patch
    dimension because there is only one whole-image description per layer.
    ``grid`` remembers the patch rows and columns so we can rebuild a map later.
    """

    # Patch tokens describe small image regions. CLS tokens describe the whole image.
    patches: Tensor  # [layers, batch, patches, channels]
    cls: Tensor  # [layers, batch, channels]
    grid: tuple[int, int]

    def normalized(self) -> "Features":
        # Check the shapes first. Combining incompatible arrays could mix up images
        # or layers, so it is better to stop with an error than return wrong scores.
        patch_tokens, cls_tokens = self.patches, self.cls
        if (
            patch_tokens.ndim != 4
            or cls_tokens.ndim != 3
            or cls_tokens.shape != patch_tokens.shape[:2] + patch_tokens.shape[3:]
        ):
            raise ValueError("Expected patches [L,B,N,D] and CLS [L,B,D]")
        if any(dimension_size == 0 for dimension_size in patch_tokens.shape):
            raise ValueError("Feature dimensions must be nonempty")
        if (
            len(self.grid) != 2
            or min(self.grid) < 1
            or math.prod(self.grid) != patch_tokens.shape[2]
        ):
            raise ValueError("Grid must match patch count")
        # device tells us whether the data is on the CPU or GPU; dtype is the number format.
        # Both token arrays must use the same device and format to work together.
        if (
            patch_tokens.device != cls_tokens.device
            or patch_tokens.dtype != cls_tokens.dtype
            or not patch_tokens.is_floating_point()
        ):
            raise ValueError(
                "Patch and CLS features must share floating dtype and device"
            )
        # Reject NaN/infinity and all-zero tokens: they cannot give useful cosine matches.
        if (
            not torch.isfinite(patch_tokens).all()
            or not torch.isfinite(cls_tokens).all()
        ):
            raise ValueError("Features must be finite")
        if (patch_tokens.norm(dim=-1) == 0).any() or (
            cls_tokens.norm(dim=-1) == 0
        ).any():
            raise ValueError("Input tokens must have nonzero norms")
        # Give every token length 1, so a dot product gives its cosine similarity.
        feature_dtype = (
            torch.float64 if patch_tokens.dtype == torch.float64 else torch.float32
        )
        return Features(
            F.normalize(patch_tokens.to(feature_dtype), dim=-1),
            F.normalize(cls_tokens.to(feature_dtype), dim=-1),
            self.grid,
        )


@dataclass(frozen=True)
class Prediction:
    """The detector's output for a batch of query images.

    patch_map shows which regions look unusual. spatial_score is its largest
    value, global_score comes from CLS matching, and image_score combines them.
    These are distances from normal references, not defect probabilities.
    """

    patch_map: Tensor  # [batch, grid height, grid width]
    image_score: Tensor  # [batch]
    spatial_score: Tensor
    global_score: Tensor

    def resize_maps(self, size: tuple[int, int]) -> Tensor:
        """Resize the maps. Image scores still come from the patch grid."""
        # A batch is simply several images processed together.
        # interpolate expects [batch, channels, height, width]. Our maps have no
        # channel axis, so [:, None] adds a size-1 axis and [:, 0] removes it later.
        # Bilinear interpolation fills the larger grid by blending nearby scores.
        return F.interpolate(
            self.patch_map[:, None], size=size, mode="bilinear", align_corners=False
        )[:, 0]


class HyperFSAD:
    """Store normal features and score new images without training."""

    def __init__(self, fusion_weight: float = 0.5, chunk_size: int = 256):
        if not 0 <= fusion_weight <= 1:
            raise ValueError("fusion_weight must lie in [0,1]")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        # fusion_weight controls how much the local score contributes (0.5 means half).
        # chunk_size limits how many query patches we match at once, to save memory.
        self.fusion_weight = fusion_weight
        self.chunk_size = chunk_size
        self._memory: Features | None = None

    # We only calculate outputs here, so PyTorch does not need to track gradients
    # (the values used to update weights during training).
    @torch.inference_mode()
    def fit(self, support: Features) -> "HyperFSAD":
        # Despite the name fit(), this does not train anything. It only prepares
        # and stores descriptions of the normal images for later comparisons.
        support_features = support.normalized()
        # Copy the normal features so changes to the original tensors cannot change our bank.
        self._memory = Features(
            support_features.patches.detach().clone(),
            support_features.cls.detach().clone(),
            support_features.grid,
        )
        return self

    @torch.inference_mode()
    def predict(self, query: Features) -> Prediction:
        if self._memory is None:
            raise RuntimeError("Call fit with normal support features first")
        # Normalize the new image descriptions the same way as the stored references.
        query_features, support_features = query.normalized(), self._memory
        if (query_features.patches.shape[0], query_features.patches.shape[-1]) != (
            support_features.patches.shape[0],
            support_features.patches.shape[-1],
        ):
            raise ValueError(
                "Query and support must share layer count and feature dimension"
            )
        if (
            query_features.patches.device != support_features.patches.device
            or query_features.patches.dtype != support_features.patches.dtype
        ):
            raise ValueError("Query and support must share device and precision")
        num_layers, batch_size, num_patches, feature_size = query_features.patches.shape
        # Start each score at zero. The loop below adds one contribution per layer.
        patch_scores = torch.zeros(
            (batch_size, num_patches),
            device=query_features.patches.device,
            dtype=query_features.patches.dtype,
        )
        global_score = torch.zeros(
            batch_size, device=patch_scores.device, dtype=patch_scores.dtype
        )
        # Each layer has its own normal memory and contributes equally to the final scores.
        for layer_index in range(num_layers):
            # reshape changes the layout without changing the feature values.
            # -1 asks PyTorch to calculate the row count. Normal images and their
            # patches become one list, but every row still holds one full feature vector.
            support_patches = support_features.patches[layer_index].reshape(
                -1, feature_size
            )
            query_patches = query_features.patches[layer_index].reshape(
                -1, feature_size
            )
            chunk_distances = []
            # Split queries to save memory, but match each one against all support patches.
            for query_chunk in query_patches.split(self.chunk_size):
                # @ is matrix multiplication; .T swaps rows and columns.
                # [query patches, features] @ [features, normal patches] produces
                # one similarity for every query/reference pair. Tokens have length 1,
                # so these dot products are cosine similarities.
                support_weights = sparsemax(query_chunk @ support_patches.T)
                # The weighted normal patches form our guess of what this patch should look like.
                reconstructed_patches = support_weights @ support_patches
                # Similarity is high for a good reconstruction. Subtract it from 1
                # to make unusual patches score higher. dim=-1 means compare along
                # the last axis (the feature values). clamp removes tiny rounding
                # errors outside [0, 2]. A zero reconstruction gets distance 1.
                chunk_distances.append(
                    (
                        1
                        - F.cosine_similarity(
                            query_chunk, reconstructed_patches, dim=-1
                        )
                    ).clamp(0, 2)
                )
            # Join the chunks in their original order, then restore the image/patch
            # axes. Dividing each layer's contribution by num_layers gives the mean.
            patch_scores += (
                torch.cat(chunk_distances).reshape(batch_size, num_patches) / num_layers
            )
            # For the whole-image score, use the most similar normal CLS token.
            nearest_cls_similarity = (
                query_features.cls[layer_index] @ support_features.cls[layer_index].T
            ).amax(dim=-1)
            global_score += (1 - nearest_cls_similarity).clamp(0, 2) / num_layers
        # Put the patch scores back into their image grid and take the highest local score.
        patch_map = patch_scores.reshape(batch_size, *query_features.grid)
        spatial_score = patch_scores.amax(dim=-1)
        # The default gives equal importance to local defects and whole-image differences.
        image_score = (
            self.fusion_weight * spatial_score + (1 - self.fusion_weight) * global_score
        )
        return Prediction(patch_map, image_score, spatial_score, global_score)
