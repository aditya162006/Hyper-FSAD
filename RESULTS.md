# Initial DINOv2 and DINOv3 benchmarks

Both runs evaluate the complete MVTec AD bottle test split: 20 normal and 63 defective images, using one normal support image and seed 0.

Shared support: `bottle/train/good/177.png`. Both use layers 3/6/9/12, equal local/global fusion, full-image square resize without cropping, no map smoothing, and native-resolution pixel metrics. Runs used CPU.

| Metric | DINOv2 ViT-B/14 (224×224) | DINOv3 ViT-B/16 (448×448) |
|---|---:|---:|
| Image AUROC | 99.37% | 99.37% |
| Image average precision | 99.80% | 99.80% |
| Image maximum F1 | 97.64% | 97.67% |
| Pixel AUROC | 98.68% | 98.73% |
| Pixel average precision | 79.86% | 85.26% |
| Pixel maximum F1 | 75.59% | 78.28% |

DINOv3 uses the paper’s default backbone, resolution, and selected layers. These are single-category, single-seed measurements of an independent implementation, not a reproduction of the complete paper. The different backbone resolutions mean this is a comparison of configurations, not an isolated backbone ablation. AUROC is not percentage classification accuracy. Maximum F1 uses the best threshold on the test set and is not a calibrated deployment result. PRO and GPU speed/memory have not been evaluated.

DINOv3 model revision: `5931719e67bbdb9737e363e781fb0c67687896bc`.

Full configurations and all 83 image predictions per run are in `benchmarks/bottle-dinov2-1shot/` and `benchmarks/bottle-dinov3-1shot/`.
