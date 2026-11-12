# Hyper-FSAD

My PyTorch implementation of **Hyper-FSAD: Training-Free and Language-Free Few-Shot Anomaly Detection via Sparse Hyper Matching**. I built this project to understand how a frozen vision model can detect defects using only a few normal examples.

Paper: [arXiv:2605.10628v2](https://arxiv.org/abs/2605.10628v2), August 2026, by Guohuan Xie, Xin He, Dingying Fan, Siqi Li and Yun Liu.

This is an **unofficial implementation**. The core method is implemented, but I have only evaluated it on the bottle category of MVTec AD so far. It is not a reproduction of all the paper's experiments.

## What it does

The model takes a few normal images as references and compares a new image against them. It returns an image-level anomaly score and a map of the areas that look unusual.

1. Extract patch and CLS tokens from layers 3, 6, 9 and 12 of a frozen DINO encoder.
2. Store the normal reference tokens in a separate memory bank for each layer.
3. Use sparsemax to select and weight normal patches for reconstructing each query patch.
4. Calculate local anomaly scores from the cosine distance between query patches and their reconstructions.
5. Combine the maximum local score with the nearest-reference CLS distance, using equal weights.

There is no fine-tuning, text prompt or training loop. `fit()` just stores the normal features. Higher scores mean the image differs more from the normal references; they are not probabilities.

## Results so far

Both runs used **one normal bottle image**, seed **0**, and all **83 test images** (20 normal and 63 defective). The selected reference was `bottle/train/good/177.png`.

| Metric | DINOv2 ViT-B/14 | DINOv3 ViT-B/16 |
|---|---:|---:|
| Input size | 224 × 224 | 448 × 448 |
| Image AUROC | 99.37% | 99.37% |
| Image average precision | 99.80% | 99.80% |
| Image maximum F1 | 97.64% | 97.67% |
| Pixel AUROC | 98.68% | 98.73% |
| Pixel average precision | 79.86% | 85.26% |
| Pixel maximum F1 | 75.59% | 78.28% |

DINOv3 gave better pixel-level results in this run. The input sizes are different, so this does not isolate the effect of the backbone alone. AUROC is not classification accuracy, and maximum F1 is measured using the best threshold on the test set, not a threshold chosen for deployment.

The full configuration, model revisions and per-image predictions are saved in [benchmarks](benchmarks/). See [RESULTS.md](RESULTS.md) for the comparison.

## Setup

Python 3.10 or newer is required. I tested the project on Python 3.12 with CPU PyTorch.

```bash
git clone https://github.com/aditya162006/Hyper-FSAD.git
cd Hyper-FSAD
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. For a CUDA GPU, install matching PyTorch and torchvision builds from [pytorch.org](https://pytorch.org/get-started/locally/) instead of the CPU command above.

The tested package versions are listed in `requirements-tested.txt` as a record of the environment. Use `pyproject.toml` for the normal installation.

### Model weights

- **DINOv3:** `facebook/dinov3-vitb16-pretrain-lvd1689m` is the default. Request access on its [Hugging Face page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), then authenticate locally with `hf auth login` using an account with access.
- **DINOv2:** `facebook/dinov2-base` is public. Select it with `--backbone dinov2`.

Weights are downloaded on first use. You can also pass `--model /path/to/local/checkpoint`. The directory needs both the model and image processor files. Do not commit tokens or `.env` files.

## Run on your own images

Put normal reference images and query images in separate folders. Use images of the same category for each run.

```bash
hyper-fsad --support /path/to/normal-images --query /path/to/test-images \
  --backbone dinov3 --shots 1 --seed 0 --batch-size 1 \
  --device cpu --output outputs/bottle
```

The command writes:

- `support.txt`: the normal images selected for the run.
- `scores.csv`: image scores and the separate local and global scores.
- `maps/`: NumPy anomaly maps resized to the original image dimensions.

Use `--shots 2` or `--shots 4` for more references. On a GPU, use `--device cuda`; increase the batch size if memory allows.

## Evaluate on MVTec AD

Download the data from the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). The dataset and pretrained weights are not included here.

The evaluator expects the original folder layout:

```text
data/mvtec/
└── bottle/
    ├── train/good/
    ├── test/
    │   ├── good/
    │   └── broken_large/
    └── ground_truth/
        └── broken_large/
```

Other defect folders follow the same layout. Defect masks must be named `<image_stem>_mask.png`.

To repeat the DINOv3 run reported above:

```bash
hyper-fsad-evaluate --data data/mvtec --categories bottle \
  --backbone dinov3 --shots 1 --seeds 0 --batch-size 1 \
  --device cpu --output results/bottle-dinov3-1shot
```

For DINOv2, change `--backbone` to `dinov2` and use a different output folder. For multiple support selections, use `--seeds 0 1 2`. Repeat with `--shots 2` and `--shots 4` for the other shot settings. More categories can be passed after `--categories` once downloaded.

Supports are sampled only from `train/good`. Every test image is evaluated, and pixel metrics use the original image resolution. Run JSON files contain the support selection, metrics and predictions. `summary.json` averages over seeds, then over categories with equal category weights. Standard deviations use `ddof=0`.

Pixel metrics keep all scores in memory, which can use several GB. On a small CPU machine, `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` may help avoid excessive threading. Cached models can be used offline with `HF_HUB_OFFLINE=1`.

## Tests

```bash
pytest -q
```

There are 13 tests covering sparsemax, known cosine-distance examples, batch and chunk consistency, DINO token extraction, folder inference, dataset masks and evaluation metrics. Backbone integration tests use tiny random models, so the tests do not need pretrained weights or a Hugging Face account.

## Implementation notes

- DINOv3 uses the paper's default backbone, 448 × 448 resolution and layers 3/6/9/12.
- DINOv2 uses 224 × 224, matching its processor's usual output size. The paper does not give an exact resolution for this variant beyond referring to standard resolutions.
- Images are resized to a square without cropping. Intermediate block outputs are L2-normalized; register tokens are excluded from the patch bank.
- Maps use bilinear upsampling with `align_corners=False` and no smoothing. Image scores are calculated before upsampling.
- Query chunks reduce matching memory usage. Each query still uses the full support bank.

Some preprocessing details are not fully specified in the paper. These choices are documented so they can be checked against the authors' code when available.

## Still to do

- Evaluate all MVTec AD categories with several seeds and 1/2/4-shot supports.
- Add VisA, MPDD, BTAD, RESC and BraTS with the correct evaluation splits.
- Add PRO and reproduce the matching, layer, resolution, fusion and pooling ablations.
- Measure GPU runtime and memory under settings comparable to the paper.
- Add visual examples of predicted anomaly maps.

## Credits and license

The method comes from [Hyper-FSAD](https://arxiv.org/abs/2605.10628v2). The pretrained encoders come from Meta's [DINOv2](https://github.com/facebookresearch/dinov2) and [DINOv3](https://github.com/facebookresearch/dinov3) projects.

The implementation code is released under the [MIT License](LICENSE). Datasets and pretrained weights have their own licenses; this code license does not replace them. MVTec AD is distributed under CC BY-NC-SA 4.0.
