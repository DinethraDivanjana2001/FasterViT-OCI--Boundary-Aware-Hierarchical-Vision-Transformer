# Experiment 2 — FasterViT on CIFAR-10

This folder fine-tunes FasterViT variants on CIFAR-10 and compares them to Swin-Tiny, including accuracy curves, confusion matrices, qualitative samples, and throughput measurements.

## Files
- [experiment2/fastervit_experiment2_cifar10V1.ipynb](experiment2/fastervit_experiment2_cifar10V1.ipynb) — Baseline: full fine-tune `faster_vit_0_224` on CIFAR-10 (ImageNet-style transforms, 5–10 epochs), saves `fastervit0_cifar10_best.pth`.
- [experiment2/fastervit_experiment2_cifar10V2.ipynb](experiment2/fastervit_experiment2_cifar10V2.ipynb) — Expanded study:
  - Full FT `faster_vit_0_224` and head-only `faster_vit_1_224` with plots + confusion matrices.
  - Head-only FasterViT-4 pretrain checkpoints (`faster_vit_4_21k_224` at 224px, `faster_vit_4_21k_384` at 384px) with AMP; saves `fvit4_224_headonly_cifar10.pth` and `fvit4_384_headonly_cifar10.pth`.
  - Swin-Tiny baseline fine-tune (5 epochs) for accuracy and confusion-matrix comparison vs FasterViT-0; saves `swin_cifar10_best.pth` and another `fastervit0_cifar10_best.pth` copy.
  - Throughput benchmark (dummy data, 224×224, batch 64) comparing FasterViT-0 vs Swin-Tiny.

## Requirements
- Python 3.10+ with PyTorch; CUDA GPU strongly recommended.
- Python packages: `fastervit`, `timm`, `torchvision`, `matplotlib`, `seaborn`, `scikit-learn`, `torchmetrics` (only for V2 comparison cell), `numpy`.
- CIFAR-10 downloads automatically to `./data/`. Hugging Face downloads for FasterViT-4 weights (21K pretrain) occur inside V2.

## Quick Start
1. Open either notebook in VS Code / Jupyter.
2. Run the first cell to install `fastervit` and `timm` (and `seaborn`/`torchmetrics` where present).
3. Ensure GPU is selected (`device` prints). Reduce `BATCH_SIZE` if you hit OOM (e.g., 64 → 32 for 224px, 8 for 384px).
4. Execute cells in order. Best checkpoints are saved alongside the notebook. Training uses ImageNet-style transforms via `create_transform` from each model’s pretrained config.

## Notebook Highlights
- **V1**: Single-model pipeline (FasterViT-0). Full fine-tune, cosine LR scheduler, accuracy printouts each epoch, checkpoint on best test accuracy, and a quick visualization of preprocessed CIFAR-10 samples.
- **V2**:
  - Shared helpers for training/eval/plotting; seeds fixed to 42.
  - FasterViT-0 full FT + FasterViT-1 head-only for a light compute scaling comparison.
  - FasterViT-4 (21K pretrain) head-only at 224px and 384px using AMP; lower batch sizes to fit memory.
  - Swin-Tiny vs FasterViT-0 accuracy + confusion matrices + sample predictions.
  - Throughput micro-benchmark mirroring Experiment 1 style (images/sec, speedup ratio printed).

## Tips
- On Windows, set `num_workers=0` if DataLoader workers cause issues.
- If you load checkpoints that rely on `argparse.Namespace`, keep the `torch.serialization.add_safe_globals([argparse.Namespace])` lines present in the notebooks.
- To resume or reuse weights, point `load_state_dict` to the saved `.pth` files in this folder.
