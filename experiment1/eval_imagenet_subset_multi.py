import os
import argparse

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from timm.data import resolve_data_config, create_transform
from fastervit import create_model

print("=== eval_imagenet_subset_multi.py starting ===")

# -------------------------------------------------
# 0) Allow argparse.Namespace in checkpoints (PyTorch 2.6+)
#     -> needed for some FasterViT .pth.tar files
# -------------------------------------------------
torch.serialization.add_safe_globals([argparse.Namespace])

# ==========================
# 1) Global config
# ==========================

# Auto-select device: GPU if available, else CPU
if torch.cuda.is_available():
    device = "cuda"
    print("Using device: CUDA")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("Using device: CPU (no CUDA available)")

# Path to your ImageNet subset (folder with n0144..., etc.)
IMAGENET_ROOT = r"C:\Users\Rebecca Fernando\FastVit_experiments\imagenet1kvaldataset"

# Where FasterViT weights are stored
WEIGHTS_DIR = r"C:\Users\Rebecca Fernando\FastVit_experiments\weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Evaluate models 0–6
MODEL_NAMES = [
    "faster_vit_0_224",
    "faster_vit_1_224",
    "faster_vit_2_224",
    "faster_vit_3_224",
    "faster_vit_4_224",
    "faster_vit_5_224",
    "faster_vit_6_224",
]

BATCH_SIZE   = 16      # you can increase this on GPU if VRAM allows
NUM_WORKERS  = 4
MAX_SAMPLES  = 1000    # keep same as your first run for fair comparison


# ==========================
# 2) Utility functions
# ==========================

def accuracy_topk(output, target, topk=(1,)):
    """Compute top-k accuracies."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * 100.0 / batch_size).item())
        return res


def build_dataloader(transform):
    print("  -> Building dataset from:", IMAGENET_ROOT)
    dataset = datasets.ImageFolder(IMAGENET_ROOT, transform=transform)

    print("     Total images found:", len(dataset))
    if MAX_SAMPLES is not None and MAX_SAMPLES < len(dataset):
        print(f"     Using first {MAX_SAMPLES} samples as subset.")
        dataset = Subset(dataset, range(MAX_SAMPLES))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),  # good when using CUDA
    )

    if isinstance(dataset, Subset):
        num_classes = len(dataset.dataset.classes)
    else:
        num_classes = len(dataset.classes)

    print("     Num classes:", num_classes)
    return loader


def evaluate_model(model_name):
    # ----------------- load model -----------------
    model_path = os.path.join(WEIGHTS_DIR, f"{model_name}.pth.tar")
    print("\n========================================")
    print(f" Evaluating model: {model_name}")
    print("  -> Using weights:", model_path)

    model = create_model(
        model_name,
        pretrained=True,
        model_path=model_path,
    ).to(device)
    model.eval()

    # ----------------- data config -----------------
    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**cfg)
    print("  Data config:", cfg)

    # ----------------- dataloader ------------------
    loader = build_dataloader(transform)

    # ----------------- eval loop -------------------
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    print("  -> Running evaluation...")
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)  # [B, 1000]
            top1, top5 = accuracy_topk(outputs, targets, topk=(1, 5))

            bs = images.size(0)
            total_top1 += top1 * bs
            total_top5 += top5 * bs
            total_samples += bs

    top1_avg = total_top1 / total_samples
    top5_avg = total_top5 / total_samples

    # ----------------- summary ---------------------
    print(" ---------- RESULT ----------")
    print(f"  Samples:   {total_samples}")
    print(f"  Top-1 Acc: {top1_avg:.2f}%")
    print(f"  Top-5 Acc: {top5_avg:.2f}%")
    print(" ============================\n")

    return top1_avg, top5_avg, total_samples


# ==========================
# 3) Main
# ==========================

if __name__ == "__main__":
    print("Main block started.")
    all_results = {}
    for name in MODEL_NAMES:
        try:
            top1, top5, n = evaluate_model(name)
            all_results[name] = (top1, top5, n)
        except Exception as e:
            print(f" !! Failed on {name}: {e}")

    print("\n\n========= SUMMARY OVER ALL MODELS =========")
    for name, (top1, top5, n) in all_results.items():
        print(f"{name:18s} | N={n:4d} | Top-1={top1:6.2f}% | Top-5={top5:6.2f}%")
    print("===========================================")
